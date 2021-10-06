#!/usr/bin/env python
# coding:utf-8
import argparse
import glob
import logging
import os
import sys
import time
import json
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import random
import numpy as np
import datetime
import penman

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from callbacks import Seq2SeqLoggingCallback, get_checkpoint_callback, get_early_stopping_callback
from transformers import MBartTokenizer, AutoConfig

# from dataset import Seq2SeqDataModule
from dataset_new import Seq2SeqDataModule
# from dataset_new import Seq2SeqDataModule
from spring_amr.tokenization_bart import AMRBartTokenizer, PENMANBartTokenizer


from utils import (
    ROUGE_KEYS,
    LegacySeq2SeqDataset,
    Data2SeqDataset,
    Seq2SeqDataset,
    assert_all_frozen,
    calculate_bleu,
    calculate_rouge,
    calculate_smatch,
    flatten_list,
    freeze_embeds,
    freeze_params,
    get_git_info,
    label_smoothed_nll_loss,
    lmap,
    pickle_save,
    save_git_info,
    save_json,
    use_task_specific_params,
    build_vocab,
)


from utils_graph2text import (
    convert_text,
    eval_meteor,
    eval_bleu_sents,
    eval_bleu_sents_tok,
    eval_chrf,
)

# need the parent dir module
sys.path.insert(2, str(Path(__file__).resolve().parents[1]))
from lightning_base import BaseTransformer, add_generic_args, generic_train  # noqa


logger = logging.getLogger(__name__)


def setup_seed(seed):
    # print(f"Setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SummarizationModule(BaseTransformer):
    mode = "summarization"
    loss_names = ["loss"]
    metric_names = ROUGE_KEYS
    default_val_metric = "rouge2"

    def __init__(self, hparams, **kwargs):
        if hparams.sortish_sampler and hparams.gpus > 1:
            hparams.replace_sampler_ddp = False
        elif hparams.max_tokens_per_batch is not None:
            if hparams.gpus > 1:
                raise NotImplementedError("Dynamic Batch size does not work for multi-gpu training")
            if hparams.sortish_sampler:
                raise ValueError(
                    "--sortish_sampler and --max_tokens_per_batch may not be used simultaneously"
                )
        super().__init__(hparams, num_labels=None, num_rels=0, mode=self.mode, **kwargs)
        # use_task_specific_params(self.model, "summarization")
        save_git_info(self.hparams.output_dir)
        self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        pickle_save(self.hparams, self.hparams_save_path)
        self.step_count = -2
        self.metrics = defaultdict(list)
        self.model_type = self.config.model_type
        self.vocab_size = (
            self.config.tgt_vocab_size if self.model_type == "fsmt" else self.config.vocab_size
        )
        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            prefix=self.model.config.prefix or "",
        )
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.val_max_target_length,
            "test": self.hparams.test_max_target_length,
        }
        assert (
            self.target_lens["train"] <= self.target_lens["val"]
        ), f"target_lens: {self.target_lens}"
        assert (
            self.target_lens["train"] <= self.target_lens["test"]
        ), f"target_lens: {self.target_lens}"

        if self.hparams.freeze_embeds:
            freeze_embeds(self.model)
        if self.hparams.freeze_encoder:
            freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())

        self.hparams.git_sha = get_git_info()["repo_sha"]
        self.num_workers = hparams.num_workers
        self.decoder_start_token_id = None  # default to config

        self.already_saved_batch = False
        self.eval_beams = (
            self.model.config.num_beams
            if self.hparams.eval_beams is None
            else self.hparams.eval_beams
        )
        if self.hparams.eval_max_gen_length is not None:
            self.eval_max_length = self.hparams.eval_max_gen_length
        else:
            self.eval_max_length = self.model.config.max_length
        self.val_metric = (
            self.default_val_metric if self.hparams.val_metric is None else self.hparams.val_metric
        )

    def save_readable_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        """A debugging utility"""

        readable_batch = {
            k: self.tokenizer.batch_decode(v.tolist())
            if v is not None and "mask" not in k and "rel" not in k and "dis" not in k
            else v.shape
            if v is not None
            else None
            for k, v in batch.items()
        }

        save_json(readable_batch, Path(self.output_dir) / "text_batch.json")

        tb = {}
        for k, v in batch.items():
            if v is not None:
                tb[k] = v.tolist()
            else:
                tb[k] = v

        save_json(tb, Path(self.output_dir) / "tok_batch.json")

        self.already_saved_batch = True
        return readable_batch

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def _step(self, batch: dict) -> Tuple:
        pad_token_id = self.tokenizer.pad_token_id
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        decoder_input_ids, tgt_ids = batch["decoder_input_ids"], batch["labels"]
        # y = batch["labels"]
        # decoder_input_ids = y[:, :-1].contiguous()
        # tgt_ids = y[:, 1:].clone()

        if (
            not self.already_saved_batch
        ):  # This would be slightly better if it only happened on rank zero
            batch["decoder_input_ids"] = decoder_input_ids
            self.save_readable_batch(batch)

        outputs = self(
            src_ids,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            return_dict=False,
        )
        # print('Outputs:', outputs)
        lm_logits = outputs[0]
        if self.hparams.label_smoothing == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
            assert lm_logits.shape[-1] == self.vocab_size
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, tgt_ids, self.hparams.label_smoothing, ignore_index=pad_token_id
            )
        return (loss,)

    @property
    def pad(self) -> int:
        return self.tokenizer.pad_token_id

    def training_step(self, batch, batch_idx) -> Dict:
        loss_tensors = self._step(batch)

        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        # tokens per batch
        logs["tpb"] = batch["input_ids"].ne(self.pad).sum() + batch["labels"].ne(self.pad).sum()
        logs["bs"] = batch["input_ids"].shape[0]
        logs["src_pad_tok"] = batch["input_ids"].eq(self.pad).sum()
        logs["src_pad_frac"] = batch["input_ids"].eq(self.pad).float().mean()
        # print("loss_tensors:", loss_tensors)
        # print("lr_scheduler:", self.trainer.lr_schedulers[0]["scheduler"].get_lr()[0])
        self.log("train_loss", loss_tensors[0].item(), prog_bar=True)
        self.log("lr", self.trainer.lr_schedulers[0]["scheduler"].get_lr()[0], prog_bar=True)
        return {"loss": loss_tensors[0], "log": logs}

    def training_epoch_end(self, outputs, prefix="train") -> Dict:
        # print('train Ouputs:', outputs)
        # losses = {"loss": torch.stack([x["loss"] for x in outputs]).mean()}
        losses = {k: torch.stack([x[k] for x in outputs]).mean().item() for k in self.loss_names}
        loss = losses["loss"]
        self.metrics["training"].append(losses)

    def validation_step(self, batch, batch_idx) -> Dict:
        return self._generative_step(batch)

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        print(f"Generating Kwargs: Num_beam: {self.eval_beams}, Max_len: {self.eval_max_length}")
        self.step_count += 1
        outputs = self.all_gather(outputs)
        # print('Gathered outputs', outputs)
        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        loss = losses["loss"]
        generative_metrics = {
            k: torch.stack([x[k] for x in outputs]).mean().item() for k in ["gen_time", "gen_len"]
        }

        generative_metrics.update({k: v.item() for k, v in losses.items()})
        losses.update(generative_metrics)
        all_metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        all_metrics["step_count"] = self.step_count
        # print('all_metrics:', all_metrics)
        # self.metrics[prefix].append(all_metrics)  # callback writes this to self.metrics_save_path
        # print("Len(outputs)", len(outputs))
        preds = [[[itm.item() for itm in y] for y in x["preds"]] for x in outputs]

        preds_new = []
        for iter in preds:
            for ith_batch in iter:
                preds_new.append(ith_batch)
        preds = preds_new

        lin_sentences = []
        for idx, tokens_same_source in enumerate(preds):
            # print("token_same_source", tokens_same_source)
            tokens_same_source_ = [self.tokenizer.bos_token_id] + tokens_same_source[1:]
            lin_sentences.append(str(idx) + " " + self.tokenizer.decode(tokens_same_source_).strip())
        json.dump(
            lin_sentences,
            open(f"{self.hparams.output_dir}/dev-nodes.json", "w", encoding="utf-8"),
            indent=4,
        )

        graphs = []
        for idx in range(len(preds)):
            graphs_same_source = []
            graphs.append(graphs_same_source)
            ith_pred = preds[idx]
            graph, status, (lin, backr) = self.tokenizer.decode_amr(
                ith_pred, restore_name_ops=False
            )
            graph.status = status
            graph.nodes = lin
            graph.backreferences = backr
            graph.tokens = ith_pred
            graphs_same_source.append(graph)

        graphs_same_source[:] = tuple(
            zip(*sorted(enumerate(graphs_same_source), key=lambda x: (x[1].status.value, x[0])))
        )[1]
        idx = 0
        for gps in graphs:
            for gp in gps:
                # metadata = gg.metadata.copy()
                metadata = {}
                metadata["id"] = str(idx)
                metadata["annotator"] = "bart-amr"
                metadata["date"] = str(datetime.datetime.now())
                if "save-date" in metadata:
                    del metadata["save-date"]
                gp.metadata = metadata
                idx += 1
        # print("Before Penman Encoding")
        pieces = [penman.encode(g[0]) for g in graphs]
        # print("After Penman Encoding")
        val_outputs_folder = "val_outputs"
        os.system("mkdir -p " + os.path.join(self.hparams.output_dir, val_outputs_folder))

        if "preds" in outputs[0]:
            output_test_predictions_file = os.path.join(
                self.hparams.output_dir,
                val_outputs_folder,
                "validation_predictions_" + str(self.step_count) + ".txt",
            )
            # write predictions and targets for later rouge evaluation.
            with open(output_test_predictions_file, "w") as p_writer:
                p_writer.write("\n\n".join(pieces))
            try:
                smatch_score = calculate_smatch(
                    self.hparams.data_dir + f"/{prefix}.amr", output_test_predictions_file
                )
            except AttributeError:
                smatch_score = {"smatch": 0.0}
                
            rank_zero_info("number epoch: %s", self.step_count)
            rank_zero_info("%s smatch_info_bpe: %s", self.step_count, smatch_score)

        all_metrics[f"{prefix}_avg_smatch"] = smatch_score["smatch"]
        metric_tensor: torch.FloatTensor = torch.tensor(smatch_score["smatch"]).type_as(loss)
        self.metrics[prefix].append(all_metrics)  # callback writes this to self.metrics_save_path
        self.log_dict(all_metrics, sync_dist=True)
        return {
            "log": all_metrics,
            "preds": preds,
            f"{prefix}_loss": loss,
            f"{prefix}_{self.val_metric}": metric_tensor,
        }

    def _generative_step(self, batch: dict) -> dict:
        t0 = time.time()
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]

        generated_ids = self.model.generate(
            src_ids,
            attention_mask=src_mask,
            use_cache=True,
            decoder_start_token_id=self.decoder_start_token_id,
            num_beams=self.eval_beams,
            no_repeat_ngram_size=0,
            min_length=0,
            max_length=self.eval_max_length,
            length_penalty=1.0,
        )

        gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
        preds = [itm.tolist() for itm in generated_ids]
        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        summ_len = np.mean(lmap(len, generated_ids))
        base_metrics.update(
            gen_time=gen_time, gen_len=summ_len, preds=preds,
        )
        return base_metrics

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")

    def get_datamodule(self):
        return Seq2SeqDataModule(
            data_dir=self.hparams.data_dir,
            tokenizer=self.tokenizer,
            n_obs=self.n_obs,
            target_lens=self.target_lens,
            max_source_length=self.hparams.max_source_length,
            max_target_length=self.hparams.max_target_length,
            train_batch_size=self.hparams.train_batch_size,
            val_batch_size=self.hparams.eval_batch_size,
            dataloader_num_workers=self.hparams.num_workers,
            prefix=" ",
        )

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        add_generic_args(parser, root_dir)
        parser.add_argument(
            "--max_source_length",
            default=1024,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=56,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--val_max_target_length",
            default=142,  # these defaults are optimized for CNNDM. For xsum, see README.md.
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--test_max_target_length",
            default=142,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument("--freeze_encoder", action="store_true")
        parser.add_argument("--freeze_embeds", action="store_true")
        parser.add_argument("--sortish_sampler", action="store_true", default=False)
        parser.add_argument("--max_tokens_per_batch", type=int, default=None)
        parser.add_argument(
            "--logger_name",
            type=str,
            choices=["default", "wandb", "wandb_shared"],
            default="default",
        )
        parser.add_argument(
            "--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all."
        )
        parser.add_argument(
            "--n_val", type=int, default=-1, required=False, help="# examples. -1 means use all."
        )
        parser.add_argument(
            "--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all."
        )
        parser.add_argument(
            "--task",
            type=str,
            default="summarization",
            required=False,
            help="# examples. -1 means use all.",
        )
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument("--src_lang", type=str, default="", required=False)
        parser.add_argument("--tgt_lang", type=str, default="", required=False)
        parser.add_argument("--eval_beams", type=int, default=None, required=False)
        parser.add_argument("--checkpoint", type=str, default=None, required=False)
        parser.add_argument(
            "--val_metric",
            type=str,
            default=None,
            required=False,
            choices=["bleu", "rouge2", "loss", None],
        )
        parser.add_argument(
            "--eval_max_gen_length",
            type=int,
            default=None,
            help="never generate more than n tokens",
        )
        parser.add_argument(
            "--save_top_k", type=int, default=1, required=False, help="How many checkpoints to save"
        )
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            required=False,
            help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
        )
        parser.add_argument(
            "--smart_init", action="store_true", default=False, help="smart init new embeddings."
        )
        return parser


class TranslationModule(SummarizationModule):
    mode = "translation"
    loss_names = ["loss"]
    metric_names = ["bleu"]
    default_val_metric = "bleu"

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.dataset_kwargs["src_lang"] = hparams.src_lang
        self.dataset_kwargs["tgt_lang"] = hparams.tgt_lang

    def calc_generative_metrics(self, preds, target) -> dict:
        return calculate_bleu(preds, target)


class Graph2TextModule(SummarizationModule):
    mode = "Bart"
    loss_names = ["loss"]
    metric_names = ["bleu"]
    default_val_metric = "bleu"

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        rank_zero_info("parameters %s", hparams)

    def calc_generative_metrics(self, preds, target) -> dict:
        return calculate_bleu(preds, target)


class AMRparsingModule(SummarizationModule):
    mode = "Bart"
    loss_names = ["loss"]
    metric_names = ["smatch"]
    default_val_metric = "smatch"

    def __init__(self, hparams, **kwargs):
        config = AutoConfig.from_pretrained(hparams.model_name_or_path)
        amr_tokenizer = PENMANBartTokenizer.from_pretrained(
            "facebook/bart-large",
            collapse_name_ops=False,
            use_pointer_tokens=True,
            raw_graph=False,
        )
        super().__init__(hparams, config=config, tokenizer=amr_tokenizer, **kwargs)
        rank_zero_info("parameters %s", hparams)
        # self.decoder_start_token_id = config.bos_token_id
        self.decoder_start_token_id = amr_tokenizer.amr_bos_token_id

    def calc_generative_metrics(self, preds, target) -> dict:
        return calculate_smatch(preds, target)


def main(args, model=None) -> SummarizationModule:
    Path(args.output_dir).mkdir(exist_ok=True)
    if len(os.listdir(args.output_dir)) > 3 and args.do_train:
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(args.output_dir)
        )
    if model is None:
        if "summarization" in args.task:
            model: SummarizationModule = SummarizationModule(args)
        elif "translation" in args.task:
            model: SummarizationModule = TranslationModule(args)
        elif "graph2text" in args.task:
            model: SummarizationModule = Graph2TextModule(args)
        elif "amrparsing" in args.task:
            model: SummarizationModule = AMRparsingModule(args)

    print(model.model)
    print(
        "num. model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.model.parameters()),
            sum(p.numel() for p in model.model.parameters() if p.requires_grad),
        )
    )

    dataset = Path(args.data_dir).name
    if (
        args.logger_name == "default"
        or args.fast_dev_run
        or str(args.output_dir).startswith("/tmp")
        or str(args.output_dir).startswith("/var")
    ):
        logger = True  # don't pollute wandb logs unnecessarily
    elif args.logger_name == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        project = os.environ.get("WANDB_PROJECT", dataset)
        logger = WandbLogger(name=model.output_dir.name, project=project)

    elif args.logger_name == "wandb_shared":
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(name=model.output_dir.name, project=f"hf_{dataset}")

    if args.early_stopping_patience >= 0:
        es_callback = get_early_stopping_callback(model.val_metric, args.early_stopping_patience)
    else:
        es_callback = False

    lower_is_better = args.val_metric == "loss"
    datamodule = model.get_datamodule()

    trainer: pl.Trainer = generic_train(
        model,
        args,
        logging_callback=Seq2SeqLoggingCallback(),
        checkpoint_callback=get_checkpoint_callback(
            args.output_dir, model.val_metric, args.save_top_k, lower_is_better
        ),
        early_stopping_callback=es_callback,
        logger=logger,
    )
    if args.do_train:
        trainer.fit(model, datamodule=datamodule)

    pickle_save(model.hparams, model.output_dir / "hparams.pkl")

    if not args.do_predict:
        return model

    model.hparams.test_checkpoint = ""

    if not args.checkpoint:
        checkpoints = list(
            sorted(glob.glob(os.path.join(args.output_dir, "*.ckpt"), recursive=True))
        )
    else:
        checkpoints = [args.checkpoint]

    print("checkpoints:", checkpoints)
    if checkpoints:
        model.hparams.test_checkpoint = checkpoints[-1]
        # trainer.resume_from_checkpoint = checkpoints[-1]
        if args.do_predict and not args.do_train:

            checkpoint = checkpoints[-1]
            print('Evaluation on checkpint', checkpoint)
            # trainer.test(model, ckpt_path=checkpoints[-1], datamodule=datamodule)
            print("Valid Set ...")
            trainer.validate(model, datamodule=datamodule)
            print("Test Set ...")
            trainer.test(model, datamodule=datamodule)
            return model

    trainer.logger.log_hyperparams(model.hparams)

    # test() without a model tests using the best checkpoint automatically
    trainer.test()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SummarizationModule.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()
    setup_seed(args.seed)
    main(args)

