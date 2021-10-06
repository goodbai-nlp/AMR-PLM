# coding:utf-8
import itertools
import json
import linecache
import math
import os
import sys
import tqdm
import datetime
import pickle
import socket
import smatch
import postprocessing
import penman
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple, Union
from collections import Counter

# import git
import numpy as np
import torch
import torch.distributed as dist
from rouge_score import rouge_scorer, scoring
from sacrebleu import corpus_bleu
from torch import nn
from torch.utils.data import Dataset, Sampler

from transformers import BartTokenizer, EvalPrediction, PreTrainedTokenizer, T5Tokenizer
from transformers.file_utils import cached_property

# from transformers.modeling_bart import shift_tokens_right

# from transformers.models.bart.modeling_bart import shift_tokens_right

from pytorch_lightning.utilities import rank_zero_info
import re

nltk = None

import nltk

try:
    from fairseq.data.data_utils import batch_by_size

    FAIRSEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    FAIRSEQ_AVAILABLE = False


def add_newline_to_end_of_each_sentence(x: str) -> str:
    """This was added to get rougeLsum scores matching published rougeL scores for BART and PEGASUS."""
    re.sub("<n>", "", x)  # remove pegasus newline char
    assert nltk, "nltk must be installed to separate newlines between sentences. (pip install nltk)"
    return "\n".join(nltk.sent_tokenize(x))


def build_vocab(file, max_vocab=5000, undirectional=False):
    with open(file, "r", encoding="utf-8") as fin:
        vocab = ["<pad>", "<unk>"]
        rel_vocab = Counter()
        for line in fin:
            ith_data = line.strip().split()
            if undirectional:
                tmp = line.strip().split()
                ith_data = [tok.split(":")[-1] for tok in tmp]
            # print(ith_data)
            rel_vocab.update(ith_data)
        # print('rel_vocab', rel_vocab)
        for rel, _ in rel_vocab.most_common(min(len(rel_vocab), max_vocab)):
            vocab.append(rel)
        vocab2id = {itm: idx for idx, itm in enumerate(vocab)}
    print("ALL {} relations".format(len(vocab2id)))
    # print('vocab:', vocab2id)
    return vocab2id


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def calculate_bleu(output_lns, refs_lns, **kwargs) -> dict:
    """Uses sacrebleu's corpus_bleu implementation."""
    return {"bleu": round(corpus_bleu(output_lns, [refs_lns], **kwargs).score, 4)}



def build_compute_metrics_fn(
    task_name: str, tokenizer: PreTrainedTokenizer
) -> Callable[[EvalPrediction], Dict]:
    def non_pad_len(tokens: np.ndarray) -> int:
        return np.count_nonzero(tokens != tokenizer.pad_token_id)

    def decode_pred(pred: EvalPrediction) -> Tuple[List[str], List[str]]:
        pred_str = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
        pred_str = lmap(str.strip, pred_str)
        label_str = lmap(str.strip, label_str)
        return pred_str, label_str

    def summarization_metrics(pred: EvalPrediction) -> Dict:
        pred_str, label_str = decode_pred(pred)
        rouge: Dict = calculate_rouge(pred_str, label_str)
        summ_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
        rouge.update({"gen_len": summ_len})
        return rouge

    def translation_metrics(pred: EvalPrediction) -> Dict:
        pred_str, label_str = decode_pred(pred)
        bleu: Dict = calculate_bleu(pred_str, label_str)
        gen_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
        bleu.update({"gen_len": gen_len})
        return bleu

    compute_metrics_fn = (
        summarization_metrics if "summarization" in task_name else translation_metrics
    )
    return compute_metrics_fn


def trim_batch(input_ids, pad_token_id, attention_mask=None):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


# def trim_batch_matrix(
#     input_ids,
#     pad_token_id
# ):
#     """input are """
#     max_seq_len = max


class AbstractSeq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        prefix="",
        **dataset_kwargs,
    ):
        super().__init__()
        self.src_file = Path(data_dir).joinpath(type_path + ".src")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".tgt")
        self.len_file = Path(data_dir).joinpath(type_path + ".len")
        if os.path.exists(self.len_file):
            self.src_lens = pickle_load(self.len_file)
            self.used_char_len = False
        else:
            self.src_lens = self.get_char_lens(self.src_file)
            self.used_char_len = True
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix is not None else ""

        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.dataset_kwargs = dataset_kwargs
        dataset_kwargs.update(
            {"add_prefix_space": True} if isinstance(self.tokenizer, BartTokenizer) else {}
        )

    def __len__(self):
        return len(self.src_lens)

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    @cached_property
    def tgt_lens(self):
        """Length in characters of target documents"""
        return self.get_char_lens(self.tgt_file)

    def make_sortish_sampler(self, batch_size, distributed=False, shuffle=True, **kwargs):
        if distributed:
            return DistributedSortishSampler(self, batch_size, shuffle=shuffle, **kwargs)
        else:
            return SortishSampler(self.src_lens, batch_size, shuffle=shuffle)

    def make_dynamic_sampler(self, max_tokens_per_batch=1024, **kwargs):
        assert FAIRSEQ_AVAILABLE, "Dynamic batch size requires `pip install fairseq`"
        assert (
            not self.used_char_len
        ), "You must call  python make_len_file.py before calling make_dynamic_sampler"
        sorted_indices = list(self.make_sortish_sampler(1024, shuffle=False))

        def num_tokens_in_example(i):
            return min(self.src_lens[i], self.max_target_length)

        # call fairseq cython function
        batch_sampler: List[List[int]] = batch_by_size(
            sorted_indices,
            num_tokens_fn=num_tokens_in_example,
            max_tokens=max_tokens_per_batch,
            required_batch_size_multiple=64,
        )
        shuffled_batches = [
            batch_sampler[i] for i in np.random.permutation(range(len(batch_sampler)))
        ]
        # move the largest batch to the front to OOM quickly (uses an approximation for padding)
        approximate_toks_per_batch = [
            max(self.src_lens[i] for i in batch) * len(batch) for batch in shuffled_batches
        ]
        largest_batch_idx = np.argmax(approximate_toks_per_batch)
        shuffled_batches[0], shuffled_batches[largest_batch_idx] = (
            shuffled_batches[largest_batch_idx],
            shuffled_batches[0],
        )
        return shuffled_batches

    def __getitem__(self, item):
        raise NotImplementedError("You must implement this")

    def collate_fn(self, batch):
        raise NotImplementedError("You must implement this")


class LegacySeq2SeqDataset(AbstractSeq2SeqDataset):
    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        """Call tokenizer on src and tgt_lines"""

        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        source_inputs = self.encode_line(self.tokenizer, source_line, self.max_source_length)
        target_inputs = self.encode_line(self.tokenizer, tgt_line, self.max_target_length)

        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "labels": target_ids,
        }

    def encode_line(self, tokenizer, line, max_length, pad_to_max_length=True, return_tensors="pt"):
        """Only used by LegacyDataset"""
        return tokenizer(
            [line],
            max_length=max_length,
            padding="max_length" if pad_to_max_length else None,
            truncation=True,
            return_tensors=return_tensors,
            **self.dataset_kwargs,
        )

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        target_ids = torch.stack([x["labels"] for x in batch])
        pad_token_id = self.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": y,
        }
        return batch


class Data2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        rel_vocab,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        prefix="",
        use_rel=False,
        use_path_rel=False,
        **dataset_kwargs,
    ):
        super().__init__()
        self.use_global = True
        self.use_rel = use_rel
        self.use_path = use_path_rel
        self.src_file = Path(data_dir).joinpath(type_path + ".concept.spm")
        if self.use_rel:
            self.rel_file = Path(data_dir).joinpath(type_path + ".rel.mat")
            if self.use_path:
                self.rel_file = Path(data_dir).joinpath(type_path + ".path")

        self.dis_file = Path(data_dir).joinpath(type_path + ".dis")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.len_file = Path(data_dir).joinpath(type_path + ".len")

        self.undirectional = False
        self.rel_vocab = rel_vocab
        if os.path.exists(self.len_file):
            self.src_lens = pickle_load(self.len_file)
            self.used_char_len = False
        else:
            self.src_lens = self.get_char_lens(self.src_file)
            self.used_char_len = True

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix is not None else ""
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id

        self.pad_rel_id = self.rel_vocab.get("<pad>")
        self.non_rel_id = self.rel_vocab.get("None")

        self.dataset_kwargs = dataset_kwargs
        dataset_kwargs.update(
            {"add_prefix_space": True} if isinstance(self.tokenizer, BartTokenizer) else {}
        )

    def __len__(self):
        return len(self.src_lens)

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    @cached_property
    def tgt_lens(self):
        """Length in characters of target documents"""
        return self.get_char_lens(self.tgt_file)

    def make_sortish_sampler(self, batch_size, distributed=False, shuffle=True, **kwargs):
        if distributed:
            return DistributedSortishSampler(self, batch_size, shuffle=shuffle, **kwargs)
        else:
            return SortishSampler(self.src_lens, batch_size, shuffle=shuffle)

    def make_dynamic_sampler(self, max_tokens_per_batch=1024, **kwargs):
        assert FAIRSEQ_AVAILABLE, "Dynamic batch size requires `pip install fairseq`"
        assert (
            not self.used_char_len
        ), "You must call  python make_len_file.py before calling make_dynamic_sampler"
        sorted_indices = list(self.make_sortish_sampler(1024, shuffle=False))

        def num_tokens_in_example(i):
            return min(self.src_lens[i], self.max_target_length)

        # call fairseq cython function
        batch_sampler: List[List[int]] = batch_by_size(
            sorted_indices,
            num_tokens_fn=num_tokens_in_example,
            max_tokens=max_tokens_per_batch,
            required_batch_size_multiple=64,
        )
        shuffled_batches = [
            batch_sampler[i] for i in np.random.permutation(range(len(batch_sampler)))
        ]
        # move the largest batch to the front to OOM quickly (uses an approximation for padding)
        approximate_toks_per_batch = [
            max(self.src_lens[i] for i in batch) * len(batch) for batch in shuffled_batches
        ]
        largest_batch_idx = np.argmax(approximate_toks_per_batch)
        shuffled_batches[0], shuffled_batches[largest_batch_idx] = (
            shuffled_batches[largest_batch_idx],
            shuffled_batches[0],
        )
        return shuffled_batches

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        """Call tokenizer on src and tgt_lines"""

        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip()
        if self.use_global:
            source_line = source_line + " ."
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        if self.use_rel:
            rel_line = linecache.getline(str(self.rel_file), index).rstrip("\n")
        dis_line = linecache.getline(str(self.dis_file), index).rstrip("\n")

        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        if self.use_rel:
            assert rel_line, f"empty rel line for index {index}"
        assert dis_line, f"empty dis line for index {index}"

        source_inputs, raw_src_len = self.encode_line2(
            self.tokenizer, source_line, self.max_source_length, pad_to_max_length=True
        )
        # src_len = source_inputs["input_ids"].size(1)
        if self.use_rel:
            rel_inputs, raw_rel_len = self.encode_rel(
                vocab=self.rel_vocab,
                line=rel_line,
                max_length=self.max_source_length,
                pad_to_max_length=True,
            )

        dis_inputs = self.encode_dis(dis_line, self.max_source_length)

        target_inputs = self.encode_line(self.tokenizer, tgt_line, self.max_target_length)

        if self.use_rel:
            assert (
                raw_src_len - 1 == raw_rel_len
            ), "Inconsistent size: src_ids: {}, rel_matrix:{}".format(raw_src_len - 1, raw_rel_len)

        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        if self.use_rel:
            rel_ids = rel_inputs.type_as(source_ids)
        dis_ids = dis_inputs.type_as(source_ids)
        src_mask = source_inputs["attention_mask"].squeeze()  #

        return {
            "input_ids": source_ids,
            "rel_ids": rel_ids if self.use_rel else None,
            "dis_ids": dis_ids,
            "attention_mask": src_mask,
            "labels": target_ids,
        }

    def encode_line(self, tokenizer, line, max_length, pad_to_max_length=True, return_tensors="pt"):
        """Only used by LegacyDataset"""
        return tokenizer(
            [line],
            max_length=max_length,
            padding="max_length" if pad_to_max_length else "do_not_pad",
            truncation=True,
            return_tensors=return_tensors,
            **self.dataset_kwargs,
        )
        # return tokenizer.encode_plus(line.split(), max_length=max_length,
        #     is_split_into_words=True, padding='do_not_pad',truncation=True
        # )
        # return tokenizer.convert_tokens_to_ids(line.split())

    def encode_line2(
        self, tokenizer, line, max_length, pad_to_max_length=True, return_tensors="pt"
    ):
        """Only used by LegacyDataset"""
        raw_ids = tokenizer.convert_tokens_to_ids(line.split())
        raw_ids.append(tokenizer.eos_token_id)
        ids = [tokenizer.pad_token_id for _ in range(max_length)]
        ids[: min(len(raw_ids), max_length)] = map(int, raw_ids[: min(len(raw_ids), max_length)])
        ids_tensor = torch.Tensor(ids).long().unsqueeze(0)  # [1, seq_len]
        attn_mask = 1 - (ids_tensor == tokenizer.pad_token_id).int()
        return {"input_ids": ids_tensor, "attention_mask": attn_mask}, len(raw_ids)

    def encode_rel(self, vocab, line, max_length, pad_to_max_length=True, return_tensors="pt"):
        rel_matrix = [itm.split() for itm in line.split("\t")]
        rel_ids = [
            [
                vocab.get(tok if not self.undirectional else tok.split(":")[-1], vocab.get("<unk>"))
                for tok in row
            ]
            for row in rel_matrix
        ]
        # assert len(rel_ids) == src_len + 1, "Inconsistent rel_matrix {}x{} and src_seq: {} \n{}".format(
        #     len(rel_ids), len(rel_ids), src_len + 1, line
        # )
        row_len, col_len = len(rel_ids), len(rel_ids[0])
        assert row_len == col_len, "Invalid relation matrix: {} vs {}".format(row_len, col_len)
        if pad_to_max_length:
            pad_ids = [[vocab.get("<pad>") for __ in range(max_length)] for _ in range(max_length)]
            row_len_tmp = row_len if self.use_global else row_len - 1
            for j in range(row_len_tmp):
                rel_len_j = len(rel_ids[j]) if self.use_global else len(rel_ids[j]) - 1
                pad_ids[j][:rel_len_j] = map(int, rel_ids[j][:rel_len_j])
            return torch.tensor(pad_ids, dtype=torch.long), row_len
        return torch.tensor(rel_ids, dtype=torch.long), row_len

    def encode_dis(self, line, max_length, pad_to_max_length=True, return_tensors="pt", src_len=0):
        dis_ids = [[int(iitm) for iitm in itm.split()] for itm in line.split("\t")]
        # assert len(dis_ids) == src_len+1, "Inconsistent dis_matrix {}x{} and src_seq: {} \n{}".format(
        #     len(dis_ids), len(dis_ids), src_len+1, line
        # )

        row_len, col_len = len(dis_ids), len(dis_ids[0])
        assert row_len == col_len, "Invalid relation matrix: {} vs {}".format(row_len, col_len)
        if pad_to_max_length:
            pad_ids = [
                [9 for __ in range(max_length)] for _ in range(max_length)
            ]  # max_len:8 by default
            row_len = row_len if self.use_global else row_len - 1
            for j in range(row_len):
                rel_len_j = len(dis_ids[j]) if self.use_global else len(dis_ids[j]) - 1
                pad_ids[j][:rel_len_j] = map(int, dis_ids[j][:rel_len_j])
            return torch.tensor(pad_ids, dtype=torch.long)
        return torch.tensor(dis_ids, dtype=torch.long)

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        if self.use_rel:
            rel_ids = torch.stack([x["rel_ids"] for x in batch])
        dis = torch.stack([x["dis_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        target_ids = torch.stack([x["labels"] for x in batch])
        pad_token_id = self.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        if self.use_rel:
            rel_ids = rel_ids[
                :, : source_ids.size(1), : source_ids.size(1)
            ]  # [bsz, max_seq_len, max_seq_len]
        dis = dis[:, : source_ids.size(1), : source_ids.size(1)]  # [bsz, max_seq_len, max_seq_len]
        # rel_mask_bool = rel_ids.eq(self.pad_rel_id) ^ rel_ids.eq(self.non_rel_id)       # padding or None
        # rel_mask_bool = dis > 1
        # rel_mask = (1 - rel_mask_bool.int()).type_as(masks)                               # [bsz, seq_len, seq_len]       1: non-masked, 0: masked tokens
        # rel_mask = (dis <= 1).int().type_as(masks)
        # rel_mask = rel_ids
        batch = {
            "input_ids": source_ids,
            "rel_ids": rel_ids if self.use_rel else None,
            "dis": dis,
            "attention_mask": source_mask,
            "labels": y,
        }
        return batch


class Data2SeqDatasetSim(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        rel_vocab,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        prefix="",
        **dataset_kwargs,
    ):
        super().__init__()
        self.use_global = True
        self.src_file = Path(data_dir).joinpath(type_path + ".concept.spm")
        self.rel_file = Path(data_dir).joinpath(type_path + ".path")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.len_file = Path(data_dir).joinpath(type_path + ".len")

        self.undirectional = False
        self.rel_vocab = rel_vocab

        if os.path.exists(self.len_file):
            self.src_lens = pickle_load(self.len_file)
            self.used_char_len = False
        else:
            self.src_lens = self.get_char_lens(self.src_file)
            self.used_char_len = True

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix is not None else ""
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id

        self.pad_rel_id = self.rel_vocab.get("<pad>")
        self.non_rel_id = self.rel_vocab.get("None")

        self.dataset_kwargs = dataset_kwargs
        dataset_kwargs.update(
            {"add_prefix_space": True} if isinstance(self.tokenizer, BartTokenizer) else {}
        )

    def __len__(self):
        return len(self.src_lens)

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    @cached_property
    def tgt_lens(self):
        """Length in characters of target documents"""
        return self.get_char_lens(self.tgt_file)

    def make_sortish_sampler(self, batch_size, distributed=False, shuffle=True, **kwargs):
        if distributed:
            return DistributedSortishSampler(self, batch_size, shuffle=shuffle, **kwargs)
        else:
            return SortishSampler(self.src_lens, batch_size, shuffle=shuffle)

    def make_dynamic_sampler(self, max_tokens_per_batch=1024, **kwargs):
        assert FAIRSEQ_AVAILABLE, "Dynamic batch size requires `pip install fairseq`"
        assert (
            not self.used_char_len
        ), "You must call  python make_len_file.py before calling make_dynamic_sampler"
        sorted_indices = list(self.make_sortish_sampler(1024, shuffle=False))

        def num_tokens_in_example(i):
            return min(self.src_lens[i], self.max_target_length)

        # call fairseq cython function
        batch_sampler: List[List[int]] = batch_by_size(
            sorted_indices,
            num_tokens_fn=num_tokens_in_example,
            max_tokens=max_tokens_per_batch,
            required_batch_size_multiple=64,
        )
        shuffled_batches = [
            batch_sampler[i] for i in np.random.permutation(range(len(batch_sampler)))
        ]
        # move the largest batch to the front to OOM quickly (uses an approximation for padding)
        approximate_toks_per_batch = [
            max(self.src_lens[i] for i in batch) * len(batch) for batch in shuffled_batches
        ]
        largest_batch_idx = np.argmax(approximate_toks_per_batch)
        shuffled_batches[0], shuffled_batches[largest_batch_idx] = (
            shuffled_batches[largest_batch_idx],
            shuffled_batches[0],
        )
        return shuffled_batches

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        """Call tokenizer on src and tgt_lines"""

        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip()
        if self.use_global:
            source_line = source_line + " ."
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        rel_line = linecache.getline(str(self.rel_file), index).rstrip("\n")

        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        assert rel_line, f"empty rel line for index {index}"

        source_inputs, raw_src_len = self.encode_line2(
            self.tokenizer, source_line, self.max_source_length, pad_to_max_length=True
        )
        # src_len = source_inputs["input_ids"].size(1)

        rel_inputs, raw_rel_len = self.encode_rel(
            vocab=self.rel_vocab,
            line=rel_line,
            max_length=self.max_source_length,
            pad_to_max_length=True,
        )

        target_inputs = self.encode_line(self.tokenizer, tgt_line, self.max_target_length)

        assert (
            raw_src_len - 1 == raw_rel_len
        ), "Inconsistent size: src_ids: {}, rel_matrix:{}".format(raw_src_len - 1, raw_rel_len)

        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        rel_ids = rel_inputs.type_as(source_ids)
        src_mask = source_inputs["attention_mask"].squeeze()  #

        return {
            "input_ids": source_ids,
            "rel_ids": rel_ids if self.use_rel else None,
            "attention_mask": src_mask,
            "labels": target_ids,
        }

    def encode_line(self, tokenizer, line, max_length, pad_to_max_length=True, return_tensors="pt"):
        """Only used by LegacyDataset"""
        return tokenizer(
            [line],
            max_length=max_length,
            padding="max_length" if pad_to_max_length else "do_not_pad",
            truncation=True,
            return_tensors=return_tensors,
            **self.dataset_kwargs,
        )

    def encode_line2(
        self, tokenizer, line, max_length, pad_to_max_length=True, return_tensors="pt"
    ):
        """Only used by LegacyDataset"""
        raw_ids = tokenizer.convert_tokens_to_ids(line.split())
        raw_ids.append(tokenizer.eos_token_id)
        ids = [tokenizer.pad_token_id for _ in range(max_length)]
        ids[: min(len(raw_ids), max_length)] = map(int, raw_ids[: min(len(raw_ids), max_length)])
        ids_tensor = torch.Tensor(ids).long().unsqueeze(0)  # [1, seq_len]
        attn_mask = 1 - (ids_tensor == tokenizer.pad_token_id).int()
        return {"input_ids": ids_tensor, "attention_mask": attn_mask}, len(raw_ids)

    def encode_rel(self, vocab, line, max_length, pad_to_max_length=True, return_tensors="pt"):
        rel_matrix = [itm.split() for itm in line.split("\t")]
        rel_ids = [
            [
                vocab.get(tok if not self.undirectional else tok.split(":")[-1], vocab.get("<unk>"))
                for tok in row
            ]
            for row in rel_matrix
        ]
        # assert len(rel_ids) == src_len + 1, "Inconsistent rel_matrix {}x{} and src_seq: {} \n{}".format(
        #     len(rel_ids), len(rel_ids), src_len + 1, line
        # )
        row_len, col_len = len(rel_ids), len(rel_ids[0])
        assert row_len == col_len, "Invalid relation matrix: {} vs {}".format(row_len, col_len)
        if pad_to_max_length:
            pad_ids = [[vocab.get("<pad>") for __ in range(max_length)] for _ in range(max_length)]
            row_len_tmp = row_len if self.use_global else row_len - 1
            for j in range(row_len_tmp):
                rel_len_j = len(rel_ids[j]) if self.use_global else len(rel_ids[j]) - 1
                pad_ids[j][:rel_len_j] = map(int, rel_ids[j][:rel_len_j])
            return torch.tensor(pad_ids, dtype=torch.long), row_len
        return torch.tensor(rel_ids, dtype=torch.long), row_len

    def encode_dis(self, line, max_length, pad_to_max_length=True, return_tensors="pt", src_len=0):
        dis_ids = [[int(iitm) for iitm in itm.split()] for itm in line.split("\t")]
        # assert len(dis_ids) == src_len+1, "Inconsistent dis_matrix {}x{} and src_seq: {} \n{}".format(
        #     len(dis_ids), len(dis_ids), src_len+1, line
        # )

        row_len, col_len = len(dis_ids), len(dis_ids[0])
        assert row_len == col_len, "Invalid relation matrix: {} vs {}".format(row_len, col_len)
        if pad_to_max_length:
            pad_ids = [
                [9 for __ in range(max_length)] for _ in range(max_length)
            ]  # max_len:8 by default
            row_len = row_len if self.use_global else row_len - 1
            for j in range(row_len):
                rel_len_j = len(dis_ids[j]) if self.use_global else len(dis_ids[j]) - 1
                pad_ids[j][:rel_len_j] = map(int, dis_ids[j][:rel_len_j])
            return torch.tensor(pad_ids, dtype=torch.long)
        return torch.tensor(dis_ids, dtype=torch.long)

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        rel_ids = torch.stack([x["rel_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        target_ids = torch.stack([x["labels"] for x in batch])
        pad_token_id = self.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)

        rel_ids = rel_ids[
            :, : source_ids.size(1), : source_ids.size(1)
        ]  # [bsz, max_seq_len, max_seq_len]
        batch = {
            "input_ids": source_ids,
            "rel_ids": rel_ids if self.use_rel else None,
            "attention_mask": source_mask,
            "labels": y,
        }
        return batch


class Seq2SeqDataset(AbstractSeq2SeqDataset):
    """A dataset that calls prepare_seq2seq_batch."""

    def __getitem__(self, index) -> Dict[str, str]:

        # print(self.dataset_kwargs['model_t'])
        # if 't5' in self.dataset_kwargs['model_t']:
        #     self.prefix = 'translate Graph to English: '
        #     print('aac')
        #     exit()

        # print('prefix', self.prefix, '0)', self.prefix, self.prefix, self.prefix, 'prefix')

        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")

        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")

        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1}

    def collate_fn(self, batch):
        """Call prepare_seq2seq_batch."""
        batch_encoding: Dict[str, torch.Tensor] = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            tgt_texts=[x["tgt_texts"] for x in batch],
            max_length=self.max_source_length,
            max_target_length=self.max_target_length,
            return_tensors="pt",
            **self.dataset_kwargs,
        ).data
        # lens = (batch_encoding['attention_mask'] == 1.).sum(dim=1).tolist()

        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])

        return batch_encoding


class Dual2SeqDataset(Dataset):
    """A dataset that calls prepare_seq2seq_batch."""

    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        prefix="",
        **dataset_kwargs,
    ):
        super().__init__()
        self.src_file = Path(data_dir).joinpath(type_path + ".src")
        self.src_graph_file = Path(data_dir).joinpath(type_path + ".amr")
        # self.src_graph_file = Path(data_dir).joinpath(type_path + ".concept")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".tgt")
        self.len_file = Path(data_dir).joinpath(type_path + ".len")

        if os.path.exists(self.len_file):
            self.src_lens = pickle_load(self.len_file)
            self.used_char_len = False
        else:
            self.src_lens = self.get_char_lens(self.src_file)
            self.used_char_len = True

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix is not None else ""

        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.dataset_kwargs = dataset_kwargs
        dataset_kwargs.update(
            {"add_prefix_space": True} if isinstance(self.tokenizer, BartTokenizer) else {}
        )

    def __len__(self):
        return len(self.src_lens)

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    @cached_property
    def tgt_lens(self):
        """Length in characters of target documents"""
        return self.get_char_lens(self.tgt_file)

    def make_sortish_sampler(self, batch_size, distributed=False, shuffle=True, **kwargs):
        if distributed:
            return DistributedSortishSampler(self, batch_size, shuffle=shuffle, **kwargs)
        else:
            return SortishSampler(self.src_lens, batch_size, shuffle=shuffle)

    def make_dynamic_sampler(self, max_tokens_per_batch=1024, **kwargs):
        assert FAIRSEQ_AVAILABLE, "Dynamic batch size requires `pip install fairseq`"
        assert (
            not self.used_char_len
        ), "You must call  python make_len_file.py before calling make_dynamic_sampler"
        sorted_indices = list(self.make_sortish_sampler(1024, shuffle=False))

        def num_tokens_in_example(i):
            return min(self.src_lens[i], self.max_target_length)

        # call fairseq cython function
        batch_sampler: List[List[int]] = batch_by_size(
            sorted_indices,
            num_tokens_fn=num_tokens_in_example,
            max_tokens=max_tokens_per_batch,
            required_batch_size_multiple=64,
        )
        shuffled_batches = [
            batch_sampler[i] for i in np.random.permutation(range(len(batch_sampler)))
        ]
        # move the largest batch to the front to OOM quickly (uses an approximation for padding)
        approximate_toks_per_batch = [
            max(self.src_lens[i] for i in batch) * len(batch) for batch in shuffled_batches
        ]
        largest_batch_idx = np.argmax(approximate_toks_per_batch)
        shuffled_batches[0], shuffled_batches[largest_batch_idx] = (
            shuffled_batches[largest_batch_idx],
            shuffled_batches[0],
        )
        return shuffled_batches

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        graph_line = self.prefix + linecache.getline(str(self.src_graph_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")

        assert source_line, f"empty source line for index {index}"
        assert graph_line, f"empty graph line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"

        source_inputs = self.encode_line(self.tokenizer, source_line, self.max_source_length)
        graph_inputs = self.encode_line(self.tokenizer, graph_line, self.max_source_length)
        target_inputs = self.encode_line(self.tokenizer, tgt_line, self.max_target_length)

        source_ids = source_inputs["input_ids"].squeeze()
        graph_ids = graph_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        graph_mask = graph_inputs["attention_mask"].squeeze()

        return {
            "input_ids": source_ids,
            "graph_ids": graph_ids,
            "attention_mask": src_mask,
            "graph_attention_mask": graph_mask,
            "labels": target_ids,
        }

    def encode_line(self, tokenizer, line, max_length, pad_to_max_length=True, return_tensors="pt"):
        """Only used by LegacyDataset"""
        return tokenizer(
            [line],
            max_length=max_length,
            padding="max_length" if pad_to_max_length else None,
            truncation=True,
            return_tensors=return_tensors,
            **self.dataset_kwargs,
        )

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        graph_ids = torch.stack([x["graph_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        graph_masks = torch.stack([x["graph_attention_mask"] for x in batch])
        target_ids = torch.stack([x["labels"] for x in batch])
        pad_token_id = self.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        graph_ids, graph_mask = trim_batch(graph_ids, pad_token_id, attention_mask=graph_masks)
        batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "graph_ids": graph_ids,
            "graph_attention_mask": graph_mask,
            "labels": y,
        }
        return batch


class Seq2SeqDataCollator:
    def __init__(self, tokenizer, data_args, tpu_num_cores=None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        assert (
            self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        self.data_args = data_args
        self.tpu_num_cores = tpu_num_cores
        self.dataset_kwargs = {"add_prefix_space": isinstance(tokenizer, BartTokenizer)}
        if data_args.src_lang is not None:
            self.dataset_kwargs["src_lang"] = data_args.src_lang
        if data_args.tgt_lang is not None:
            self.dataset_kwargs["tgt_lang"] = data_args.tgt_lang

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        if hasattr(self.tokenizer, "prepare_seq2seq_batch"):
            batch = self._encode(batch)
            input_ids, attention_mask, labels = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
            )
        else:
            input_ids = torch.stack([x["input_ids"] for x in batch])
            attention_mask = torch.stack([x["attention_mask"] for x in batch])
            labels = torch.stack([x["labels"] for x in batch])

            labels = trim_batch(labels, self.pad_token_id)
            input_ids, attention_mask = trim_batch(
                input_ids, self.pad_token_id, attention_mask=attention_mask
            )

        if isinstance(self.tokenizer, T5Tokenizer):
            decoder_input_ids = self._shift_right_t5(labels)
        else:
            decoder_input_ids = shift_tokens_right(labels, self.pad_token_id)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }
        return batch

    def _shift_right_t5(self, input_ids):
        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.pad_token_id
        return shifted_input_ids

    def _encode(self, batch) -> Dict[str, torch.Tensor]:
        batch_encoding = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            tgt_texts=[x["tgt_texts"] for x in batch],
            max_length=self.data_args.max_source_length,
            max_target_length=self.data_args.max_target_length,
            padding="max_length" if self.tpu_num_cores is not None else "longest",  # TPU hack
            return_tensors="pt",
            **self.dataset_kwargs,
        )
        return batch_encoding.data


class SortishSampler(Sampler):
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."

    def __init__(self, data, batch_size, shuffle=True):
        self.data, self.bs, self.shuffle = data, batch_size, shuffle

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(sortish_sampler_indices(self.data, self.bs, shuffle=self.shuffle))


def sortish_sampler_indices(data: List, bs: int, shuffle=True) -> np.array:
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."
    if not shuffle:
        return np.argsort(np.array(data) * -1)

    def key_fn(i):
        return data[i]

    idxs = np.random.permutation(len(data))
    sz = bs * 50
    ck_idx = [idxs[i : i + sz] for i in range(0, len(idxs), sz)]
    sort_idx = np.concatenate([sorted(s, key=key_fn, reverse=True) for s in ck_idx])
    sz = bs
    ck_idx = [sort_idx[i : i + sz] for i in range(0, len(sort_idx), sz)]
    max_ck = np.argmax([key_fn(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
    ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]  # then make sure it goes first.
    sort_idx = (
        np.concatenate(np.random.permutation(ck_idx[1:]))
        if len(ck_idx) > 1
        else np.array([], dtype=np.int)
    )
    sort_idx = np.concatenate((ck_idx[0], sort_idx))
    return sort_idx


class DistributedSortishSampler(Sampler):
    """Copied from torch DistributedSampler"""

    def __init__(
        self,
        dataset,
        batch_size,
        num_replicas=None,
        rank=None,
        add_extra_examples=True,
        shuffle=True,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        if add_extra_examples:
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(dataset)
            self.num_samples = len(self.available_indices)
        self.batch_size = batch_size
        self.add_extra_examples = add_extra_examples
        self.shuffle = shuffle

    def __iter__(self) -> Iterable:
        g = torch.Generator()
        g.manual_seed(self.epoch)

        sortish_data = [self.dataset.src_lens[i] for i in self.available_indices]
        sortish_indices = sortish_sampler_indices(
            sortish_data, self.batch_size, shuffle=self.shuffle
        )
        indices = [self.available_indices[i] for i in sortish_indices]
        assert len(indices) == self.num_samples
        return iter(indices)

    @cached_property
    def available_indices(self) -> np.array:
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        available_indices = indices[self.rank : self.total_size : self.num_replicas]
        return available_indices

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


logger = getLogger(__name__)


def use_task_specific_params(model, task):
    """Update config with summarization specific params."""
    task_specific_params = model.config.task_specific_params

    if task_specific_params is not None:
        pars = task_specific_params.get(task, {})
        logger.info(f"using task specific params for {task}: {pars}")
        model.config.update(pars)


def pickle_load(path):
    """pickle.load(path)"""
    with open(path, "rb") as f:
        return pickle.load(f)


def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def flatten_list(summary_ids: List[List]):
    return [x for x in itertools.chain.from_iterable(summary_ids)]


def save_git_info(folder_path: str) -> None:
    """Save git information to output_dir/git_log.json"""
    repo_infos = get_git_info()
    save_json(repo_infos, os.path.join(folder_path, "git_log.json"))


def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, **json_dump_kwargs)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_git_info():
    # repo = git.Repo(search_parent_directories=True)
    # repo_infos = {
    #     "repo_id": str(repo),
    #     "repo_sha": str(repo.head.object.hexsha),
    #     "repo_branch": str(repo.active_branch),
    #     "hostname": str(socket.gethostname()),
    # }
    repo_infos = {
        "repo_id": "",
        "repo_sha": "",
        "repo_branch": "",
        "hostname": "",
    }
    return repo_infos


ROUGE_KEYS = ["rouge1", "rouge2", "rougeL", "rougeLsum"]


def extract_rouge_mid_statistics(dct):
    new_dict = {}
    for k1, v1 in dct.items():
        mid = v1.mid
        new_dict[k1] = {
            stat: round(getattr(mid, stat), 4) for stat in ["precision", "recall", "fmeasure"]
        }
    return new_dict


def calculate_rouge(
    pred_lns: List[str],
    tgt_lns: List[str],
    use_stemmer=True,
    rouge_keys=ROUGE_KEYS,
    return_precision_and_recall=False,
    bootstrap_aggregation=True,
    newline_sep=True,
) -> Dict:
    """Calculate rouge using rouge_scorer package.

    Args:
        pred_lns: list of summaries generated by model
        tgt_lns: list of groundtruth summaries (e.g. contents of val.target)
        use_stemmer:  Bool indicating whether Porter stemmer should be used to
        strip word suffixes to improve matching.
        rouge_keys:  which metrics to compute, defaults to rouge1, rouge2, rougeL, rougeLsum
        return_precision_and_recall: (False) whether to also return precision and recall.
        bootstrap_aggregation: whether to do the typical bootstrap resampling of scores. Defaults to True, if False
            this function returns a collections.defaultdict[metric: list of values for each observation for each subscore]``
        newline_sep:(default=True) whether to add newline between sentences. This is essential for calculation rougeL
        on multi sentence summaries (CNN/DM dataset).

    Returns:
         Dict[score: value] if aggregate else defaultdict(list) keyed by rouge_keys

    """

    scorer = rouge_scorer.RougeScorer(rouge_keys, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()
    for pred, tgt in zip(tgt_lns, pred_lns):
        # rougeLsum expects "\n" separated sentences within a summary
        if newline_sep:
            pred = add_newline_to_end_of_each_sentence(pred)
            tgt = add_newline_to_end_of_each_sentence(tgt)
            # pred = pred + '\n'
            # tgt = tgt + '\n'
        scores = scorer.score(pred, tgt)
        aggregator.add_scores(scores)

    if bootstrap_aggregation:
        result = aggregator.aggregate()
        if return_precision_and_recall:
            return extract_rouge_mid_statistics(result)  # here we return dict
        else:
            return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

    else:
        return aggregator._scores  # here we return defaultdict(list)


# Utilities for freezing parameters and checking whether they are frozen


def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


def freeze_embeds(model):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    model_type = model.config.model_type

    if model_type == "t5":
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)
    elif model_type == "fsmt":
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)
    else:
        freeze_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)


def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())


def any_requires_grad(model: nn.Module) -> bool:
    return any(grad_status(model))


def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"


def assert_not_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    npars = len(model_grads)
    assert any(model_grads), f"none of {npars} weights require grad"


def parse_numeric_n_bool_cl_kwargs(unparsed_args: List[str]) -> Dict[str, Union[int, float, bool]]:
    """
    Parse an argv list of unspecified command line args to a dict.
    Assumes all values are either numeric or boolean in the form of true/false.
    """
    result = {}
    assert len(unparsed_args) % 2 == 0, f"got odd number of unparsed args: {unparsed_args}"
    num_pairs = len(unparsed_args) // 2
    for pair_num in range(num_pairs):
        i = 2 * pair_num
        assert unparsed_args[i].startswith("--")
        if unparsed_args[i + 1].lower() == "true":
            value = True
        elif unparsed_args[i + 1].lower() == "false":
            value = False
        else:
            try:
                value = int(unparsed_args[i + 1])
            except ValueError:
                value = float(unparsed_args[i + 1])  # this can raise another informative ValueError

        result[unparsed_args[i][2:]] = value
    return result


def write_txt_file(ordered_tgt, path):
    f = Path(path).open("w")
    for ln in ordered_tgt:
        f.write(ln + "\n")
        f.flush()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def smart_emb_init(tokenizer, model, old_vocab_size):
    INIT = "Ġ"
    vocab = tokenizer.get_vocab()
    print("Initializing AMR Vocab according to similar tokens")
    for tok, idx in vocab.items():
        tok = tok.lstrip(INIT).strip()
        if idx < old_vocab_size:
            continue
        elif tok.startswith("<pointer:") and tok.endswith(">"):
            tok_split = ["pointer", str(tok.split(":")[1].strip(">"))]

        elif tok.startswith("<"):
            continue

        elif tok.startswith(":"):

            if tok.startswith(":op"):
                tok_split = ["relation", "operator", str(int(tok[3:]))]

            elif tok.startswith(":snt"):
                tok_split = ["relation", "sentence", str(int(tok[4:]))]

            elif tok.startswith(":ARG"):
                tok_split = ["relation", "argument", str(int(tok[4:]))]

            else:
                tok_split = ["relation"] + tok.lstrip(":").split("-")
        elif "-" in tok:
            tok_split = tok.split("-")
        elif "_" in tok:
            tok_split = tok.split("_")
        else:
            tok_split = [tok]

        vecs, idxs = [], []
        for s in tok_split:
            idx_split = tokenizer.convert_tokens_to_ids(INIT + s)
            idxs.append(str(idx_split))
            if idx_split != 3:
                vec_split = model.model.shared.weight.data[idx_split].clone()
                vecs.append(vec_split)

        # print(f"Ori:{tok}, Tok:{','.join(tok_split)}, idx:{','.join(idxs)}")

        if vecs:
            vec = torch.stack(vecs, 0).mean(0)
            noise = torch.empty_like(vec)
            noise.uniform_(-0.1, +0.1)
            model.model.shared.weight.data[idx] = vec + noise
    # exit()
    return model


def calculate_smatch(test_path, predictions_path) -> dict:
    with Path(predictions_path).open() as p, Path(test_path).open() as g:
        score = next(smatch.score_amr_pairs(p, g))
    return {"smatch": score[2]}
