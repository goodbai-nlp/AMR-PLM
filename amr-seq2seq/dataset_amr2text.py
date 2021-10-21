# coding:utf-8

import itertools
import json
import linecache
import math
import os
import pickle
import socket
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple, Union
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import warnings
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

import torch
from torch.nn.utils.rnn import pad_sequence

from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

import numpy as np
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler

from transformers import BartTokenizer, EvalPrediction, PreTrainedTokenizer, T5Tokenizer
from transformers.file_utils import cached_property

from pytorch_lightning.utilities import rank_zero_info
import re

try:
    from fairseq.data.data_utils import batch_by_size

    FAIRSEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    FAIRSEQ_AVAILABLE = False

# nltk = None
# import nltk

# def add_newline_to_end_of_each_sentence(x: str) -> str:
#     """This was added to get rougeLsum scores matching published rougeL scores for BART and PEGASUS."""
#     re.sub("<n>", "", x)  # remove pegasus newline char
#     assert nltk, "nltk must be installed to separate newlines between sentences. (pip install nltk)"
#     return "\n".join(nltk.sent_tokenize(x))


# def trim_batch(input_ids, pad_token_id, attention_mask=None):
#     """Remove columns that are populated exclusively by pad_token_id"""
#     keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
#     if attention_mask is None:
#         return input_ids[:, keep_column_mask]
#     else:
#         return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class AMR2TextDataset(Dataset):
    def __init__(
        self, tokenizer, data_dir, type_path="train", max_source_length=512, max_target_length=512,
    ):
        INIT = "Ġ"
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        print("Tokenizer:", len(self.tokenizer), self.tokenizer)
        self.sentences = json.load(
            open(f"{data_dir}/{type_path}_tgt_tokens.json", "r", encoding="utf-8")
        )
        self.linearized_tokens = json.load(
            open(f"{data_dir}/{type_path}_linearized_tokens.json", "r", encoding="utf-8")
        )
        self.linearized_ids = json.load(
            open(f"{data_dir}/{type_path}_linearized_ids.json", "r", encoding="utf-8")
        )

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sample = {}
        sample["id"] = idx
        sample["sentences"] = self.sentences[idx]
        sample["linearized_graphs_ids"] = self.linearized_ids[idx]
        sample["linearized_graph"] = self.linearized_tokens[idx]
        return sample

    def size(self, sample):
        return len(sample["linearized_graphs_ids"])

    def collate_fn(self, samples, device=torch.device("cpu")):
        # x = [
        #     [self.tokenizer.bos_token] + s["linearized_graph"] + [self.tokenizer.eos_token]
        #     for s in samples
        # ]
        x = [
            [self.tokenizer.amr_bos_token] + s["linearized_graph"] + [self.tokenizer.eos_token]
            for s in samples
        ]
        x = self.amr_batch_encode(x, max_length=self.max_target_length, pad_to_max_length=True).to(
            device
        )
        x_mask = x.ne(self.tokenizer.pad_token_id).int()
        y = [s["sentences"] for s in samples]
        y = self.tokenizer.batch_encode_plus(y, return_tensors="pt", padding="longest")["input_ids"].to(device)
        dec_inp = shift_tokens_right(y[:, 1:], self.tokenizer.pad_token_id, self.tokenizer.bos_token_id).to(device)

        return {
            "input_ids": x,
            "attention_mask": x_mask,
            "decoder_input_ids": dec_inp,
            "labels": y[:, 1:],
        }

    def amr_batch_encode(
        self, input_lst, max_length, pad_to_max_length=False, device=torch.device("cpu")
    ):
        res = []
        for itm_lst in input_lst:
            res.append(
                self.get_ids(itm_lst, max_length=max_length, pad_to_max_length=pad_to_max_length)
            )
        raw_data = torch.stack(res, dim=0)
        keep_column_mask = raw_data.ne(self.tokenizer.pad_token_id).any(dim=0)
        return raw_data[:, keep_column_mask]

    def get_ids(self, tokens, max_length=0, pad_to_max_length=False):
        token_ids = [self.tokenizer.encoder.get(b, self.tokenizer.unk_token_id) for b in tokens]
        if pad_to_max_length:
            assert max_length > 0, "Invalid max-length: {}".format(max_length)
            pad_ids = [self.tokenizer.pad_token_id for _ in range(max_length)]
            len_tok = len(token_ids)
            if max_length > len_tok:
                pad_ids[:len_tok] = map(int, token_ids)
            else:
                pad_ids = token_ids[:max_length]
            return torch.tensor(pad_ids, dtype=torch.long)
        return torch.tensor(token_ids, dtype=torch.long)


class AMR2TextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        tokenizer,
        n_obs,
        target_lens,
        max_source_length,
        max_target_length,
        train_batch_size,
        val_batch_size,
        dataloader_num_workers,
        prefix="",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.n_obs = n_obs
        self.target_lens = target_lens
        self.tokenizer = tokenizer
        self.prefix = prefix

    def setup(self, stage=None):

        self.train_dataset = AMR2TextDataset(
            self.tokenizer,
            data_dir=self.data_dir,
            type_path="train",
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
        )
        self.val_dataset = AMR2TextDataset(
            self.tokenizer,
            data_dir=self.data_dir,
            type_path="dev",
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
        )
        self.test_dataset = AMR2TextDataset(
            self.tokenizer,
            data_dir=self.data_dir,
            type_path="test",
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            collate_fn=self.train_dataset.collate_fn,
            shuffle=True,
            num_workers=self.dataloader_num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            collate_fn=self.val_dataset.collate_fn,
            shuffle=False,
            num_workers=self.dataloader_num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            collate_fn=self.test_dataset.collate_fn,
            shuffle=False,
            num_workers=self.dataloader_num_workers,
        )


# @dataclass
# class DataCollatorForSeq2Seq:
#     """
#     Data collator that will dynamically pad the inputs received, as well as the labels.

#     Args:
#         tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
#             The tokenizer used for encoding the data.
#         model (:class:`~transformers.PreTrainedModel`):
#             The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
#             prepare the `decoder_input_ids`

#             This is useful when using `label_smoothing` to avoid calculating loss twice.
#         padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
#             Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
#             among:

#             * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
#               sequence is provided).
#             * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
#               maximum acceptable input length for the model if that argument is not provided.
#             * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
#               different lengths).
#         max_length (:obj:`int`, `optional`):
#             Maximum length of the returned list and optionally padding length (see above).
#         pad_to_multiple_of (:obj:`int`, `optional`):
#             If set will pad the sequence to a multiple of the provided value.

#             This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
#             7.5 (Volta).
#         label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
#             The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
#     """

#     tokenizer: PreTrainedTokenizerBase
#     model: Optional[PreTrainedModel] = None
#     padding: Union[bool, str, PaddingStrategy] = True
#     max_length: Optional[int] = None
#     pad_to_multiple_of: Optional[int] = None
#     label_pad_token_id: int = -100

#     def __call__(self, batch):

#         input_ids = torch.stack([x["input_ids"] for x in batch])
#         masks = torch.stack([x["attention_mask"] for x in batch])
#         target_ids = torch.stack([x["labels"] for x in batch])
#         y = trim_batch(target_ids, self.label_pad_token_id)
#         source_ids, source_mask = trim_batch(
#             input_ids, self.tokenizer.pad_token_id, attention_mask=masks
#         )
#         if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
#             decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=y)
#         # print('input_id', source_ids.size(), source_ids)
#         # print('labels', y.size(), y)
#         # exit()
#         return {
#             "input_ids": source_ids,
#             "attention_mask": source_mask,
#             "labels": y,
#             "decoder_input_ids": decoder_input_ids,
#         }

