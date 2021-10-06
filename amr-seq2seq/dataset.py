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

# import git
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

nltk = None
import nltk


def add_newline_to_end_of_each_sentence(x: str) -> str:
    """This was added to get rougeLsum scores matching published rougeL scores for BART and PEGASUS."""
    re.sub("<n>", "", x)  # remove pegasus newline char
    assert nltk, "nltk must be installed to separate newlines between sentences. (pip install nltk)"
    return "\n".join(nltk.sent_tokenize(x))


def trim_batch(input_ids, pad_token_id, attention_mask=None):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


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
        tgt_pad_token_id=-100,
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
        self.tgt_pad_token_id = tgt_pad_token_id
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


class Seq2SeqDataset(AbstractSeq2SeqDataset):
    """A dataset that calls prepare_seq2seq_batch."""

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = self.prefix + linecache.getline(str(self.tgt_file), index).rstrip("\n")

        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        source_inputs = self.tokenizer(
            source_line,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with self.tokenizer.as_target_tokenizer():
            target_inputs = self.tokenizer(
                tgt_line, max_length=self.max_target_length, padding="max_length", truncation=True, return_tensors="pt",
            )
        # print('\ntarget input:', target_inputs)
        if self.pad_token_id != self.tgt_pad_token_id:
            target_inputs["input_ids"].masked_fill_(target_inputs["input_ids"] == self.pad_token_id, self.tgt_pad_token_id)

        source_ids = source_inputs["input_ids"]
        target_ids = target_inputs["input_ids"]
        src_mask = source_inputs["attention_mask"]
        # print('target ids:', target_ids)
        # exit()
        return {"labels": target_ids, "input_ids": source_ids, "attention_mask": src_mask}

    def collate_fn(self, batch):
        """Call prepare_seq2seq_batch."""
        # print('batch', batch)
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        target_ids = torch.stack([x["labels"] for x in batch])
        y = trim_batch(target_ids, self.tgt_pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, self.pad_token_id, attention_mask=masks)
        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": y,
        }


class Seq2SeqDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, tokenizer, n_obs, target_lens, max_source_length,
                 train_batch_size, val_batch_size, dataloader_num_workers, prefix=""):
        super().__init__()
        self.data_dir = data_dir
        self.max_source_length = max_source_length
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.n_obs = n_obs
        self.target_lens = target_lens
        self.tokenizer = tokenizer
        self.prefix = prefix

    def setup(self, stage=None):
        
        self.train_dataset = Seq2SeqDataset(
            self.tokenizer,
            data_dir=self.data_dir,
            type_path='train',
            n_obs=self.n_obs['train'],
            max_target_length=self.target_lens['train'],
            max_source_length=self.max_source_length,
            prefix=self.prefix,
            tgt_pad_token_id=self.tokenizer.pad_token_id,
        )
        self.val_dataset = Seq2SeqDataset(
            self.tokenizer,
            data_dir=self.data_dir,
            type_path='dev',
            n_obs=self.n_obs['val'],
            max_target_length=self.target_lens['val'],
            max_source_length=self.max_source_length,
            prefix=self.prefix,
            tgt_pad_token_id=self.tokenizer.pad_token_id,
        )
        self.test_dataset = Seq2SeqDataset(
            self.tokenizer,
            data_dir=self.data_dir,
            type_path='test',
            n_obs=self.n_obs['test'],
            max_target_length=self.target_lens['test'],
            max_source_length=self.max_source_length,
            prefix=self.prefix,
            tgt_pad_token_id=self.tokenizer.pad_token_id,
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

@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, batch):

        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        target_ids = torch.stack([x["labels"] for x in batch])
        y = trim_batch(target_ids, self.label_pad_token_id)
        source_ids, source_mask = trim_batch(
            input_ids, self.tokenizer.pad_token_id, attention_mask=masks
        )
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=y)
        # print('input_id', source_ids.size(), source_ids)
        # print('labels', y.size(), y)
        # exit()
        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": y,
            "decoder_input_ids": decoder_input_ids,
        }