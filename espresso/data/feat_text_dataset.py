# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List, Optional

import numpy as np

import torch

from fairseq.data import data_utils
from fairseq.tokenizer import tokenize_line

from espresso.tools.specaug_interpolate import specaug

try:
    import kaldi_io
except ImportError:
    raise ImportError("Please install kaldi_io with: pip install kaldi_io")


class FeatScpDataset(torch.utils.data.Dataset):
    """
    A dataset for audio features prepared in Kaldi scp format (e.g., feats.scp).
    See http://kaldi-asr.org/doc/tutorial_running.html#tutorial_running_feats
    for the format descriptions. This class loads a feature matrix from the disk
    every time each entry is inquired, thus incurs the most intensive I/O.
    """

    def __init__(
        self,
        utt_ids: List[str],
        rxfiles: List[str],
        utt2num_frames: Optional[List[int]] = None,
        seed=1,
        specaugment_config: Optional[str] = None,
    ):
        super().__init__()
        assert len(utt_ids) == len(rxfiles)
        self.dtype = np.float
        self.utt_ids = utt_ids
        self.rxfiles = rxfiles
        self.size = len(utt_ids)  # number of utterances
        self.sizes = []  # length of each utterance
        if utt2num_frames is not None and len(utt2num_frames) > 0:
            assert len(utt2num_frames) == self.size
            self.sizes = utt2num_frames

        for rxfile in self.rxfiles:
            try:
                feat = kaldi_io.read_mat(rxfile)
            except Exception:
                raise Exception("failed to read feature matrix {}.".format(rxfile))
            assert feat is not None and isinstance(feat, np.ndarray)
            if len(self.sizes) == self.size:
                break
            self.sizes.append(feat.shape[0])

        assert len(self.sizes) == self.size
        self.sizes = np.array(self.sizes, dtype=np.int32)
        self.feat_dim = feat.shape[1]  # feature dimension
        self.seed = seed
        self.specaugment_config = specaugment_config
        self.epoch = 1

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    def filter_and_reorder(self, indices):
        assert isinstance(indices, (list, np.ndarray))
        indices = np.array(indices)
        assert all(indices < len(self.utt_ids)) and all(indices >= 0)
        assert len(np.unique(indices)) == len(indices), "Duplicate elements in indices."
        self.utt_ids = [self.utt_ids[i] for i in indices]
        self.rxfiles = [self.rxfiles[i] for i in indices]
        self.sizes = self.sizes[indices]
        self.size = len(self.utt_ids)
        self.ordered_indices = list(range(self.size))

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, i):
        self.check_index(i)
        feat = kaldi_io.read_mat(self.rxfiles[i])
        if self.specaugment_config is not None and self.specaugment_config != "":
            with data_utils.numpy_seed(self.seed, self.epoch, i):
                feat = specaug(feat, **eval(self.specaugment_config))
        item = torch.from_numpy(feat).float()
        return item

    def __len__(self):
        return self.size

    @staticmethod
    def exists(path):
        return os.path.exists(path)


class FeatScpCachedDataset(FeatScpDataset):
    """
    This class loads a batch of feature matrices (specified as *cache_size*)
    every time an entry is inquired. The inquire order should be known in advance.
    It balances the I/O efficiency and memory usage.
    """

    def __init__(
        self, utt_ids: List[str], rxfiles: List[str], utt2num_frames: Optional[List[int]] = None,
        seed=1, specaugment_config: Optional[str] = None, ordered_prefetch=False, cache_size=4096,
    ):
        super().__init__(
            utt_ids, rxfiles, utt2num_frames=utt2num_frames,
            seed=seed, specaugment_config=specaugment_config,
        )
        self.cache = None
        self.cache_index = {}
        self.cache_size = cache_size  # in terms of number of examples
        self.start_pos_for_next_cache = 0
        self.ordered_indices = list(range(self.size))
        # set to True ONLY if examples are queried in the same order as
        # self.ordered_indices, and doing this will speed up search of the
        # queried index
        self.ordered_prefetch = ordered_prefetch
        # a flag to indicate whether self.prefetch() has been called. It is related
        # to dummy_batch in trainer.py that uses the first batch when batch_by_size
        # has been called but self.prefetch() has not. In this case we simply only
        # load the queried samples into memory and don't do any caching.
        self.prefetch_called = False

    @property
    def supports_prefetch(self):
        return True

    def prefetch(self, indices):
        """Sets self.ordered_indices. If being called, the caller is supposed to
        query examples in the same order as self.ordered_indices.
        self.ordered_prefetch can be set to True in this case. Note: the purpose
        of this function is different from what it is supposed to do in the
        fairseq framework."""
        assert isinstance(indices, (list, np.ndarray))
        assert self.size >= len(indices)
        self.ordered_indices = indices.copy()
        self.start_pos_for_next_cache = 0
        self.prefetch_called = True

    def __getitem__(self, i):
        self.check_index(i)
        if not self.prefetch_called:  # no caching
            feat = kaldi_io.read_mat(self.rxfiles[i])
            return torch.from_numpy(feat).float()
        if i not in self.cache_index:
            assert (
                self.start_pos_for_next_cache < len(self.ordered_indices)
            ), "Position for next cache starting beyond the end of ordered_indices."
            try:
                pos_start = self.ordered_indices.index(
                    i, self.start_pos_for_next_cache,
                )
            except ValueError:
                raise ValueError(
                    "index {} not found in self.ordered_indices. Set "
                    "self.ordered_prefetch to False, and/or call self.prefetch() "
                    "with the full list of indices, and then try again.".format(i)
                )
            pos_end = min(
                pos_start + self.cache_size, len(self.ordered_indices),
            )
            self.start_pos_for_next_cache = pos_end if self.ordered_prefetch else 0
            total_size = 0
            for idx in self.ordered_indices[pos_start: pos_end]:
                total_size += self.sizes[idx]
            self.cache = np.empty((total_size, self.feat_dim), dtype=self.dtype)
            ptx = 0
            self.cache_index.clear()
            for idx in self.ordered_indices[pos_start: pos_end]:
                self.cache_index[idx] = ptx
                length = self.sizes[idx]
                dst = self.cache[ptx: ptx + length]
                feat = kaldi_io.read_mat(self.rxfiles[idx])
                if self.specaugment_config is not None and self.specaugment_config != "":
                    with data_utils.numpy_seed(self.seed, self.epoch, idx):
                        feat = specaug(feat, **eval(self.specaugment_config))
                np.copyto(dst, feat)
                ptx += length

        ptx = self.cache_index[i]
        a = self.cache[ptx: ptx + self.sizes[i]].copy()
        return torch.from_numpy(a).float()


class FeatScpInMemoryDataset(FeatScpDataset):
    """
    This class loads all feature matrices into memory at once.
    It has the maximum memory usage and least I/O.
    """

    def __init__(
        self, utt_ids: List[str], rxfiles: List[str], utt2num_frames: Optional[List[int]] = None,
        seed=1, specaugment_config: Optional[str] = None,
    ):
        super().__init__(
            utt_ids, rxfiles, utt2num_frames=utt2num_frames,
            seed=seed, specaugment_config=specaugment_config,
        )
        self.read_data()

    def read_data(self):
        self.data_offsets = np.append([0], np.cumsum(self.sizes)[:-1])
        self.buffer = np.empty(
            (sum(self.sizes), self.feat_dim),
            dtype=self.dtype,
        )
        for i in range(len(self.data_offsets)):
            ptx = self.data_offsets[i]
            dst = self.buffer[ptx: ptx + self.sizes[i]]
            np.copyto(dst, kaldi_io.read_mat(self.rxfiles[i]))

    def filter_and_reorder(self, indices):
        super().filter_and_reorder(indices)
        self.read_data()

    def __getitem__(self, i):
        self.check_index(i)
        ptx = self.data_offsets[i]
        a = self.buffer[ptx: ptx + self.sizes[i]].copy()
        if self.specaugment_config is not None and self.specaugment_config != "":
            with data_utils.numpy_seed(self.seed, self.epoch, i):
                a = specaug(a, **eval(self.specaugment_config))
        return torch.from_numpy(a).float()


class AsrTextDataset(torch.utils.data.Dataset):
    """Takes a text file as input, tokenizes and tensorizes it in memory at instantiation.
    Both original text and tokenized text are kept in memory."""

    def __init__(self, utt_ids: List[str], texts: List[str], dictionary=None, append_eos=True):
        super().__init__()
        self.dtype = np.float
        self.append_eos = append_eos
        self.read_text(utt_ids, texts, dictionary)

    def read_text(self, utt_ids: List[str], texts: List[str], dictionary=None):
        assert len(utt_ids) == len(texts)
        self.utt_ids = utt_ids
        self.texts = texts
        self.size = len(self.utt_ids)  # number of utterances
        self.token_texts = None
        self.tensor_list = None
        if dictionary is not None:
            self.token_texts = [dictionary.wordpiece_encode(x) for x in texts]
            self.tensor_list = [
                dictionary.encode_line(tokens, add_if_not_exist=False, append_eos=self.append_eos).long()
                for tokens in self.token_texts
            ]
            self.sizes = [len(tensor) for tensor in self.tensor_list]
        else:
            self.sizes = [len(tokenize_line(text)) for text in texts]

        self.sizes = np.array(self.sizes, dtype=np.int32)

        assert (
            (
                dictionary is None
                or (len(self.utt_ids) == len(self.tensor_list) and len(self.utt_ids) == len(self.token_texts))
            )
            and len(self.utt_ids) == len(self.sizes)
        )

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    def filter_and_reorder(self, indices):
        assert isinstance(indices, (list, np.ndarray))
        indices = np.array(indices)
        assert all(indices < self.size) and all(indices >= 0)
        assert (
            len(np.unique(indices)) == len(indices)
        ), "Duplicate elements in indices."
        self.utt_ids = [self.utt_ids[i] for i in indices]
        self.texts = [self.texts[i] for i in indices]
        if self.token_texts is not None:
            self.token_texts = [self.token_texts[i] for i in indices]
        if self.tensor_list is not None:
            self.tensor_list = [self.tensor_list[i] for i in indices]
        self.sizes = self.sizes[indices]
        self.size = len(self.utt_ids)

    def __getitem__(self, i):
        self.check_index(i)
        return (
            self.tensor_list[i] if self.tensor_list is not None else None,
            self.token_texts[i] if self.token_texts is not None else None,
            self.texts[i]
        )

    def __len__(self):
        return self.size

    @staticmethod
    def exists(path):
        return os.path.exists(path)
