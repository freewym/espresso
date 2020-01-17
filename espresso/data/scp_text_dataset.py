# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np
import torch

try:
    import kaldi_io
except ImportError:
    raise ImportError('Please install kaldi_io with: pip install kaldi_io')


class ScpDataset(torch.utils.data.Dataset):
    """
    A dataset for audio features prepared in Kaldi scp format (e.g., feats.scp).
    See http://kaldi-asr.org/doc/tutorial_running.html#tutorial_running_feats
    for the format descriptions. This class loads a feature matrix from the disk
    every time each entry is inquired, thus incurs the most intensive I/O.
    """

    def __init__(self, path):
        super().__init__()
        self.dtype = np.float
        self.read_scp(path)

    def read_scp(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            scp_entries = [line.strip().split(None, 1) for line in f]
        self.utt_ids = [entry[0] for entry in scp_entries]
        self.extended_filenames = [entry[1] for entry in scp_entries]
        self.size = len(scp_entries)  # number of utterances
        self.sizes = []  # length of each utterance
        for filename in self.extended_filenames:
            try:
                feat = kaldi_io.read_mat(filename)
            except Exception:
                raise Exception('failed to read feature matrix {}.'.format(filename))
            assert feat is not None and isinstance(feat, np.ndarray)
            self.sizes.append(feat.shape[0])
        self.sizes = np.array(self.sizes, dtype=np.int32)
        self.feat_dim = feat.shape[1]  # feature dimension

        assert len(self.utt_ids) == len(self.extended_filenames) and \
            len(self.utt_ids) == len(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    def filter_and_reorder(self, indices):
        assert isinstance(indices, (list, np.ndarray))
        indices = np.array(indices)
        assert all(indices < len(self.utt_ids)) and all(indices >= 0)
        assert len(np.unique(indices)) == len(indices), \
            'Duplicate elements in indices.'
        self.utt_ids = [self.utt_ids[i] for i in indices]
        self.extended_filenames = [self.extended_filenames[i] for i in indices]
        self.sizes = self.sizes[indices]
        self.size = len(self.utt_ids)
        self.ordered_indices = list(range(self.size))

    def __getitem__(self, i):
        self.check_index(i)
        feat = kaldi_io.read_mat(self.extended_filenames[i])
        item = torch.from_numpy(feat).float()
        return item

    def __len__(self):
        return self.size

    @staticmethod
    def exists(path):
        return os.path.exists(path)


class ScpCachedDataset(ScpDataset):
    """
    This class loads a batch of feature matrices (specified as *cache_size*)
    every time an entry is inquired. The inquire order should be known in advance.
    It balances the I/O efficiency and memory usage.
    """

    def __init__(self, path, ordered_prefetch=False, cache_size=4096):
        super().__init__(path)
        self.cache = None
        self.cache_index = {}
        self.cache_size = cache_size  # in terms of number of examples
        self.start_search_for_next_pos_start = 0
        self.ordered_indices = list(range(self.size))
        # set to True ONLY if examples are queried in the same order as
        # self.ordered_indices, and doing this will speed up search of the
        # queried index
        self.ordered_prefetch = ordered_prefetch

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

    def __getitem__(self, i):
        self.check_index(i)
        if i not in self.cache_index:
            assert self.start_pos_for_next_cache < \
                len(self.ordered_indices), \
                'Position for next cache starting beyond the end of ordered_indices.'
            try:
                pos_start = self.ordered_indices.index(
                    i, self.start_pos_for_next_cache,
                )
            except ValueError:
                raise ValueError(
                    'index {} not found in self.ordered_indices. Set '
                    'self.ordered_prefetch to False, and/or call self.prefetch() '
                    'with the full list of indices, and then try again.'.format(i)
                )
            pos_end = min(
                pos_start + self.cache_size, len(self.ordered_indices),
            )
            self.start_pos_for_next_cache = pos_end \
                if self.ordered_prefetch else 0
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
                np.copyto(dst, kaldi_io.read_mat(self.extended_filenames[idx]))
                ptx += length

        ptx = self.cache_index[i]
        a = self.cache[ptx: ptx + self.sizes[i]].copy()
        return torch.from_numpy(a).float()


class ScpInMemoryDataset(ScpDataset):
    """
    This class loads all feature matrices into memory at once.
    It has the maximum memory usage and least I/O.
    """

    def __init__(self, path):
        super().__init__(path)
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
            np.copyto(dst, kaldi_io.read_mat(self.extended_filenames[i]))

    def filter_and_reorder(self, indices):
        super().filter_and_reorder(indices)
        self.read_data()

    def __getitem__(self, i):
        self.check_index(i)
        ptx = self.data_offsets[i]
        a = self.buffer[ptx: ptx + self.sizes[i]].copy()
        return torch.from_numpy(a).float()


class AsrTextDataset(torch.utils.data.Dataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory. Each line of the text file is in the
    format of 'utt_id tokenized_text'."""

    def __init__(self, path, dictionary, append_eos=True):
        super().__init__()
        self.dtype = np.float
        self.append_eos = append_eos
        self.read_text(path, dictionary)

    def read_text(self, path, dictionary):
        self.utt_ids = []
        self.tokens_list = []
        self.tensor_list = []
        self.sizes = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                utt_id, tokens = line.strip().split(None, 1)
                self.utt_ids.append(utt_id)
                self.tokens_list.append(tokens)
                tensor = dictionary.encode_line(
                    tokens, add_if_not_exist=False, append_eos=self.append_eos,
                ).long()
                self.tensor_list.append(tensor)
                self.sizes.append(len(self.tensor_list[-1]))

        self.size = len(self.utt_ids)  # number of utterances
        self.sizes = np.array(self.sizes, dtype=np.int32)

        assert len(self.utt_ids) == len(self.tokens_list) and \
            len(self.utt_ids) == len(self.tensor_list) and \
            len(self.utt_ids) == len(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    def filter_and_reorder(self, indices):
        assert isinstance(indices, (list, np.ndarray))
        indices = np.array(indices)
        assert all(indices < self.size) and all(indices >= 0)
        assert len(np.unique(indices)) == len(indices), \
            'Duplicate elements in indices.'
        self.utt_ids = [self.utt_ids[i] for i in indices]
        self.tokens_list = [self.tokens_list[i] for i in indices]
        self.tensor_list = [self.tensor_list[i] for i in indices]
        self.sizes = self.sizes[indices]
        self.size = len(self.utt_ids)

    def __getitem__(self, i):
        self.check_index(i)
        return self.tensor_list[i], self.tokens_list[i]

    def __len__(self):
        return self.size

    @staticmethod
    def exists(path):
        return os.path.exists(path)
