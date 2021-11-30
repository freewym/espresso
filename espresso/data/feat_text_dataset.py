# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import re
from concurrent.futures.thread import ThreadPoolExecutor
from io import BytesIO
from subprocess import PIPE, run
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from espresso.tools.specaug_interpolate import specaug
from espresso.tools.utils import (
    compute_num_frames_from_feat_or_waveform,
    get_torchaudio_fbank_or_mfcc,
)
from fairseq.data import data_utils
from fairseq.data.audio.audio_utils import get_waveform
from fairseq.data.audio.feature_transforms import CompositeAudioFeatureTransform

try:
    import kaldi_io
except ImportError:
    raise ImportError("Please install kaldi_io with: pip install kaldi_io")


logger = logging.getLogger(__name__)


class AudioFeatDataset(torch.utils.data.Dataset):
    """
    A dataset for audio features, read features prepared in Kaldi scp format (e.g., feats.scp),
    or raw waveforms (in the form of files or commands in wav.scp).
    See http://kaldi-asr.org/doc/tutorial_running.html#tutorial_running_feats
    for the format descriptions. This class loads a feature matrix/wave files from the disk
    every time each entry is inquired, thus incurs the most intensive I/O.
    """

    def __init__(
        self,
        utt_ids: List[str],
        rxfiles: List[str],
        utt2num_frames: Optional[List[int]] = None,
        feat_dim: Optional[int] = None,  # only relevant when reading from raw waveforms
        feature_type: Optional[
            str
        ] = None,  # currently support fbank or mfcc; only relevant when reading from raw waveforms
        seed=1,
        feature_transforms_config: Optional[Dict[str, Any]] = None,
        specaugment_config: Optional[str] = None,
    ):
        super().__init__()
        assert len(utt_ids) == len(rxfiles)
        self.dtype = np.float
        self.utt_ids = utt_ids
        self.rxfiles = rxfiles
        self.size = len(utt_ids)  # number of utterances
        self.sizes = []  # length of each utterance in terms of the number of frames
        if utt2num_frames is not None and len(utt2num_frames) > 0:
            assert len(utt2num_frames) == self.size
            self.sizes = utt2num_frames

        first_rxfile = rxfiles[0]
        if re.search(r"\.ark:\d+$", first_rxfile.strip()) is not None:  # from feats.scp
            self.input_format = "feat"
            self.feat_dim = kaldi_io.read_mat(first_rxfile).shape[
                1
            ]  # feature dimension
        else:
            self.input_format = (
                "command"
                if re.search(r"\|$", first_rxfile.strip()) is not None
                else "wave"
            )
            self.feat_dim = feat_dim
            self.feature_type = feature_type
            assert self.feat_dim is not None
            assert self.feature_type in ["fbank", "mfcc"]

        if len(self.sizes) == 0:
            logger.info("Computing number of frames from audios...")
            with ThreadPoolExecutor(max_workers=32) as ex:
                futures = []
                for rxfile in self.rxfiles:
                    futures.append(
                        ex.submit(compute_num_frames_from_feat_or_waveform, rxfile)
                    )

                for future in tqdm(futures, desc="Processing", leave=False):
                    result = future.result()
                    self.sizes.append(result)

        assert len(self.sizes) == self.size
        self.sizes = np.array(self.sizes, dtype=np.int32)
        self.feature_transforms = CompositeAudioFeatureTransform.from_config_dict(
            config=feature_transforms_config
        )
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

    def _get_features(self, i):
        if self.input_format == "feat":
            feat = kaldi_io.read_mat(self.rxfiles[i])
        else:
            if self.input_format == "command":
                source = BytesIO(
                    run(self.rxfiles[i][:-1], shell=True, stdout=PIPE).stdout
                )
            else:
                source = self.rxfiles[i]
            waveform, sample_rate = get_waveform(
                source, normalization=False, always_2d=True
            )
            feat = get_torchaudio_fbank_or_mfcc(
                waveform,
                sample_rate,
                n_bins=self.feat_dim,
                feature_type=self.feature_type,
            )
            if self.feature_transforms is not None:
                feat = self.feature_transforms(feat)
        if self.specaugment_config is not None and self.specaugment_config != "":
            with data_utils.numpy_seed(self.seed, self.epoch, i):
                feat = specaug(feat, **eval(self.specaugment_config))
        return feat

    def __getitem__(self, i):
        self.check_index(i)
        feat = self._get_features(i)
        item = torch.from_numpy(feat).float()
        return item

    def __len__(self):
        return self.size

    @staticmethod
    def exists(path):
        return os.path.exists(path)


class AudioFeatCachedDataset(AudioFeatDataset):
    """
    This class loads a batch of feature/waveform matrices (specified as *cache_size*)
    every time an entry is inquired. The inquire order should be known in advance.
    It balances the I/O efficiency and memory usage.
    """

    def __init__(
        self,
        utt_ids: List[str],
        rxfiles: List[str],
        utt2num_frames: Optional[List[int]] = None,
        feat_dim: Optional[int] = None,  # only relevant when reading from raw waveforms
        feature_type: Optional[
            str
        ] = None,  # currently support fbank or mfcc; only relevant when reading from raw waveforms
        seed=1,
        feature_transforms_config: Optional[Dict[str, Any]] = None,
        specaugment_config: Optional[str] = None,
        ordered_prefetch=False,
        cache_size=4096,
    ):
        super().__init__(
            utt_ids,
            rxfiles,
            utt2num_frames=utt2num_frames,
            feat_dim=feat_dim,
            feature_type=feature_type,
            seed=seed,
            feature_transforms_config=feature_transforms_config,
            specaugment_config=specaugment_config,
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
            feat = self._get_features(i)
            return torch.from_numpy(feat).float()
        if i not in self.cache_index:
            assert self.start_pos_for_next_cache < len(
                self.ordered_indices
            ), "Position for next cache starting beyond the end of ordered_indices."
            try:
                pos_start = self.ordered_indices.index(
                    i,
                    self.start_pos_for_next_cache,
                )
            except ValueError:
                raise ValueError(
                    "index {} not found in self.ordered_indices. Set "
                    "self.ordered_prefetch to False, and/or call self.prefetch() "
                    "with the full list of indices, and then try again.".format(i)
                )
            pos_end = min(
                pos_start + self.cache_size,
                len(self.ordered_indices),
            )
            self.start_pos_for_next_cache = pos_end if self.ordered_prefetch else 0
            total_size = 0
            for idx in self.ordered_indices[pos_start:pos_end]:
                total_size += self.sizes[idx]
            self.cache = np.empty((total_size, self.feat_dim), dtype=self.dtype)
            ptx = 0
            self.cache_index.clear()
            for idx in self.ordered_indices[pos_start:pos_end]:
                self.cache_index[idx] = ptx
                length = self.sizes[idx]
                dst = self.cache[ptx : ptx + length]
                feat = self._get_features(idx)
                np.copyto(dst, feat)
                ptx += length

        ptx = self.cache_index[i]
        a = self.cache[ptx : ptx + self.sizes[i]].copy()
        return torch.from_numpy(a).float()


class AudioFeatInMemoryDataset(AudioFeatDataset):
    """
    This class loads all feature/waveform matrices into memory at once.
    It has the maximum memory usage and least I/O.
    """

    def __init__(
        self,
        utt_ids: List[str],
        rxfiles: List[str],
        utt2num_frames: Optional[List[int]] = None,
        feat_dim: Optional[int] = None,  # only relevant when reading from raw waveforms
        feature_type: Optional[
            str
        ] = None,  # currently support fbank or mfcc; only relevant when reading from raw waveforms
        seed=1,
        feature_transforms_config: Optional[Dict[str, Any]] = None,
        specaugment_config: Optional[str] = None,
    ):
        super().__init__(
            utt_ids,
            rxfiles,
            utt2num_frames=utt2num_frames,
            feat_dim=feat_dim,
            feature_type=feature_type,
            seed=seed,
            feature_transforms_config=feature_transforms_config,
            specaugment_config=specaugment_config,
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
            dst = self.buffer[ptx : ptx + self.sizes[i]]
            feat = self._get_features(i)
            np.copyto(dst, feat)

    def filter_and_reorder(self, indices):
        super().filter_and_reorder(indices)
        self.read_data()

    def __getitem__(self, i):
        self.check_index(i)
        ptx = self.data_offsets[i]
        a = self.buffer[ptx : ptx + self.sizes[i]].copy()
        return torch.from_numpy(a).float()


class AsrTextDataset(torch.utils.data.Dataset):
    """Takes a text file as input, tokenizes and tensorizes it in memory at instantiation.
    Both original text and tokenized text are kept in memory."""

    def __init__(
        self, utt_ids: List[str], texts: List[str], dictionary=None, append_eos=True
    ):
        super().__init__()
        self.dtype = np.float
        self.dictionary = dictionary
        self.append_eos = append_eos
        self.read_text(utt_ids, texts, dictionary)

    def read_text(self, utt_ids: List[str], texts: List[str], dictionary=None):
        assert len(utt_ids) == len(texts)
        self.utt_ids = utt_ids
        self.texts = texts
        self.size = len(self.utt_ids)  # number of utterances
        from fairseq.tokenizer import tokenize_line

        if dictionary is not None:
            self.sizes = [
                len(tokenize_line(dictionary.wordpiece_encode(text)))
                + (1 if self.append_eos else 0)
                for text in texts
            ]
        else:
            self.sizes = [len(tokenize_line(text)) for text in texts]

        self.sizes = np.array(self.sizes, dtype=np.int32)

        assert len(self.utt_ids) == len(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    def filter_and_reorder(self, indices):
        assert isinstance(indices, (list, np.ndarray))
        indices = np.array(indices)
        assert all(indices < self.size) and all(indices >= 0)
        assert len(np.unique(indices)) == len(indices), "Duplicate elements in indices."
        self.utt_ids = [self.utt_ids[i] for i in indices]
        self.texts = [self.texts[i] for i in indices]
        self.sizes = self.sizes[indices]
        self.size = len(self.utt_ids)

    def __getitem__(self, i):
        self.check_index(i)
        if self.dictionary is not None:
            token_text = self.dictionary.wordpiece_encode(self.texts[i])
            tensor_item = self.dictionary.encode_line(
                token_text, add_if_not_exist=False, append_eos=self.append_eos
            ).long()
        else:
            tensor_item = None
        return (tensor_item, self.texts[i])

    def __len__(self):
        return self.size

    @staticmethod
    def exists(path):
        return os.path.exists(path)
