# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np

import torch

from fairseq.data import FairseqDataset, data_utils

import espresso.tools.utils as speech_utils
try:
    # TODO use pip install once it's available
    from espresso.tools.lhotse.lhotse import CutSet
except ImportError:
    raise ImportError("Please install Lhotse by `make lhotse` after entering espresso/tools")


def collate(
    samples: List[Dict[str, Any]],
    pad_to_length: Optional[Dict[str, int]] = None,
    pad_to_multiple: int = 1,
) -> Dict[str, Any]:
    """Collate samples into a batch. We use :func:`speech_utils.collate_frames`
    to collate and pad input frames, and PyTorch's :func:`default_collate`
    to collate and pad target/supervisions (following the example provided in Lhotse).
    Samples in the batch are in descending order of their input frame lengths.
    It also allows to specify the padded input length and further enforce the length
    to be a multiple of `pad_to_multiple`
    """
    if len(samples) == 0:
        return {}

    id = torch.LongTensor([sample["id"] for sample in samples])
    src_frames = speech_utils.collate_frames(
        [sample["source"] for sample in samples], 0.0,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
        pad_to_multiple=pad_to_multiple,
    )
    # sort by descending source length
    if pad_to_length is not None:
        src_lengths = torch.IntTensor(
            [sample["source"].ne(0.0).any(dim=1).int().sum() for sample in samples]
        )
    else:
        src_lengths = torch.IntTensor([s["source"].size(0) for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    utt_id = [samples[i]["utt_id"] for i in sort_order.numpy()]
    src_frames = src_frames.index_select(0, sort_order)
    ntokens = src_lengths.sum().item()

    target = None
    if samples[0].get("target", None) is not None and len(samples[0].target) > 0:
        # reorder the list of samples to make things easier
        # (no need to reorder every element in target)
        samples = [samples[i] for i in sort_order.numpy()]

        from torch.utils.data._utils.collate import default_collate

        dataset_idx_to_batch_idx = {
            sample["target"][0]["sequence_idx"]: batch_idx
            for batch_idx, sample in enumerate(samples)
        }

        def update(d: Dict, **kwargs) -> Dict:
            for key, value in kwargs.items():
                d[key] = value
            return d

        target = default_collate([
            update(sup, sequence_idx=dataset_idx_to_batch_idx[sup["sequence_idx"]])
            for sample in samples
            for sup in sample["target"]
        ])

    batch = {
        "id": id,
        "utt_id": utt_id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_frames,
            "src_lengths": src_lengths,
        },
        "target": target,
    }
    return batch


class K2AsrDataset(FairseqDataset):
    """
    A K2 Dataset for ASR.

    Args:
        cuts (lhotse.CutSet): instance of Lhotse's CutSet to wrap
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        pad_to_multiple (int, optional): pad src lengths to a multiple of this value
    """

    def __init__(
        self,
        cuts: CutSet,
        shuffle=True,
        pad_to_multiple=1,
    ):
        self.cuts = cuts
        self.cut_ids = list(self.cuts.ids)
        self.src_sizes = np.array(
            [cut.num_frames if cut.has_features else cut.num_samples for cut in cuts]
        )
        self.tgt_sizes = None
        first_cut = next(iter(cuts))
        # assume all cuts have no supervisions if the first one does not
        if len(first_cut.supervisions) > 0:
            assert len(first_cut.supervisions) == 1, "Only single-supervision cuts are allowed"
            assert first_cut.frame_shift is not None, "features are not available in cuts"
            self.tgt_sizes = np.array(
                [
                    round(
                        cut.supervisions[0].trim(cut.duration).duration / cut.frame_shift
                    ) for cut in cuts
                ]
            )
        self.shuffle = shuffle
        self.epoch = 1
        self.sizes = (
            np.stack((self.src_sizes, self.tgt_sizes), axis=1)
            if self.tgt_sizes is not None
            else self.src_sizes
        )
        self.pad_to_multiple = pad_to_multiple
        self.feat_dim = self.cuts[self.cut_ids[0]].num_features

    def __getitem__(self, index):
        cut_id = self.cut_ids[index]
        cut = self.cuts[cut_id]
        features = torch.from_numpy(cut.load_features())

        example = {
            "id": index,
            "utt_id": cut_id,
            "source": features,
            "target": [
                {
                    "sequence_idx": index,
                    "text": sup.text,
                    "start_frame": round(sup.start / cut.frame_shift),
                    "num_frames": round(sup.duration / cut.frame_shift),
                }
                # CutSet's supervisions can exceed the cut, when the cut starts/ends in the middle
                # of a supervision (they would have relative times e.g. -2 seconds start, meaning
                # it started 2 seconds before the Cut starts). We use s.trim() to get rid of that
                # property, ensuring the supervision time span does not exceed that of the cut.
                for sup in (s.trim(cut.duration) for s in cut.supervisions)
            ]
        }
        return example

    def __len__(self):
        return len(self.cuts)

    def collater(
        self,
        samples: List[Dict[str, Any]],
        pad_to_length: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {"source": source_pad_to_length}
                to indicate the max length to pad to in source.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` -> LongTensor: example IDs in the original input order
                - `utt_id` -> List[str]: list of utterance ids
                - `nsentences` -> int: batch size
                - `ntokens` -> int: total number of tokens in the batch
                - `net_input` -> Dict: the input to the Model, containing keys:

                  - `src_tokens` -> FloatTensor: a padded 3D Tensor of features in
                    the source of shape `(bsz, src_len, feat_dim)`.
                  - `src_lengths` -> IntTensor: 1D Tensor of the unpadded
                    lengths of each source sequence of shape `(bsz)`

                - `target` -> List[Dict[str, Any]]: a List representing a batch of
                    supervisions
        """
        return collate(
            samples, pad_to_length=pad_to_length, pad_to_multiple=self.pad_to_multiple,
        )

    def num_tokens(self, index):
        """Return the number of frames in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.src_sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        # sort by target length, then source length
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
        return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]

    @property
    def supports_prefetch(self):
        return False

    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            self.tgt_sizes,
            indices,
            max_sizes,
        )

    @property
    def supports_fetch_outside_dataloader(self):
        """Whether this dataset supports fetching outside the workers of the dataloader."""
        return False

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False  # to avoid running out of CPU RAM

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.epoch = epoch
