# Copyright (c) Yiming Wang, Yiwen Shao
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import re
from typing import List

import numpy as np

import torch

from fairseq.data import FairseqDataset

import espresso.tools.utils as speech_utils


logger = logging.getLogger(__name__)


def collate(samples, src_bucketed=False):
    try:
        from pychain import ChainGraphBatch
    except ImportError:
        raise ImportError("Please install OpenFST and PyChain by `make openfst pychain` after entering espresso/tools")

    if len(samples) == 0:
        return {}

    def merge(key):
        if key == "source":
            return speech_utils.collate_frames([s[key] for s in samples], 0.0)
        elif key == "target":
            max_num_transitions = max(s["target"].num_transitions for s in samples)
            max_num_states = max(s["target"].num_states for s in samples)
            return ChainGraphBatch(
                [s["target"] for s in samples],
                max_num_transitions=max_num_transitions,
                max_num_states=max_num_states,
            )
        else:
            raise ValueError("Invalid key.")

    id = torch.LongTensor([s["id"] for s in samples])
    src_frames = merge("source")
    # sort by descending source length
    if src_bucketed:
        src_lengths = torch.IntTensor([
            s["source"].ne(0.0).any(dim=1).int().sum() for s in samples
        ])
    else:
        src_lengths = torch.IntTensor([s["source"].size(0) for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    utt_id = [samples[i]["utt_id"] for i in sort_order.numpy()]
    src_frames = src_frames.index_select(0, sort_order)
    ntokens = src_lengths.sum().item()

    target = None
    if samples[0].get("target", None) is not None:
        target = merge("target")
        target.reorder(sort_order)

    text = None
    if samples[0].get("text", None) is not None:
        text = [samples[i]["text"] for i in sort_order.numpy()]

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
        "text": text,
    }
    return batch


class NumeratorGraphDataset(FairseqDataset):
    """
    A dataset of numerator graphs for LF-MMI. It loads all graphs into memory at
    once as its relatively small.
    """

    def __init__(self, utt_ids: List[str], rxfiles: List[str]):
        super().__init__()
        self.read_fsts(utt_ids, rxfiles)

    def read_fsts(self, utt_ids: List[str], rxfiles: List[str]):
        try:
            from pychain import ChainGraph
            import simplefst
        except ImportError:
            raise ImportError("Please install OpenFST and PyChain by `make openfst pychain` after entering espresso/tools")

        self.utt_ids = []
        self.rxfiles = []
        self.size = 0  # number of utterances
        self.sizes = []  # num of states in each fst
        self.numerator_graphs = []
        for i, rxfile in enumerate(rxfiles):
            file_path, offset = self._parse_rxfile(rxfile)
            fst = simplefst.StdVectorFst.read_ark(file_path, offset)
            graph = ChainGraph(fst, leaky_mode="uniform")
            if not graph.is_empty:  # skip empty graphs
                self.utt_ids.append(utt_ids[i])
                self.rxfiles.append(rxfile)
                self.size += 1
                self.sizes.append(fst.num_states())
                self.numerator_graphs.append(graph)
        self.sizes = np.array(self.sizes, dtype=np.int32)

    def _parse_rxfile(self, rxfile):
        # separate offset from filename
        m = re.match(r"(\S+):([0-9]+)", rxfile)
        assert m is not None, "Illegal rxfile: {}".format(rxfile)
        return m.group(1), int(m.group(2))

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    def filter_and_reorder(self, indices):
        assert isinstance(indices, (list, np.ndarray))
        indices = np.array(indices)
        assert all(indices < len(self.utt_ids)) and all(indices >= 0)
        assert len(np.unique(indices)) == len(indices), \
            "Duplicate elements in indices."
        self.utt_ids = [self.utt_ids[i] for i in indices]
        self.rxfiles = [self.rxfiles[i] for i in indices]
        self.numerator_graphs = [self.numerator_graphs[i] for i in indices]
        self.sizes = self.sizes[indices]
        self.size = len(self.utt_ids)

    def __getitem__(self, i):
        self.check_index(i)
        return self.numerator_graphs[i]

    def __len__(self):
        return self.size

    @staticmethod
    def exists(path):
        return os.path.exists(path)


class AsrChainDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        tgt (espresso.data.NumeratorGraphDataset, optional): target numerator graph dataset to wrap
        tgt_sizes (List[int], optional): target sizes (num of states in the numerator graph)
        text  (torch.utils.data.Dataset, optional): text dataset to wrap
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
    """

    def __init__(
        self, src, src_sizes, tgt=None, tgt_sizes=None, text=None, shuffle=True,
        num_buckets=0,
    ):
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.text = text
        self.shuffle = shuffle
        self.epoch = 1
        num_before_matching = len(self.src.utt_ids)
        if self.tgt is not None:
            self._match_src_tgt()
        if self.text is not None:
            changed = self._match_src_text()
            if self.tgt is not None and changed:
                self._match_src_tgt()
        num_after_matching = len(self.src.utt_ids)
        num_removed = num_before_matching - num_after_matching
        if num_removed > 0:
            logger.warning(
                "Removed {} examples due to empty numerator graphs or missing entries, "
                "{} remaining".format(num_removed, num_after_matching)
            )

        if num_buckets > 0:
            from espresso.data import FeatBucketPadLengthDataset
            self.src = FeatBucketPadLengthDataset(
                self.src,
                sizes=self.src_sizes,
                num_buckets=num_buckets,
                pad_idx=0.0,
                left_pad=False,
            )
            self.src_sizes = self.src.sizes
            logger.info("bucketing source lengths: {}".format(list(self.src.buckets)))

            # determine bucket sizes using self.num_tokens, which will return
            # the padded lengths (thanks to FeatBucketPadLengthDataset)
            num_tokens = np.vectorize(self.num_tokens, otypes=[np.long])
            self.bucketed_num_tokens = num_tokens(np.arange(len(self.src)))
            self.buckets = [
                (None, num_tokens)
                for num_tokens in np.unique(self.bucketed_num_tokens)
            ]
        else:
            self.buckets = None

    def _match_src_tgt(self):
        """Makes utterances in src and tgt the same order in terms of
        their utt_ids. Removes those that are only present in one of them."""
        assert self.tgt is not None
        if self.src.utt_ids == self.tgt.utt_ids:
            return
        tgt_utt_ids_set = set(self.tgt.utt_ids)
        src_indices = [
            i for i, id in enumerate(self.src.utt_ids) if id in tgt_utt_ids_set
        ]
        self.src.filter_and_reorder(src_indices)
        self.src_sizes = np.array(self.src.sizes)
        try:
            tgt_indices = list(map(self.tgt.utt_ids.index, self.src.utt_ids))
        except ValueError:
            raise ValueError(
                "Unable to find some utt_id(s) in tgt. which is unlikely to happen. "
                "Something must be wrong."
            )
        self.tgt.filter_and_reorder(tgt_indices)
        self.tgt_sizes = np.array(self.tgt.sizes)
        assert self.src.utt_ids == self.tgt.utt_ids

    def _match_src_text(self):
        """Makes utterances in src and text the same order in terms of
        their utt_ids. Removes those that are only present in one of them."""
        assert self.text is not None
        if self.src.utt_ids == self.text.utt_ids:
            return False
        text_utt_ids_set = set(self.text.utt_ids)
        src_indices = [
            i for i, id in enumerate(self.src.utt_ids) if id in text_utt_ids_set
        ]
        self.src.filter_and_reorder(src_indices)
        self.src_sizes = np.array(self.src.sizes)
        try:
            text_indices = list(map(self.text.utt_ids.index, self.src.utt_ids))
        except ValueError:
            raise ValueError(
                "Unable to find some utt_id(s) in text. which is unlikely to happen. "
                "Something must be wrong."
            )
        self.text.filter_and_reorder(text_indices)
        assert self.src.utt_ids == self.text.utt_ids
        return True

    def get_batch_shapes(self):
        return self.buckets

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        text_item = self.text[index][1] if self.text is not None else None
        src_item = self.src[index]
        example = {
            "id": index,
            "utt_id": self.src.utt_ids[index],
            "source": src_item,
            "target": tgt_item,
            "text": text_item,
        }
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `utt_id` (List[str]): list of utterance ids
                - `nsentences` (int): batch size
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (FloatTensor): a padded 3D Tensor of features in
                    the source of shape `(bsz, src_len, feat_dim)`.
                  - `src_lengths` (IntTensor): 1D Tensor of the unpadded
                    lengths of each source sequence of shape `(bsz)`

                - `target` (ChainGraphBatch): an instance representing a batch of
                    numerator graphs
                - `text` (List[str]): list of original text
        """
        return collate(samples, src_bucketed=(self.buckets is not None))

    def num_tokens(self, index):
        """Return the number of frames in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.src_sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[
                    np.argsort(self.tgt_sizes[indices], kind="mergesort")
                ]
            return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
        else:
            # sort by bucketed_num_tokens, which is padded_src_len
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]

    @property
    def supports_prefetch(self):
        return getattr(self.src, "supports_prefetch", False)

    def prefetch(self, indices):
        """Only prefetch src."""
        self.src.prefetch(indices)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.epoch = epoch
        if hasattr(self.src, "set_epoch"):
            self.src.set_epoch(epoch)
        if self.tgt is not None and hasattr(self.tgt, "set_epoch"):
            self.tgt.set_epoch(epoch)
