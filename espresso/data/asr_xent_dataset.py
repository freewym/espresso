# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import List, Optional

import numpy as np

import torch
import torch.nn.functional as F

from fairseq.data import FairseqDataset, data_utils

import espresso.tools.utils as speech_utils

try:
    import kaldi_io
except ImportError:
    raise ImportError("Please install kaldi_io with: pip install kaldi_io")


logger = logging.getLogger(__name__)


def collate(
    samples,
    pad_idx,
    chunk_width,
    chunk_left_context,
    chunk_right_context,
    label_delay,
    seed,
    epoch,
    pad_to_length=None,
    pad_to_multiple=1,
    src_bucketed=False,
    random_chunking=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, pad_to_length=None):
        if key == "source":
            return speech_utils.collate_frames(
                [s[key] for s in samples], 0.0,
                pad_to_length=pad_to_length,
                pad_to_multiple=pad_to_multiple,
            )
        elif key == "target":
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                pad_idx=pad_idx, eos_idx=None,
                left_pad=False, move_eos_to_beginning=False,
                pad_to_length=pad_to_length,
                pad_to_multiple=pad_to_multiple,
            )
        else:
            raise ValueError("Invalid key.")

    def chunking(src_item, tgt_item, tgt_start):
        # make a src chunk in the range [begin_src, end_src)
        begin_src = max(0, tgt_start + label_delay - chunk_left_context)
        # ok if end_src past the end of utterance
        end_src = tgt_start + label_delay + chunk_width + chunk_right_context
        # replication pad if necessary
        left_pad = max(0, chunk_left_context - tgt_start - label_delay)
        right_pad = max(0, end_src - src_item.size(0))
        src_item = src_item[begin_src: end_src]
        if left_pad > 0 or right_pad > 0:
            src_item = F.pad(
                src_item.t().unsqueeze(0), (left_pad, right_pad), mode="replicate",
            ).squeeze(0).t()

        if tgt_item is not None:
            # make a tgt chunk in the range [begin_tgt, end_tgt)
            begin_tgt = tgt_start
            end_tgt = tgt_start + chunk_width  # ok if past the end of utterance
            # replication pad if necessary
            right_pad = max(0, end_tgt - tgt_item.size(0))
            tgt_item = tgt_item[begin_tgt: end_tgt]
            if right_pad > 0:
                tgt_item = torch.cat(
                    (tgt_item, tgt_item.new_full((right_pad,), pad_idx)), 0
                )
        return src_item, tgt_item

    if chunk_width is None or random_chunking:
        if chunk_width is not None:  # usually for chunk-wise train data
            # no need to sort as all chunks have exactly the same length
            for s in samples:
                with data_utils.numpy_seed(seed, epoch, s["id"]):
                    # generate a chunk by sampling the index of its first label
                    f = np.random.randint(s["source"].size(0) - chunk_width + 1)
                s["source"], s["target"] = chunking(s["source"], s["target"], f)
        elif label_delay != 0:  # shift source according to label_delay
            if label_delay > 0:
                left_pad, right_pad = 0, label_delay
            else:
                left_pad, right_pad = -label_delay, 0
            for s in samples:
                src_item = s["source"]
                src_item = F.pad(
                    src_item.t().unsqueeze(0), (left_pad, right_pad), mode="replicate",
                ).squeeze(0).t()
                if label_delay > 0:
                    s["source"] = src_item[label_delay:]
                else:
                    s["source"] = src_item[: label_delay]

        if pad_to_length is not None or src_bucketed:
            src_lengths = torch.IntTensor(
                [s["source"].ne(0.0).any(dim=1).int().sum() for s in samples]
            )
        else:
            src_lengths = torch.IntTensor([s["source"].size(0) for s in samples])
        id = torch.LongTensor([s["id"] for s in samples])
        utt_id = [s["utt_id"] for s in samples]
        src_frames = merge(
            "source",
            pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
        )

        target = None
        if samples[0].get("target", None) is not None:
            target = merge(
                "target",
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
            ntokens = sum(s["target"].ne(pad_idx).int().sum().item() for s in samples)
        else:
            ntokens = src_lengths.sum().item()

        text = None
        if samples[0].get("text", None) is not None:
            text = [s["text"] for s in samples]

        if chunk_width is None:  # for whole utterances (i.e., no chunking)
            # sort by descending source length
            src_lengths, sort_order = src_lengths.sort(descending=True)
            id = id.index_select(0, sort_order)
            utt_id = [utt_id[i] for i in sort_order.numpy()]
            src_frames = src_frames.index_select(0, sort_order)
            if target is not None:
                target = target.index_select(0, sort_order)
            if text is not None:
                text = [text[i] for i in sort_order.numpy()]

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
    else:  # sequential chunking, usually for chunk-wise test data
        if pad_to_length is not None or src_bucketed:
            src_lengths = torch.IntTensor([
                s["source"].ne(0.0).any(dim=1).int().sum() for s in samples
            ])
        else:
            src_lengths = torch.IntTensor([s["source"].size(0) for s in samples])
        id = torch.LongTensor([s["id"] for s in samples])
        utt_id = [s["utt_id"] for s in samples]
        ori_source = [s["source"] for s in samples]
        ori_target = [s["target"] for s in samples]
        text = None
        if samples[0].get("text", None) is not None:
            text = [s["text"] for s in samples]
        max_length = max(src.size(0) for src in ori_source)
        num_chunks = (max_length + chunk_width - 1) // chunk_width
        batches = []
        for k in range(num_chunks):
            f = k * chunk_width
            for i, s in enumerate(samples):
                if f < src_lengths[i].item():
                    s["source"], s["target"] = chunking(ori_source[i], ori_target[i], f)
                else:
                    s["source"] = ori_source[i].new_zeros(
                        chunk_width + chunk_left_context + chunk_right_context, ori_source[i].size(1)
                    )
                    s["target"] = (
                        ori_target[i].new_full((chunk_width,), pad_idx)
                        if ori_target[i] is not None
                        else None
                    )
            src_frames = merge(
                "source",
                pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
            )
            src_chunk_lengths = torch.IntTensor([s["source"].size(0) for s in samples])

            target = None
            if samples[0].get("target", None) is not None:
                target = merge(
                    "target",
                    pad_to_length=pad_to_length["target"]
                    if pad_to_length is not None
                    else None,
                )
                ntokens = sum(s["target"].ne(pad_idx).int().sum().item() for s in samples)
            else:
                ntokens = src_lengths.sum().item()

            batch = {
                "id": id,
                "utt_id": utt_id,
                "nsentences": len(samples) if k == 0 else 0,
                "ntokens": ntokens,
                "net_input": {
                    "src_tokens": src_frames,
                    "src_lengths": src_chunk_lengths,
                },
                "target": target,
                "text": text,
            }
            batches.append(batch)
        return batches


class AliScpCachedDataset(torch.utils.data.Dataset):
    """
    A dataset for alignments prepared in Kaldi scp format (e.g., ali.scp).
    This class loads a batch of feature matrices (specified as *cache_size*)
    every time an entry is inquired. The inquire order should be known in advance.
    It balances the I/O efficiency and memory usage.
    """

    def __init__(
        self,
        utt_ids: List[str],
        rxfiles: List[str],
        utt2num_frames: Optional[List[int]] = None,
        ordered_prefetch=False,
        cache_size=327680,
    ):
        super().__init__()
        assert len(utt_ids) == len(rxfiles)
        self.dtype = np.int16
        self.utt_ids = utt_ids
        self.rxfiles = rxfiles
        self.size = len(utt_ids)  # number of utterances
        self.sizes = []  # length of each utterance
        if utt2num_frames is not None and len(utt2num_frames) > 0:
            assert len(utt2num_frames) == self.size
            self.sizes = utt2num_frames

        if len(self.sizes) == 0:
            for rxfile in self.rxfiles:
                try:
                    ali = kaldi_io.read_vec_int(rxfile)
                except Exception:
                    raise Exception("failed to read int vector {}.".format(rxfile))
                assert ali is not None and isinstance(ali, np.ndarray)
                self.sizes.append(ali.shape[0])

        assert len(self.sizes) == self.size
        self.sizes = np.array(self.sizes, dtype=np.int32)

        self.cache = None
        self.cache_index = {}
        self.cache_size = cache_size  # in terms of number of examples
        self.start_pos_for_next_cache = 0
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

    def __getitem__(self, i):
        self.check_index(i)
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
            self.cache = np.empty(total_size, dtype=self.dtype)
            ptx = 0
            self.cache_index.clear()
            for idx in self.ordered_indices[pos_start: pos_end]:
                self.cache_index[idx] = ptx
                length = self.sizes[idx]
                dst = self.cache[ptx: ptx + length]
                np.copyto(dst, kaldi_io.read_vec_int(self.rxfiles[idx]))
                ptx += length

        ptx = self.cache_index[i]
        a = self.cache[ptx: ptx + self.sizes[i]].copy()
        return torch.from_numpy(a).long()

    def __len__(self):
        return self.size

    @staticmethod
    def exists(path):
        return os.path.exists(path)


class AsrXentDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        tgt (espresso.data.AliScpCachedDataset, optional): target alignment dataset to wrap
        tgt_sizes (List[int], optional): target sizes (num of states in the numerator graph)
        tgt_vocab_size (int, optional): used for setting padding index
        text  (torch.utils.data.Dataset, optional): text dataset to wrap
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
        pad_to_multiple (int, optional): pad src/tgt lengths to a multiple of this value
        seed (int, optional): random seed for generating a chunk from an utterance.
        chunk_width (int, optional): chunk width for chunk-wise training.
        chunk_left_context (int, optional): number of frames appended to the left of a chunk.
        chunk_right_context (int, optional): number of frames appended to the right of a chunk.
        label_delay (int, optional): offset of the alignments as prediction labels. Can be
            useful in archs such as asymmetric convolution, unidirectional LSTM, etc.
        random_chunking (bool, optional): wether do random chunking from utterance, or sequntially
            obtain chunks within each utterance. True for train and False for valid/test data.
    """

    def __init__(
        self,
        src,
        src_sizes,
        tgt: Optional[AliScpCachedDataset] = None,
        tgt_sizes=None,
        text=None,
        shuffle=True,
        num_buckets=0,
        pad_to_multiple=1,
        seed=1,
        chunk_width=None,
        chunk_left_context=None,
        chunk_right_context=None,
        label_delay=0,
        random_chunking=True,
    ):
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.text = text
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 1
        assert chunk_width is None or chunk_width > 0
        self.chunk_width = chunk_width
        assert chunk_left_context >= 0 and chunk_right_context >= 0
        self.chunk_left_context = chunk_left_context
        self.chunk_right_context = chunk_right_context
        assert (
            (label_delay < 0 and -label_delay <= chunk_right_context)
            or (label_delay >= 0 and (chunk_width is None or label_delay < chunk_width))
        )
        self.label_delay = label_delay
        self.random_chunking = random_chunking
        if self.tgt is not None:
            self._match_src_tgt()
        if self.text is not None:
            changed = self._match_src_text()
            if self.tgt is not None and changed:
                self._match_src_tgt()
        self.sizes = (
            np.vstack((self.src_sizes, self.tgt_sizes)).T
            if self.tgt_sizes is not None
            else self.src_sizes
        )

        if chunk_width is not None:
            # remove those whose lengths are shorter than chunk_size
            indices = np.flatnonzero(self.src.sizes >= chunk_width)
            if len(indices) < self.src.size:
                logger.warning(
                    "Removing {} examples whose lengths are shorter than chunk_size={}".format(
                        self.src.size - len(indices), chunk_width
                    )
                )
                self.src.filter_and_reorder(indices)
                if self.tgt is not None:
                    self.tgt.filter_and_reorder(indices)
                if self.text is not None:
                    self.text.filter_and_reorder(indices)
                logger.warning("Done removal. {} examples remaining".format(len(indices)))

        if num_buckets > 0:
            from fairseq.data import BucketPadLengthDataset
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
            if self.tgt is not None:
                self.tgt = BucketPadLengthDataset(
                    self.tgt,
                    sizes=self.tgt_sizes,
                    num_buckets=num_buckets,
                    pad_idx=self.dictionary.pad(),
                    left_pad=False,
                )
                self.tgt_sizes = self.tgt.sizes
                logger.info("bucketing target lengths: {}".format(list(self.tgt.buckets)))

            # determine bucket sizes using self.num_tokens, which will return
            # the padded lengths (thanks to FeatBucketPadLengthDataset)
            num_tokens = np.vectorize(self.num_tokens, otypes=[np.long])
            self.bucketed_num_tokens = num_tokens(np.arange(len(self.src)))
            self.buckets = [
                (None, num_tokens) for num_tokens in np.unique(self.bucketed_num_tokens)
            ]
        else:
            self.buckets = None
        self.pad_to_multiple = pad_to_multiple

    def _match_src_tgt(self):
        """Makes utterances in src and tgt the same order in terms of
        their utt_ids. Removes those that are only present in one of them."""
        assert self.tgt is not None
        if self.src.utt_ids == self.tgt.utt_ids:
            assert np.all(self.src.sizes == self.tgt.sizes), "frame and alignment lengths mismatch"
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
        assert np.all(self.src.sizes == self.tgt.sizes), "frame and alignment lengths mismatch"

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

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {"source": source_pad_to_length, "target": target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.


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

                - `target` (LongTensor): a padded 2D Tensor of indices in the
                  target alignments of shape `(bsz, tgt_len)`
                - `text` (List[str]): list of original text
        """
        # pad_idx=-100 matches the default in criterions
        return collate(
            samples,
            pad_idx=-100,
            chunk_width=self.chunk_width,
            chunk_left_context=self.chunk_left_context,
            chunk_right_context=self.chunk_right_context,
            label_delay=self.label_delay,
            seed=self.seed,
            epoch=self.epoch,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
            src_bucketed=(self.buckets is not None),
            random_chunking=self.random_chunking,
        )

    def num_tokens(self, index):
        """Return the number of frames in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        if self.chunk_width is None:
            return self.src_sizes[index]
        return self.chunk_width + self.chunk_left_context + self.chunk_right_context

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
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
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)

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
        if hasattr(self.src, "set_epoch"):
            self.src.set_epoch(epoch)
        if self.tgt is not None and hasattr(self.tgt, "set_epoch"):
            self.tgt.set_epoch(epoch)
