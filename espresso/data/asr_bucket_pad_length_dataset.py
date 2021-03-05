# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F

from fairseq.data import BucketPadLengthDataset


class FeatBucketPadLengthDataset(BucketPadLengthDataset):
    """
    Bucket and pad item lengths to the nearest bucket size for float tensors (features)
    of shape `(length, feat_dim)`. This can be used to
    reduce the number of unique batch shapes, which is important on TPUs since
    each new batch shape requires a recompilation.

    Args:
        dataset (FairseqDatset): dataset to bucket
        sizes (List[int]): all item sizes
        num_buckets (int): number of buckets to create
        pad_idx (float, optional): padding value
        left_pad (bool, optional): if True, pad on the left; otherwise right pad
    """

    def __init__(
        self,
        dataset,
        sizes,
        num_buckets,
        pad_idx=None,
        left_pad=False,
    ):
        super().__init__(dataset, sizes, num_buckets, pad_idx, left_pad)
        if self.pad_idx is None:
            self.pad_value = 0.0
        else:
            self.pad_value = pad_idx
        self.utt_ids = self.dataset.utt_ids

    def __getitem__(self, index):
        item = self.dataset[index]
        bucket_size = self._bucketed_sizes[index]
        num_pad = bucket_size - item.size(-1)
        return F.pad(
            item,
            (0, 0, num_pad if self.left_pad else 0, 0 if self.left_pad else num_pad),
            value=self.pad_value,
        )


class TextBucketPadLengthDataset(BucketPadLengthDataset):
    """
    Bucket and pad item lengths to the nearest bucket size for :class:`AsrTextDataset`.
    The main difference of this class from :class:`BucketPadLengthDataset` is that
    here we only bucket the first element in the returned tuple of
    :func:`AsrTextDataset.__getitem__`. This can be used to
    reduce the number of unique batch shapes, which is important on TPUs since
    each new batch shape requires a recompilation.

    Args:
        dataset (FairseqDatset): dataset to bucket
        sizes (List[int]): all item sizes
        num_buckets (int): number of buckets to create
        pad_idx (float, optional): padding value
        left_pad (bool, optional): if True, pad on the left; otherwise right pad
    """

    def __init__(
        self,
        dataset,
        sizes,
        num_buckets,
        pad_idx=None,
        left_pad=False,
    ):
        super().__init__(dataset, sizes, num_buckets, pad_idx, left_pad)
        self.utt_ids = self.dataset.utt_ids

    def __getitem__(self, index):
        item = self.dataset[index][0]
        bucket_size = self._bucketed_sizes[index]
        num_pad = bucket_size - item.size(-1)
        return (
            F.pad(
                item,
                (num_pad if self.left_pad else 0, 0 if self.left_pad else num_pad),
                value=self.pad_idx,
            ),
            self.dataset[index][1],
            self.dataset[index][2]
        )
