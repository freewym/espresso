# Copyright (c) 2018-present, Yiming Wang
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from . import data_utils, FairseqDataset
import speech_tools.utils as speech_utils


def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        if key == 'source':
            return speech_utils.collate_frames(
                [s[key] for s in samples], 0.0, left_pad,
            )
        elif key == 'target':
          return data_utils.collate_tokens(
              [s[key] for s in samples],
              pad_idx, eos_idx, left_pad, move_eos_to_beginning,
          )
        else:
            raise ValueError('Invalid key.')

    id = torch.LongTensor([s['id'] for s in samples])
    utt_id = [s['utt_id'] for s in samples]
    src_frames = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.IntTensor([s['source'].size(0) for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    utt_id = [utt_id[i] for i in sort_order.numpy()]
    src_frames = src_frames.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(s['source'].size(0) for s in samples)

    batch = {
        'id': id,
        'utt_id': utt_id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_frames, # key name kept due to
                                      # FairseqModel::forward(...,src_tokens,...)
            'src_lengths': src_lengths,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class SpeechDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of frames in the
            source (default: 1024).
        max_target_positions (int, optional): max number of tokens in the target
            sentence (default: 1024)
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True)
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for input feeding/teacher forcing
            (default: True)
    """

    def __init__(
        self, src, src_sizes,
        tgt=None, tgt_sizes=None, dict=None,
        left_pad_source=False, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True,
    ):
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.dict = dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        if self.tgt is not None:
            self._match_src_tgt()

    def _match_src_tgt(self):
        """Makes utterances in src and tgt the same order in terms of
        their utt_ids. Removes those that are only present in one of them."""
        assert self.tgt is not None
        if self.src.utt_ids == self.tgt.utt_ids:
            return
        tgt_utt_ids_set = set(self.tgt.utt_ids)
        src_indices = [i for i, id in enumerate(self.src.utt_ids) \
            if id in tgt_utt_ids_set]
        self.src.filter_and_reorder(src_indices)
        self.src_sizes = np.array(self.src.sizes)
        try:
            tgt_indices = list(map(self.tgt.utt_ids.index, self.src.utt_ids))
        except ValueError:
            print('Unable to find some utt_id(s) in tgt. which is unlikely to \
                happen. Something must be wrong.')
            raise
        self.tgt.filter_and_reorder(tgt_indices)
        self.tgt_sizes = np.array(self.tgt.sizes)
        assert self.src.utt_ids == self.tgt.utt_ids

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        return {
            'id': index,
            'utt_id': self.src.utt_ids[index],
            'source': src_item,
            'target': tgt_item,
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (FloatTensor): a padded 3D Tensor of features in
                    the source of shape `(bsz, src_len, feat_dim)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (IntTensor): 1D Tensor of the unpadded
                    lengths of each source sequence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.dict.pad(), eos_idx=self.dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

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
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return getattr(self.src, 'supports_prefetch', False)

    def prefetch(self, indices):
        """Only prefetch src."""
        self.src.prefetch(indices)

