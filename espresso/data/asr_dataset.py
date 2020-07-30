# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np
import torch

from fairseq.data import data_utils, FairseqDataset

import espresso.tools.utils as speech_utils


logger = logging.getLogger(__name__)


def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    src_bucketed=False,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        if key == 'source':
            return speech_utils.collate_frames(
                [s[key] for s in samples], 0.0, left_pad,
                pad_to_length=pad_to_length,
            )
        elif key == 'target' or key == 'prev_output_tokens':
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                pad_idx, eos_idx, left_pad, move_eos_to_beginning,
                pad_to_length=pad_to_length,
            )
        else:
            raise ValueError('Invalid key.')

    id = torch.LongTensor([s['id'] for s in samples])
    src_frames = merge(
        'source', left_pad=left_pad_source,
        pad_to_length=pad_to_length['source'] if pad_to_length is not None else None,
    )
    # sort by descending source length
    if pad_to_length is not None or src_bucketed:
        src_lengths = torch.IntTensor([
            s['source'].ne(0.0).any(dim=1).int().sum() for s in samples
        ])
    else:
        src_lengths = torch.IntTensor([s['source'].size(0) for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    utt_id = [samples[i]['utt_id'] for i in sort_order.numpy()]
    src_frames = src_frames.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge(
            'target', left_pad=left_pad_target,
            pad_to_length=pad_to_length['target'] if pad_to_length is not None else None,
        )
        target = target.index_select(0, sort_order)
        ntokens = sum(s['target'].ne(pad_idx).int().sum().item() for s in samples)

        if samples[0].get('prev_output_tokens', None) is not None:
            prev_output_tokens = merge('prev_output_tokens', left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length['target'] if pad_to_length is not None else None,
            )
    else:
        ntokens = src_lengths.sum().item()

    target_raw_text = None
    if samples[0].get('target_raw_text', None) is not None:
        target_raw_text = [samples[i]['target_raw_text'] for i in sort_order.numpy()]

    batch = {
        'id': id,
        'utt_id': utt_id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_frames,
            'src_lengths': src_lengths,
        },
        'target': target,
        'target_raw_text': target_raw_text,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens.index_select(0, sort_order)
    return batch


class AsrDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        dictionary (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
        src_lang_id (int, optional): source language ID, if set, the collated batch
            will contain a field 'src_lang_id' in 'net_input' which indicates the
            source language of the samples.
        tgt_lang_id (int, optional): target language ID, if set, the collated batch
            will contain a field 'tgt_lang_id' which indicates the target language
            of the samples.
    """

    def __init__(
        self, src, src_sizes,
        tgt=None, tgt_sizes=None, dictionary=None,
        left_pad_source=False, left_pad_target=False,
        shuffle=True, input_feeding=True,
        num_buckets=0,
        src_lang_id=None,
        tgt_lang_id=None,
    ):
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.dictionary = dictionary
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.src_lang_id = src_lang_id
        self.tgt_lang_id = tgt_lang_id
        if self.tgt is not None:
            self._match_src_tgt()

        if num_buckets > 0:
            from espresso.data import FeatBucketPadLengthDataset, TextBucketPadLengthDataset
            self.src = FeatBucketPadLengthDataset(
                self.src,
                sizes=self.src_sizes,
                num_buckets=num_buckets,
                pad_idx=0.0,
                left_pad=False,
            )
            self.src_sizes = self.src.sizes
            logger.info('bucketing source lengths: {}'.format(list(self.src.buckets)))
            if self.tgt is not None:
                self.tgt = TextBucketPadLengthDataset(
                    self.tgt,
                    sizes=self.tgt_sizes,
                    num_buckets=num_buckets,
                    pad_idx=self.dictionary.pad(),
                    left_pad=False,
                )
                self.tgt_sizes = self.tgt.sizes
                logger.info('bucketing target lengths: {}'.format(list(self.tgt.buckets)))

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
                'Unable to find some utt_id(s) in tgt. which is unlikely to happen. '
                'Something must be wrong.'
            )
        self.tgt.filter_and_reorder(tgt_indices)
        self.tgt_sizes = np.array(self.tgt.sizes)
        assert self.src.utt_ids == self.tgt.utt_ids

    def get_batch_shapes(self):
        return self.buckets

    def __getitem__(self, index):
        tgt_item = self.tgt[index][0] if self.tgt is not None else None
        raw_text_item = self.tgt[index][1] if self.tgt is not None else None
        src_item = self.src[index]
        example = {
            'id': index,
            'utt_id': self.src.utt_ids[index],
            'source': src_item,
            'target': tgt_item,
            'target_raw_text': raw_text_item,
        }
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `utt_id` (List[str]): list of utterance ids
                - `nsentences` (int): batch size
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (FloatTensor): a padded 3D Tensor of features in
                    the source of shape `(bsz, src_len, feat_dim)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (IntTensor): 1D Tensor of the unpadded
                    lengths of each source sequence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `target_raw_text` (List[str]): list of original text
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                  IDs of each sample in the batch
        """
        res = collate(
            samples,
            pad_idx=self.dictionary.pad(),
            eos_idx=self.dictionary.eos(),
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            src_bucketed=(self.buckets is not None),
        )
        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res['net_input']['src_tokens']
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res['net_input']['src_lang_id'] = torch.LongTensor(
                    [[self.src_lang_id]]
                ).expand(bsz, 1).to(src_tokens)
            if self.tgt_lang_id is not None:
                res['tgt_lang_id'] = torch.LongTensor(
                    [[self.tgt_lang_id]]
                ).expand(bsz, 1).to(src_tokens)
        return res

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
                    np.argsort(self.tgt_sizes[indices], kind='mergesort')
                ]
            return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]
        else:
            # sort by bucketed_num_tokens, which is padded_src_len
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind='mergesort')
            ]

    @property
    def supports_prefetch(self):
        return getattr(self.src, 'supports_prefetch', False)

    def prefetch(self, indices):
        """Only prefetch src."""
        self.src.prefetch(indices)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        if hasattr(self.src, 'set_epoch'):
            self.src.set_epoch(epoch)
        if self.tgt is not None and hasattr(self.tgt, 'set_epoch'):
            self.tgt.set_epoch(epoch)
