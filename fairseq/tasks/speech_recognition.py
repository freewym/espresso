# Copyright (c) 2018-present, Yiming Wang
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import numpy as np
import os

from fairseq import options
from fairseq.data import (
    data_utils, TokenDictionary, SpeechDataset, ConcatDataset,
    TokenTextDataset, ScpCachedDataset
)

from . import FairseqTask, register_task


@register_task('speech_recognition')
class SpeechRecognitionTask(FairseqTask):
    """
    Translate from speech (source) to token text (target).

    Args:
        dict (Dictionary): dictionary for the output tokens

    .. note::

        The speech recognition task is compatible with :mod:`train.py <train>`,
        :mod:`generate.py <generate>` and :mod:`interactive.py <interactive>`.

    The speech_recognition task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.speech_recognition_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('scp_files', nargs='+', help='path(s) to scp file(s)')
        parser.add_argument('text_files', nargs='+', help='path(s) to text file(s)')
        parser.add_argument('--dict', default=None, type=str,
                            help='path to the dictionary')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')

    def __init__(self, args, dict):
        super().__init__(args)
        self.dict = dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        # load dictionaries
        dict_path = os.path.join(os.path.dirname(args.text_files[0]),
            'dict.txt') if args.dict is None else args.dict
        dict = TokenDictionary.load(dict_path)
        print('| dictionary: {} types'.format(len(dict)))

        return cls(args, dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        src_datasets = []
        tgt_datasets = []

        assert len(self.args.scp_files) == len(self.args.text_files)
        file_pairs = zip(self.args.scp_files, self.args.text_file)
        for scp, text in enumerate(file_pairs):
            assert ScpCachedDataset.exists(scp) and TokenTextDataset.exists(text)
            src_datasets.append(ScpCachedDataset(scp, ordered_indices=True))
            tgt_datasets.append(TokenTextDataset(text, self.dict))
            print('| {} {} examples'.format(scp, len(src_datasets[-1])))
            print('| {} {} examples'.format(text, len(tgt_datasets[-1])))

            if not combine:
                break

        assert len(src_datasets) == len(tgt_datasets)

        if len(src_datasets) == 1:
            src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
        else:
            sample_ratios = [1] * len(src_datasets)
            sample_ratios[0] = self.args.upsample_primary
            src_dataset = ConcatDataset(src_datasets, sample_ratios)
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

        self.datasets[split] = SpeechDataset(
            src_dataset, src_dataset.sizes,
            tgt_dataset, tgt_dataset.sizes, self.dict,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
        )

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return None

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.dict
