# Copyright (c) 2018-present, Yiming Wang
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import os
import re

from fairseq import options, utils
from fairseq.data import (
    ConcatDataset,
    data_utils,
    ScpCachedDataset,
    SpeechDataset,
    TokenDictionary,
    TokenTextDataset,
)

from . import FairseqTask, register_task


@register_task('speech_recognition')
class SpeechRecognitionTask(FairseqTask):
    """
    Transcribe from speech (source) to token text (target).

    Args:
        dict (Dictionary): dictionary for the output tokens

    .. note::

        The speech recognition task is compatible with :mod:`speech-train`,
        :mod:`speech-recognition` and :mod:`fairseq-interactive`.

    The speech recognition task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.speech_recognition_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--train-feat-files', nargs='+',
                            help='path(s) to scp feature file(s) for training')
        parser.add_argument('--train-text-files', nargs='+',
                            help='path(s) to text file(s) for training, where '
                            'each should matches with one in --train-feat-files')
        parser.add_argument('--valid-feat-files', nargs='+',
                            help='path(s) to scp feature file(s) for validation')
        parser.add_argument('--valid-text-files', nargs='+',
                            help='path(s) to text file(s) for validation, where '
                            'each should matches with one in --valid-feat-files')
        parser.add_argument('--test-feat-files', nargs='+',
                            help='path(s) to scp feature file(s) for test')
        parser.add_argument('--test-text-files', nargs='+',
                            help='path(s) to text file(s) for test, where '
                            'each should matches with one in --test-feat-files')
        parser.add_argument('--dict', default=None, type=str,
                            help='path to the dictionary')
        parser.add_argument('--non-lang-syms', default=None, type=str,
                            help='list of non-linguistic symbols, e.g., <NOISE> '
                            'etc. To be filtered out when calculating WER/CER')
        parser.add_argument('--wer-output-filter', default=None, type=str,
                            help='path to wer_output_filter file for WER evaluation')
        parser.add_argument('--left-pad-source', default='False', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of frames in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--feat-in-channels', default=1, type=int, metavar='N',
                            help='feature input channels')
        # fmt: off

    @classmethod
    def load_dictionary(cls, filename, non_lang_syms=None):
        """Load the dictionary from the filename
        Args:
            filename (str): the filename
            non_lang_syms (str): non_lang_syms filename
        """
        return TokenDictionary.load(filename, f_non_lang_syms=non_lang_syms)

    @classmethod
    def build_dictionary(cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        """Disable this method
        """
        raise NotImplementedError

    @staticmethod
    def load_pretrained_model(path, dict_path, non_lang_syms=None,
        arg_overrides=None):
        model = utils.load_checkpoint_to_cpu(path)
        args = model['args']
        state_dict = model['model']
        args = utils.override_model_args(args, arg_overrides)
        dict = cls.load_dictionary(dict_path, non_lang_syms=non_lang_syms)

        task = SpeechRecognitionTask(args, dict)
        model = task.build_model(args)
        model.upgrade_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=True)
        return model

    def __init__(self, args, dict):
        super().__init__(args)
        self.dict = dict
        self.feat_in_channels = args.feat_in_channels

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
        dict = cls.load_dictionary(dict_path, non_lang_syms=args.non_lang_syms)
        print('| dictionary: {} types'.format(len(dict)))

        return cls(args, dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        src_datasets = []
        tgt_datasets = []

        if split == 'train':
            feat_files = self.args.train_feat_files
            text_files = self.args.train_text_files
            assert len(feat_files) > 0 and len(text_files) > 0
        elif re.match(r"^valid\d*$", split):
            m = re.match(r"^valid(\d*)$", split)
            idx = 0 if m.group(1) == '' else int(m.group(1))
            if idx >= len(self.args.valid_feat_files) or \
                idx >= len(self.args.valid_text_files):
                raise FileNotFoundError
            feat_files = [self.args.valid_feat_files[idx]]
            text_files = [self.args.valid_text_files[idx]]
            assert len(feat_files) > 0 and len(text_files) > 0
        elif split == 'test':
            feat_files = self.args.test_feat_files
            text_files = self.args.test_text_files
            assert len(feat_files) > 0 and len(text_files) > 0
        else:
            raise ValueError('split should be one of "train", "valid*", "test"')
        assert len(feat_files) == len(text_files)
        file_pairs = zip(feat_files, text_files)
        for feat, text in file_pairs:
            assert ScpCachedDataset.exists(feat) and TokenTextDataset.exists(text)
            src_datasets.append(ScpCachedDataset(feat, ordered_prefetch=True))
            tgt_datasets.append(TokenTextDataset(text, self.dict))
            print('| {} {} examples'.format(feat, len(src_datasets[-1])))
            print('| {} {} examples'.format(text, len(tgt_datasets[-1])))

            if not combine:
                break

        assert len(src_datasets) == len(tgt_datasets)

        self.feat_dim = src_datasets[0].feat_dim

        if len(src_datasets) == 1:
            src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
        else:
            for i in range(1, len(src_datasets)):
                assert self.feat_dim == src_datasets[i].feat_dim, \
                    'feature dimension does not match across multiple scp files'
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

    def build_generator(self, args):
        if args.score_reference:
            args.score_reference = False
            print('| --score-reference is not applicable to speech recognition,'
                ' ignoring it.')
        return super().build_generator(args)

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return SpeechDataset(src_tokens, src_lengths)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.dict
