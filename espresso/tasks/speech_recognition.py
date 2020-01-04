# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch

from fairseq import options, search
from fairseq.data import ConcatDataset

from fairseq.tasks import FairseqTask, register_task

from espresso.data import (
    AsrDictionary,
    AsrTextDataset,
    ScpCachedDataset,
    SpeechDataset,
)


@register_task('speech_recognition_espresso')
class SpeechRecognitionEspressoTask(FairseqTask):
    """
    Transcribe from speech (source) to token text (target).

    Args:
        dictionary (~fairseq.data.AsrDictionary): dictionary for the output tokens
        word_dict (~fairseq.data.AsrDictionary): dictionary for the words
            (for decoding with word-based LMs)
        feat_in_channels (int): input feature channels

    .. note::

        The speech recognition task is compatible with :mod:`speech-train`,
        :mod:`speech-recognize` and :mod:`fairseq-interactive`.

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
                            help='path(s) to scp feature file(s) for training, '
                            'will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--train-text-files', nargs='+',
                            help='path(s) to text file(s) for training, where '
                            'each should matches with one in --train-feat-files, '
                            'will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--valid-feat-files', nargs='+',
                            help='path(s) to scp feature file(s) for validation')
        parser.add_argument('--valid-text-files', nargs='+',
                            help='path(s) to text file(s) for validation, where '
                            'each should matches with one in --valid-feat-files')
        parser.add_argument('--test-feat-files', nargs='+',
                            help='path(s) to scp feature file(s) for test')
        parser.add_argument('--test-text-files', nargs='*', default=None,
                            help='path(s) to text file(s) for test. if not None, '
                            'each one should matches with one in --test-feat-files')
        parser.add_argument('--train-subset-feat-files', nargs='+',
                            help='path(s) to scp feature file(s) for validation')
        parser.add_argument('--train-subset-text-files', nargs='+',
                            help='path(s) to text file(s) for validation, where '
                            'each should matches with one in --train-subset-feat-files')
        parser.add_argument('--dict', default=None, type=str,
                            help='path to the dictionary')
        parser.add_argument('--non-lang-syms', default=None, type=str,
                            help='path to a file listing non-linguistic symbols, e.g., <NOISE> '
                            'etc. One entry per line. To be filtered out when calculating WER/CER.')
        parser.add_argument('--word-dict', default=None, type=str,
                            help='path to the word dictionary. Only relevant for decoding')
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
        return AsrDictionary.load(filename, f_non_lang_syms=non_lang_syms)

    @classmethod
    def build_dictionary(cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        """Disable this method
        """
        raise NotImplementedError

    def __init__(self, args, dictionary, word_dict=None):
        super().__init__(args)
        self.dictionary = dictionary
        self.word_dict = word_dict
        self.feat_in_channels = args.feat_in_channels
        torch.backends.cudnn.deterministic = True
        # Compansate for the removel of :func:`torch.rand()` from
        # :func:`fairseq.distributed_utils.distributed_init()` by fairseq,
        # to make previous experiments reproducible.
        torch.rand(1)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        # load dictionaries
        dict_path = os.path.join(os.path.dirname(args.train_text_files[0]), 'dict.txt') \
            if args.dict is None and args.train_text_files is not None else args.dict
        assert dict_path is not None, 'Please specify --dict'
        dictionary = cls.load_dictionary(dict_path, non_lang_syms=args.non_lang_syms)
        print('| dictionary: {} types'.format(len(dictionary)))
        if args.word_dict is not None:
            word_dict = cls.load_dictionary(args.word_dict)
            print('| word dictionary: {} types'.format(len(word_dict)))
            return cls(args, dictionary, word_dict)

        else:
            return cls(args, dictionary)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        src_datasets = []
        tgt_datasets = []

        if split == 'train':
            feat_files = self.args.train_feat_files
            text_files = self.args.train_text_files
            assert len(feat_files) > 0 and len(feat_files) == len(text_files)
            feat_files = [feat_files[epoch % len(feat_files)]]
            text_files = [text_files[epoch % len(text_files)]]
        elif split == 'valid':
            feat_files = self.args.valid_feat_files
            text_files = self.args.valid_text_files
        elif split == 'test':
            feat_files = self.args.test_feat_files
            text_files = self.args.test_text_files  # can be empty
            if text_files is None:
                text_files = [None] * len(feat_files)
        elif split == 'train_subset':
            feat_files = self.args.train_subset_feat_files
            text_files = self.args.train_subset_text_files
        else:
            raise ValueError('split should be one of "train", "valid", "test", "train_subset"')

        assert len(feat_files) > 0 and len(feat_files) == len(text_files)
        file_pairs = zip(feat_files, text_files)
        for feat, text in file_pairs:
            assert ScpCachedDataset.exists(feat), feat + ' does not exists'
            assert text is None or AsrTextDataset.exists(text), text + ' does not exists'
            src_datasets.append(ScpCachedDataset(feat, ordered_prefetch=True))
            print('| {} {} examples'.format(feat, len(src_datasets[-1])))
            if text is not None:
                tgt_datasets.append(AsrTextDataset(text, self.dictionary))
                print('| {} {} examples'.format(text, len(tgt_datasets[-1])))

            if not combine:
                break

        if len(tgt_datasets) > 0:
            assert len(src_datasets) == len(tgt_datasets)

        self.feat_dim = src_datasets[0].feat_dim

        if len(src_datasets) == 1:
            src_dataset = src_datasets[0]
            tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
        else:
            for i in range(1, len(src_datasets)):
                assert self.feat_dim == src_datasets[i].feat_dim, \
                    'feature dimension does not match across multiple scp files'
            sample_ratios = [1] * len(src_datasets)
            sample_ratios[0] = self.args.upsample_primary
            src_dataset = ConcatDataset(src_datasets, sample_ratios)
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios) \
                if len(tgt_datasets) > 0 else None

        self.datasets[split] = SpeechDataset(
            src_dataset, src_dataset.sizes,
            tgt_dataset, tgt_dataset.sizes if tgt_dataset is not None else None,
            self.dictionary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
        )

        # update the counts of <eos> and <unk> in dictionary with training data
        if split == 'train':
            self.dictionary.count[self.dictionary.eos()] = len(tgt_dataset)
            unk_count = 0
            for i in range(len(tgt_dataset)):
                unk_count += (tgt_dataset[i][0] == self.dictionary.unk()).int().sum().item()
            self.dictionary.count[self.dictionary.unk()] = unk_count

    def build_generator(self, args):
        if args.score_reference:
            args.score_reference = False
            print('| --score-reference is not applicable to speech recognition,'
                  ' ignoring it.')
        from fairseq.sequence_generator import SequenceGenerator

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, 'sampling', False)
        sampling_topk = getattr(args, 'sampling_topk', -1)
        sampling_topp = getattr(args, 'sampling_topp', -1.0)
        diverse_beam_groups = getattr(args, 'diverse_beam_groups', -1)
        diverse_beam_strength = getattr(args, 'diverse_beam_strength', 0.5),
        match_source_len = getattr(args, 'match_source_len', False)
        diversity_rate = getattr(args, 'diversity_rate', -1)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError('Provided Search parameters are mutually exclusive.')
        assert sampling_topk < 0 or sampling, '--sampling-topk requires --sampling'
        assert sampling_topp < 0 or sampling, '--sampling-topp requires --sampling'

        if sampling:
            search_strategy = search.Sampling(self.target_dictionary, sampling_topk, sampling_topp)
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength)
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary, min_len_a=1, min_len_b=0, max_len_a=1, max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(self.target_dictionary, diversity_rate)
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        return SequenceGenerator(
            self.target_dictionary,
            beam_size=getattr(args, 'beam', 5),
            max_len_a=getattr(args, 'max_len_a', 0),
            max_len_b=getattr(args, 'max_len_b', 200),
            min_len=getattr(args, 'min_len', 1),
            normalize_scores=(not getattr(args, 'unnormalized', False)),
            len_penalty=getattr(args, 'lenpen', 1),
            unk_penalty=getattr(args, 'unkpen', 0),
            temperature=getattr(args, 'temperature', 1.),
            match_source_len=getattr(args, 'match_source_len', False),
            no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
            search_strategy=search_strategy,
            eos_factor=getattr(args, 'eos_factor', None),
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return SpeechDataset(src_tokens, src_lengths)

    def inference_step(self, generator, models, sample, prefix_tokens=None, lm_weight=0.0):
        with torch.no_grad():
            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, lm_weight=lm_weight,
            )

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.AsrDictionary`."""
        return self.dictionary

    @property
    def word_dictionary(self):
        """Return the target :class:`~fairseq.data.AsrDictionary`."""
        return self.word_dict
