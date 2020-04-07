# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import itertools
import json
import logging
import os

import torch

from fairseq import search, utils
from fairseq.data import ConcatDataset
from fairseq.logging import metrics
from fairseq.tasks import FairseqTask, register_task

from espresso.data import (
    AsrDataset,
    AsrDictionary,
    AsrTextDataset,
    FeatScpCachedDataset,
)


logger = logging.getLogger(__name__)


def get_asr_dataset_from_json(
    data_path, split, tgt_dict,
    combine, upsample_primary,
    max_source_positions, max_target_positions,
    seed=1, specaugment_config=None,
):
    """
    Parse data json and create dataset.
    See espresso/tools/asr_prep_json.py which pack json from raw files
    Json example:
    {
        "011c0202": {
            "feat": "fbank/raw_fbank_pitch_train_si284.1.ark:54819",
            "token_text": "T H E <space> H O T E L",
            "utt2num_frames": "693",
        },
        "011c0203": {
            ...
        }
    }
    """
    src_datasets = []
    tgt_datasets = []
    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")
        data_json_path = os.path.join(data_path, "{}.json".format(split_k))
        if not os.path.isfile(data_json_path):
            if k > 0:
                break
            else:
                raise FileNotFoundError("Dataset not found: {}".format(data_json_path))

        with open(data_json_path, "rb") as f:
            loaded_json = json.load(f, object_pairs_hook=OrderedDict)

        utt_ids, feats, token_text, utt2num_frames = [], [], [], []
        for utt_id, val in loaded_json.items():
            utt_ids.append(utt_id)
            feats.append(val["feat"])
            if "token_text" in val:
                token_text.append(val["token_text"])
            if "utt2num_frames" in val:
                utt2num_frames.append(int(val["utt2num_frames"]))

        assert len(utt2num_frames) == 0 or len(utt_ids) == len(utt2num_frames)
        src_datasets.append(FeatScpCachedDataset(
            utt_ids, feats, utt2num_frames=utt2num_frames, seed=seed,
            specaugment_config=specaugment_config if split == "train" else None,
            ordered_prefetch=True,
        ))
        if len(token_text) > 0:
            assert len(utt_ids) == len(token_text)
            assert tgt_dict is not None
            tgt_datasets.append(AsrTextDataset(utt_ids, token_text, tgt_dict))

        logger.info("{} {} examples".format(data_json_path, len(src_datasets[-1])))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    feat_dim = src_datasets[0].feat_dim

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        for i in range(1, len(src_datasets)):
            assert feat_dim == src_datasets[i].feat_dim, \
                "feature dimension does not match across multiple json files"
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return AsrDataset(
        src_dataset, src_dataset.sizes,
        tgt_dataset, tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=False,
        left_pad_target=False,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
    )


@register_task("speech_recognition_espresso")
class SpeechRecognitionEspressoTask(FairseqTask):
    """
    Transcribe from speech (source) to token text (target).

    Args:
        tgt_dict (~fairseq.data.AsrDictionary): dictionary for the output tokens
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
        parser.add_argument("data", help="path to data directory")
        parser.add_argument("--dict", default=None, type=str,
                            help="path to the dictionary")
        parser.add_argument("--non-lang-syms", default=None, type=str,
                            help="path to a file listing non-linguistic symbols, e.g., <NOISE> "
                            "etc. One entry per line. To be filtered out when calculating WER/CER.")
        parser.add_argument("--word-dict", default=None, type=str,
                            help="path to the word dictionary. Only relevant for decoding")
        parser.add_argument("--wer-output-filter", default=None, type=str,
                            help="path to wer_output_filter file for WER evaluation")
        parser.add_argument("--max-source-positions", default=1024, type=int, metavar="N",
                            help="max number of frames in the source sequence")
        parser.add_argument("--max-target-positions", default=1024, type=int, metavar="N",
                            help="max number of tokens in the target sequence")
        parser.add_argument("--upsample-primary", default=1, type=int,
                            help="amount to upsample primary dataset")
        parser.add_argument("--feat-in-channels", default=1, type=int, metavar="N",
                            help="feature input channels")
        parser.add_argument("--specaugment-config", default=None, type=str, metavar="EXPR",
                            help="SpecAugment config string. If not None and not empty, "
                            "then apply SpecAugment. Should be an evaluatable expression of "
                            "a python dict. See speech_tools.specaug_interpolate.specaug() for "
                            "all allowed arguments. Argments not appearing in this string "
                            "will take on their default values")
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
    def build_dictionary(
        cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8
    ):
        """Disable this method
        """
        raise NotImplementedError

    def __init__(self, args, tgt_dict, word_dict=None):
        super().__init__(args)
        self.tgt_dict = tgt_dict
        self.tgt_dict.build_tokenizer(args)
        self.tgt_dict.build_bpe(args)
        self.word_dict = word_dict
        self.feat_in_channels = args.feat_in_channels
        self.specaugment_config = args.specaugment_config
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
        # load dictionaries
        dict_path = os.path.join(args.data, "dict.txt") if args.dict is None else args.dict
        tgt_dict = cls.load_dictionary(dict_path, non_lang_syms=args.non_lang_syms)
        logger.info("dictionary: {} types".format(len(tgt_dict)))
        if args.word_dict is not None:
            word_dict = cls.load_dictionary(args.word_dict)
            logger.info("word dictionary: {} types".format(len(word_dict)))
            return cls(args, tgt_dict, word_dict)

        else:
            return cls(args, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        self.datasets[split] = get_asr_dataset_from_json(
            data_path, split, self.tgt_dict,
            combine=combine,
            upsample_primary=self.args.upsample_primary,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            seed=self.args.seed,
            specaugment_config=self.specaugment_config,
        )

        src_dataset = self.datasets[split].src
        self.feat_dim = src_dataset.feat_dim if not isinstance(src_dataset, ConcatDataset) \
            else src_dataset.datasets[0].feat_dim

        # update the counts of <eos> and <unk> in tgt_dict with training data
        if split == "train":
            tgt_dataset = self.datasets[split].tgt
            self.tgt_dict.count[self.tgt_dict.eos()] = len(tgt_dataset)
            unk_count = 0
            for i in range(len(tgt_dataset)):
                unk_count += (tgt_dataset[i][0] == self.tgt_dict.unk()).int().sum().item()
            self.tgt_dict.count[self.tgt_dict.unk()] = unk_count

    def build_generator(self, models, args):
        if getattr(args, "score_reference", False):
            args.score_reference = False
            logger.warning(
                "--score-reference is not applicable to speech recognition, ignoring it."
            )

        from fairseq.sequence_generator import SequenceGenerator

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
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
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        return SequenceGenerator(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            lm_weight = getattr(args, "lm_weight", 0.0),
            eos_factor=getattr(args, "eos_factor", None),
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return AsrDataset(src_tokens, src_lengths)

    def build_model(self, args):
        model = super().build_model(args)
        # build the greedy decoder for validation with WER
        from espresso.tools.simple_greedy_decoder import SimpleGreedyDecoder
        self.decoder_for_validation = SimpleGreedyDecoder(
            [model], self.target_dictionary, for_validation=True,
        )
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        (
            logging_output["word_error"], logging_output["word_count"],
            logging_output["char_error"], logging_output["char_count"],
        ) = self._inference_with_wer(self.decoder_for_validation, sample, model)
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        word_error = sum(log.get("word_error", 0) for log in logging_outputs)
        word_count = sum(log.get("word_count", 0) for log in logging_outputs)
        char_error = sum(log.get("char_error", 0) for log in logging_outputs)
        char_count = sum(log.get("char_count", 0) for log in logging_outputs)
        if word_count > 0:
            metrics.log_scalar("wer", float(word_error) / word_count * 100, word_count, round=4)
        if char_count > 0:
            metrics.log_scalar("cer", float(char_error) / char_count * 100, char_count, round=4)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.AsrDictionary`."""
        return self.tgt_dict

    @property
    def word_dictionary(self):
        """Return the target :class:`~fairseq.data.AsrDictionary`."""
        return self.word_dict

    def _inference_with_wer(self, decoder, sample, model):
        from espresso.tools import wer

        scorer = wer.Scorer(self.target_dictionary, wer_output_filter=self.args.wer_output_filter)
        tokens, lprobs, _ = decoder.decode([model], sample)
        pred = tokens[:, 1:].data.cpu()  # bsz x len
        target = sample["target"]
        assert pred.size(0) == target.size(0)
        # compute word error stats
        scorer.reset()
        for i in range(target.size(0)):
            utt_id = sample["utt_id"][i]
            ref_tokens = sample["target_raw_text"][i]
            pred_tokens = self.target_dictionary.string(pred.data[i])
            scorer.add_evaluation(utt_id, ref_tokens, pred_tokens)
        return (
            scorer.tot_word_error(), scorer.tot_word_count(),
            scorer.tot_char_error(), scorer.tot_char_count(),
        )
