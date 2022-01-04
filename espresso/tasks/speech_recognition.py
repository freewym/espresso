# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import json
import logging
import os
from argparse import Namespace
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
from omegaconf import II, DictConfig

from espresso.data import AsrDataset, AsrDictionary, AsrTextDataset, AudioFeatDataset
from fairseq import utils
from fairseq.data import BaseWrapperDataset, ConcatDataset
from fairseq.dataclass import FairseqDataclass
from fairseq.logging import metrics
from fairseq.tasks import FairseqTask, register_task

logger = logging.getLogger(__name__)


@dataclass
class SpeechRecognitionEspressoConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None, metadata={"help": "path to data directory"}
    )
    dict: Optional[str] = field(
        default=None, metadata={"help": "path to the dictionary"}
    )
    non_lang_syms: Optional[str] = field(
        default=None,
        metadata={
            "help": "path to a file listing non-linguistic symbols, e.g., <NOISE> "
            "etc. One entry per line. To be filtered out when calculating WER/CER"
        },
    )
    word_dict: Optional[str] = field(
        default=None,
        metadata={"help": "path to the word dictionary. Only relevant for decoding"},
    )
    wer_output_filter: Optional[str] = field(
        default=None,
        metadata={"help": "path to wer_output_filter file for WER evaluation"},
    )
    max_source_positions: Optional[int] = field(
        default=1024, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: Optional[int] = field(
        default=1024, metadata={"help": "max number of tokens in the target sequence"}
    )
    upsample_primary: int = field(
        default=1,
        metadata={"help": "amount to upsample primary dataset"},
    )
    num_batch_buckets: Optional[int] = field(
        default=0,
        metadata={
            "help": "if >0, then bucket source and target lengths into N "
            "buckets and pad accordingly; this is useful on TPUs "
            "to minimize the number of compilations"
        },
    )
    feat_in_channels: int = field(
        default=1, metadata={"help": "feature input channels"}
    )
    max_num_expansions_per_step: int = field(
        default=2,
        metadata={
            "help": "the maximum number of non-blank expansions in a single "
            "time step of decoding; only relavant when training with transducer loss"
        },
    )
    specaugment_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "SpecAugment config string. If not None and not empty, "
            "then apply SpecAugment. Should be an evaluatable expression of "
            "a python dict. See speech_tools.specaug_interpolate.specaug() for "
            "all allowed arguments. Argments not appearing in this string "
            "will take on their default values"
        },
    )
    global_cmvn_stats_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "If not None, apply global cmvn using this global cmvn stats file (.npz)."
        },
    )
    criterion_name: Optional[str] = field(
        default=II("criterion._name"),
        metadata={
            "help": "Some class instantiations rely on this value, e.g., dataset, dictionary, decoder, etc."
        },
    )
    # TODO common vars below add to parent
    seed: int = II("common.seed")
    data_buffer_size: int = II("dataset.data_buffer_size")
    tpu: bool = II("common.tpu")
    train_subset: str = II("dataset.train_subset")
    valid_subset: str = II("dataset.valid_subset")
    gen_subset: str = II("dataset.gen_subset")
    required_seq_len_multiple: int = II("dataset.required_seq_len_multiple")


def get_asr_dataset_from_json(
    data_path,
    split,
    tgt_dict,
    combine,
    upsample_primary=1,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    batch_based_on_both_src_tgt=False,
    seed=1,
    global_cmvn_stats_path=None,
    specaugment_config=None,
):
    """
    Parse data json and create dataset.
    See espresso/tools/asr_prep_json.py which pack json from raw files
    Json example:
    {
        "011c0202": {
            "feat": "fbank/raw_fbank_pitch_train_si284.1.ark:54819" or
            "wave": "/export/corpora5/LDC/LDC93S6B/11-1.1/wsj0/si_tr_s/011/011c0202.wv1" or
            "command": "sph2pipe -f wav /export/corpora5/LDC/LDC93S6B/11-1.1/wsj0/si_tr_s/011/011c0202.wv1 |",
            "text": "THE HOTEL",
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

        utt_ids, audios, texts, utt2num_frames = [], [], [], []
        for utt_id, val in loaded_json.items():
            utt_ids.append(utt_id)
            if "feat" in val:
                audio = val["feat"]
            elif "wave" in val:
                audio = val["wave"]
            elif "command" in val:
                audio = val["command"]
            else:
                raise KeyError(
                    f"'feat', 'wave' or 'command' should be present as a field for the entry {utt_id} in {data_json_path}"
                )
            audios.append(audio)
            if "text" in val:
                texts.append(val["text"])
            if "utt2num_frames" in val:
                utt2num_frames.append(int(val["utt2num_frames"]))

        assert len(utt2num_frames) == 0 or len(utt_ids) == len(utt2num_frames)
        if "feat" in next(iter(loaded_json.items())):
            extra_kwargs = {}
        else:
            extra_kwargs = {"feat_dim": 80, "feature_type": "fbank"}
            if global_cmvn_stats_path is not None:
                feature_transforms_config = {
                    "transforms": ["global_cmvn"],
                    "global_cmvn": {"stats_npz_path": global_cmvn_stats_path},
                }
                extra_kwargs["feature_transforms_config"] = feature_transforms_config
        src_datasets.append(
            AudioFeatDataset(
                utt_ids,
                audios,
                utt2num_frames=utt2num_frames,
                seed=seed,
                specaugment_config=specaugment_config if split == "train" else None,
                **extra_kwargs,
            )
        )
        if len(texts) > 0:
            assert len(utt_ids) == len(texts)
            assert tgt_dict is not None
            tgt_datasets.append(AsrTextDataset(utt_ids, texts, tgt_dict))

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
            assert (
                feat_dim == src_datasets[i].feat_dim
            ), "feature dimension does not match across multiple json files"
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return AsrDataset(
        src_dataset,
        src_dataset.sizes,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=False,
        left_pad_target=False,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
        batch_based_on_both_src_tgt=batch_based_on_both_src_tgt,
    )


@register_task("speech_recognition_espresso", dataclass=SpeechRecognitionEspressoConfig)
class SpeechRecognitionEspressoTask(FairseqTask):
    """
    Transcribe from speech (source) to text (target).

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

    @classmethod
    def load_dictionary(cls, filename, enable_bos=False, non_lang_syms=None):
        """Load the dictionary from the filename
        Args:
            filename (str): the filename
            enable_bos (bool, optional): optionally enable bos symbol
            non_lang_syms (str, optional): non_lang_syms filename
        """
        return AsrDictionary.load(
            filename, enable_bos=enable_bos, f_non_lang_syms=non_lang_syms
        )

    @classmethod
    def build_dictionary(
        cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8
    ):
        """Disable this method"""
        raise NotImplementedError

    def __init__(
        self, cfg: SpeechRecognitionEspressoConfig, tgt_dict, feat_dim, word_dict=None
    ):
        super().__init__(cfg)
        self.tgt_dict = tgt_dict
        self.word_dict = word_dict
        self.feat_dim = feat_dim
        self.feat_in_channels = cfg.feat_in_channels
        self.extra_symbols_to_ignore = {tgt_dict.pad()}  # for validation with WER
        if cfg.criterion_name in ["transducer_loss", "ctc"]:
            self.blank_symbol = tgt_dict.bos_word  # reserve the bos symbol for blank
            self.extra_symbols_to_ignore.add(tgt_dict.index(self.blank_symbol))
        torch.backends.cudnn.deterministic = True
        # Compansate for the removel of :func:`torch.rand()` from
        # :func:`fairseq.distributed_utils.distributed_init()` by fairseq,
        # to make previous experiments reproducible.
        torch.rand(1)

    @classmethod
    def setup_task(cls, cfg: SpeechRecognitionEspressoConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (SpeechRecognitionEspressoConfig): configuration of this task
        """
        # load dictionaries
        dict_path = os.path.join(cfg.data, "dict.txt") if cfg.dict is None else cfg.dict
        enable_bos = True if cfg.criterion_name in ["transducer_loss", "ctc"] else False
        tgt_dict = cls.load_dictionary(
            dict_path,
            enable_bos=enable_bos,
            non_lang_syms=cfg.non_lang_syms,
        )
        logger.info("dictionary: {} types".format(len(tgt_dict)))

        # minimum code for loading data in order to obtain feat_dim
        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        data_path = paths[0]
        split = cfg.valid_subset.split(",")[
            0
        ]  # valid set is usually much smaller than train set, so it's faster
        try:
            src_dataset = get_asr_dataset_from_json(
                data_path, split, tgt_dict, combine=False
            ).src
        except FileNotFoundError:
            logger.warning(
                f"'{split}' set not found. Try to obtain feat_dim from '{cfg.gen_subset}'"
            )
            src_dataset = get_asr_dataset_from_json(
                data_path, cfg.gen_subset, tgt_dict, combine=False
            ).src
        if isinstance(src_dataset, ConcatDataset):
            feat_dim = src_dataset.datasets[0].feat_dim
        elif isinstance(src_dataset, BaseWrapperDataset):
            feat_dim = src_dataset.dataset.feat_dim
        else:
            feat_dim = src_dataset.feat_dim

        if cfg.word_dict is not None:
            word_dict = cls.load_dictionary(cfg.word_dict, enable_bos=False)
            logger.info("word dictionary: {} types".format(len(word_dict)))
            return cls(cfg, tgt_dict, feat_dim, word_dict=word_dict)

        else:
            return cls(cfg, tgt_dict, feat_dim)

    def load_dataset(
        self,
        split: str,
        epoch: int = 1,
        combine: bool = False,
        task_cfg: DictConfig = None,
        **kwargs,
    ):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
            epoch (int): epoch number determining which shard of training data to load
            combine (bool): combines a split segmented into pieces into one dataset
            task_cfg (DictConfig): optional task configuration stored in the checkpoint that can be used
                                         to load datasets
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]
        task_cfg = task_cfg or self.cfg

        self.datasets[split] = get_asr_dataset_from_json(
            data_path,
            split,
            self.tgt_dict,
            combine=combine,
            upsample_primary=self.cfg.upsample_primary,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != self.cfg.gen_subset),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
            batch_based_on_both_src_tgt=(self.cfg.criterion_name == "transducer_loss"),
            seed=self.cfg.seed,
            global_cmvn_stats_path=self.cfg.global_cmvn_stats_path,
            specaugment_config=self.cfg.specaugment_config,
        )

        # update the counts of <eos> and <unk> in tgt_dict with training data
        if split == "train":
            tgt_dataset = self.datasets[split].tgt
            self.tgt_dict.count[self.tgt_dict.eos()] = len(tgt_dataset)
            unk_count = 0
            for i in range(len(tgt_dataset)):
                unk_count += (
                    (tgt_dataset[i][0] == self.tgt_dict.unk()).int().sum().item()
                )
            self.tgt_dict.count[self.tgt_dict.unk()] = unk_count

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return AsrDataset(
            src_tokens,
            src_lengths,
            dictionary=self.target_dictionary,
            constraints=constraints,
        )

    def build_model(self, model_cfg: DictConfig):
        model = super().build_model(model_cfg)
        # build a greedy decoder for validation with WER
        if self.cfg.criterion_name == "transducer_loss":  # a transducer model
            from espresso.tools.transducer_greedy_decoder import TransducerGreedyDecoder

            self.decoder_for_validation = TransducerGreedyDecoder(
                [model],
                self.target_dictionary,
                max_num_expansions_per_step=self.cfg.max_num_expansions_per_step,
            )
        elif self.cfg.criterion_name == "ctc":  # a ctc model
            raise NotImplementedError
        else:  # assume it is an attention-based encoder-decoder model
            from espresso.tools.simple_greedy_decoder import SimpleGreedyDecoder

            self.decoder_for_validation = SimpleGreedyDecoder(
                [model],
                self.target_dictionary,
                for_validation=True,
            )

        return model

    def build_criterion(self, cfg: DictConfig):
        # keep a reference to the criterion instance in task for convenience
        # (to be used inside self.begin_epoch())
        self.criterion = super().build_criterion(cfg)
        return self.criterion

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
        prefix_allowed_tokens_fn=None,
    ):
        if self.cfg.criterion_name == "transducer_loss":
            from espresso.tools.transducer_beam_search_decoder import (
                TransducerBeamSearchDecoder,
            )
            from espresso.tools.transducer_greedy_decoder import TransducerGreedyDecoder

            extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
            if getattr(args, "print_alignment", False):
                extra_gen_cls_kwargs["print_alignment"] = True

            if seq_gen_cls is None:
                seq_gen_cls = (
                    TransducerGreedyDecoder
                    if getattr(args, "beam", 1) == 1
                    else TransducerBeamSearchDecoder
                )

            return seq_gen_cls(
                models,
                self.target_dictionary,
                temperature=getattr(args, "temperature", 1.0),
                # the arguments below are not being used in :class:`~TransducerGreedyDecoder`
                beam_size=getattr(args, "beam", 1),
                normalize_scores=(not getattr(args, "unnormalized", False)),
                max_num_expansions_per_step=getattr(
                    args, "transducer_max_num_expansions_per_step", 2
                ),
                expansion_beta=getattr(args, "transducer_expansion_beta", 0),
                expansion_gamma=getattr(args, "transducer_expansion_gamma", None),
                prefix_alpha=getattr(args, "transducer_prefix_alpha", None),
                **extra_gen_cls_kwargs,
            )
        elif self.cfg.criterion_name == "ctc":
            raise NotImplementedError

        return super().build_generator(
            models,
            args,
            seq_gen_cls=seq_gen_cls,
            extra_gen_cls_kwargs=extra_gen_cls_kwargs,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        (
            logging_output["word_error"],
            logging_output["word_count"],
            logging_output["char_error"],
            logging_output["char_count"],
        ) = self._inference_with_wer(self.decoder_for_validation, sample, model)
        return loss, sample_size, logging_output

    def begin_epoch(self, epoch, model):
        """Hook function called before the start of each epoch."""
        super().begin_epoch(epoch, model)
        if hasattr(self.criterion, "set_epoch"):
            self.criterion.set_epoch(epoch)

    def reduce_metrics(self, logging_outputs, criterion):
        """Aggregate logging outputs from data parallel training."""
        super().reduce_metrics(logging_outputs, criterion)
        word_error = sum(log.get("word_error", 0) for log in logging_outputs)
        word_count = sum(log.get("word_count", 0) for log in logging_outputs)
        char_error = sum(log.get("char_error", 0) for log in logging_outputs)
        char_count = sum(log.get("char_count", 0) for log in logging_outputs)
        if word_count > 0:
            metrics.log_scalar(
                "wer", float(word_error) / word_count * 100, word_count, round=4
            )
        if char_count > 0:
            metrics.log_scalar(
                "cer", float(char_error) / char_count * 100, char_count, round=4
            )

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.AsrDictionary`."""
        return self.tgt_dict

    def build_tokenizer(self, cfg: Union[DictConfig, Namespace]):
        """Build the pre-tokenizer for this task."""
        if hasattr(self.tgt_dict, "build_tokenizer"):
            # the instance is built within self.tgt_dict
            self.tgt_dict.build_tokenizer(cfg)
            return self.tgt_dict.tokenizer
        else:
            return super().build_tokenizer(cfg)

    def build_bpe(self, cfg: Union[DictConfig, Namespace]):
        """Build the tokenizer for this task."""
        if hasattr(self.tgt_dict, "build_bpe"):
            # the instance is built within self.tgt_dict
            self.tgt_dict.build_bpe(cfg)
            return self.tgt_dict.bpe
        else:
            return super().build_bpe(cfg)

    @property
    def word_dictionary(self):
        """Return the target :class:`~fairseq.data.AsrDictionary`."""
        return self.word_dict

    def _inference_with_wer(self, decoder, sample, model):
        from espresso.tools import wer

        scorer = wer.Scorer(
            self.target_dictionary, wer_output_filter=self.cfg.wer_output_filter
        )
        tokens, lprobs, _ = decoder.decode([model], sample)
        pred = tokens.data.cpu()  # bsz x len
        target = sample["target"]
        assert pred.size(0) == target.size(0)
        # compute word error stats
        scorer.reset()
        for i in range(target.size(0)):
            utt_id = sample["utt_id"][i]
            ref_tokens = self.target_dictionary.wordpiece_encode(sample["text"][i])
            pred_tokens = self.target_dictionary.string(
                pred.data[i], extra_symbols_to_ignore=self.extra_symbols_to_ignore
            )
            scorer.add_evaluation(utt_id, ref_tokens, pred_tokens)
        return (
            scorer.tot_word_error(),
            scorer.tot_word_count(),
            scorer.tot_char_error(),
            scorer.tot_char_count(),
        )
