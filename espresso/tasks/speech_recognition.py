# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import itertools
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch

from fairseq import utils
from fairseq.data import BaseWrapperDataset, ConcatDataset
from fairseq.dataclass import FairseqDataclass
from fairseq.logging import metrics
from fairseq.tasks import FairseqTask, register_task
from omegaconf import II

from espresso.data import (
    AsrDataset,
    AsrDictionary,
    AsrTextDataset,
    FeatScpCachedDataset,
)


logger = logging.getLogger(__name__)


@dataclass
class SpeechRecognitionEspressoConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None, metadata={"help": "path to data directory"}
    )
    dict: Optional[str] = field(default=None, metadata={"help": "path to the dictionary"})
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
        default=1, metadata={"help": "amount to upsample primary dataset"},
    )
    num_batch_buckets: Optional[int] = field(
        default=0,
        metadata={
            "help": "if >0, then bucket source and target lengths into N "
            "buckets and pad accordingly; this is useful on TPUs "
            "to minimize the number of compilations"
        },
    )
    feat_in_channels: int = field(default=1, metadata={"help": "feature input channels"})
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
    # TODO common vars below add to parent
    seed: int = II("common.seed")
    data_buffer_size: int = II("dataset.data_buffer_size")
    tpu: bool = II("common.tpu")
    train_subset: str = II("dataset.train_subset")
    gen_subset: str = II("dataset.gen_subset")
    required_seq_len_multiple: int = II("dataset.required_seq_len_multiple")


def get_asr_dataset_from_json(
    data_path,
    split,
    tgt_dict,
    combine,
    upsample_primary,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    seed=1,
    specaugment_config=None,
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
                raise FileNotFoundError(
                    "Dataset not found: {}".format(data_json_path)
                )

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
    )


@register_task("speech_recognition_espresso", dataclass=SpeechRecognitionEspressoConfig)
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

    def __init__(self, cfg: SpeechRecognitionEspressoConfig, tgt_dict, word_dict=None):
        super().__init__(cfg)
        self.tgt_dict = tgt_dict
        self.word_dict = word_dict
        self.feat_in_channels = cfg.feat_in_channels
        self.specaugment_config = cfg.specaugment_config
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
        tgt_dict = cls.load_dictionary(dict_path, non_lang_syms=cfg.non_lang_syms)
        logger.info("dictionary: {} types".format(len(tgt_dict)))
        if cfg.word_dict is not None:
            word_dict = cls.load_dictionary(cfg.word_dict)
            logger.info("word dictionary: {} types".format(len(word_dict)))
            return cls(cfg, tgt_dict, word_dict)

        else:
            return cls(cfg, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        self.datasets[split] = get_asr_dataset_from_json(
            data_path,
            split,
            self.tgt_dict,
            combine=combine,
            upsample_primary=self.cfg.upsample_primary,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != self.cfg.gen_subset),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
            seed=self.cfg.seed,
            specaugment_config=self.specaugment_config,
        )

        src_dataset = self.datasets[split].src
        if isinstance(src_dataset, ConcatDataset):
            self.feat_dim = src_dataset.datasets[0].feat_dim
        elif isinstance(src_dataset, BaseWrapperDataset):
            self.feat_dim = src_dataset.dataset.feat_dim
        else:
            self.feat_dim = src_dataset.feat_dim

        # update the counts of <eos> and <unk> in tgt_dict with training data
        if split == "train":
            tgt_dataset = self.datasets[split].tgt
            self.tgt_dict.count[self.tgt_dict.eos()] = len(tgt_dataset)
            unk_count = 0
            for i in range(len(tgt_dataset)):
                unk_count += (tgt_dataset[i][0] == self.tgt_dict.unk()).int().sum().item()
            self.tgt_dict.count[self.tgt_dict.unk()] = unk_count

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return AsrDataset(
            src_tokens,
            src_lengths,
            dictionary=self.target_dictionary,
            constraints=constraints,
        )

    def build_model(self, model_cfg: FairseqDataclass):
        model = super().build_model(model_cfg)
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
        return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.AsrDictionary`."""
        return self.tgt_dict

    def build_tokenizer(self, cfg: FairseqDataclass):
        """Build the pre-tokenizer for this task."""
        self.tgt_dict.build_tokenizer(cfg)
        # the instance is built within self.tgt_dict
        return self.tgt_dict.tokenizer

    def build_bpe(self, cfg: FairseqDataclass):
        """Build the tokenizer for this task."""
        self.tgt_dict.build_bpe(cfg)
        # the instance is built within self.tgt_dict
        return self.tgt_dict.bpe

    @property
    def word_dictionary(self):
        """Return the target :class:`~fairseq.data.AsrDictionary`."""
        return self.word_dict

    def _inference_with_wer(self, decoder, sample, model):
        from espresso.tools import wer

        scorer = wer.Scorer(self.target_dictionary, wer_output_filter=self.cfg.wer_output_filter)
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
