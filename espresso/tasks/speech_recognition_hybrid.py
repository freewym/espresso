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
from fairseq.dataclass.configs import GenerationConfig
from fairseq.tasks import FairseqTask, register_task
from omegaconf import II

from espresso.data import (
    AliScpCachedDataset,
    AsrChainDataset,
    AsrXentDataset,
    AsrDictionary,
    AsrTextDataset,
    FeatScpCachedDataset,
    NumeratorGraphDataset,
)

try:
    import kaldi_io
except ImportError:
    raise ImportError("Please install kaldi_io with: pip install kaldi_io")


logger = logging.getLogger(__name__)


@dataclass
class SpeechRecognitionHybridConfig(FairseqDataclass):
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
    num_targets: int = field(
        default=3000,
        metadata={"help": "number of targets for training (e.g., num pdf-ids)"},
    )
    initial_state_prior_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "path to the file containing initial state prior. Only relevant "
            "with cross-entropy training"
        },
    )
    state_prior_update_interval: Optional[int] = field(
        default=None,
        metadata={
            "help": "state prior estimate will be updated every this number of updates "
            "during training. If None, then use the initial value estimated from the "
            "alignments. Only relevant with cross-entropy training"
        },
    )
    state_prior_update_smoothing: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "smoothing factor while updating state prior estimate. Only "
            "relevant with cross-entropy training"
        },
    )
    chunk_width: Optional[int] = field(
        default=None,
        metadata={
            "help": "chunk width for train/test data. Only relevant with chunk-wise "
            "training (including both cross-entropy and Lattice-free MMI). "
            "Do utterance-wise training/test if not specified"
        },
    )
    chunk_left_context: Optional[int] = field(
        default=0,
        metadata={"help": "number of frames appended to the left of a chunk"},
    )
    chunk_right_context: Optional[int] = field(
        default=0,
        metadata={"help": "number of frames appended to the right of a chunk"},
    )
    label_delay: Optional[int] = field(
        default=0,
        metadata={
            "help": "offet of alignments as prediction labels. Maybe useful "
            "in archs such as asymmetric convolution, unidirectional LSTM, etc. "
            "It can be negative. Only relevant with chunk-wise cross-entropy training"
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
    criterion_name: str = II("criterion._name")
    max_epoch: int = II("optimization.max_epoch")  # to determine whether in trainig stage


def get_asr_dataset_from_json(
    data_path,
    split,
    dictionary,
    combine,
    upsample_primary=1,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    lf_mmi=True,
    seed=1,
    specaugment_config=None,
    chunk_width=None,
    chunk_left_context=None,
    chunk_right_context=None,
    label_delay=0,
):
    """
    Parse data json and create dataset.
    See espresso/tools/asr_prep_json.py which pack json from raw files
    Json example:
    {
        "011c0202": {
            "feat": "data/train_si284_spe2e_hires/data/raw_mfcc_train_si284_spe2e_hires.1.ark:24847",
            "numerator_fst": "exp/chain/e2e_bichar_tree_tied1a/fst.1.ark:6704",
            "alignment": "exp/tri3/ali.ark:8769",
            "text": "THE HOTELi OPERATOR'S EMBASSY",
            "utt2num_frames": "693",
        },
        "011c0203": {
            ...
        }
    }
    """
    src_datasets = []
    tgt_datasets = []
    text_datasets = []

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

        utt_ids, feats, numerator_fsts, alignments, text, utt2num_frames = [], [], [], [], [], []
        for utt_id, val in loaded_json.items():
            utt_ids.append(utt_id)
            feats.append(val["feat"])
            if "numerator_fst" in val:
                numerator_fsts.append(val["numerator_fst"])
            if "alignment" in val:
                alignments.append(val["alignment"])
            if "text" in val:
                text.append(val["text"])
            if "utt2num_frames" in val:
                utt2num_frames.append(int(val["utt2num_frames"]))

        assert len(utt2num_frames) == 0 or len(utt_ids) == len(utt2num_frames)
        src_datasets.append(FeatScpCachedDataset(
            utt_ids, feats, utt2num_frames=utt2num_frames, seed=seed,
            specaugment_config=specaugment_config if split == "train" else None,
            ordered_prefetch=True,
        ))
        if lf_mmi:
            if len(numerator_fsts) > 0:
                assert len(utt_ids) == len(numerator_fsts)
                tgt_datasets.append(NumeratorGraphDataset(utt_ids, numerator_fsts))
        else:  # cross-entropy
            if len(alignments) > 0:
                assert len(utt_ids) == len(alignments)
                tgt_datasets.append(
                    AliScpCachedDataset(
                        utt_ids, alignments, utt2num_frames=utt2num_frames,
                        ordered_prefetch=True,
                    )
                )

        if len(text) > 0:
            assert len(utt_ids) == len(text)
            text_datasets.append(AsrTextDataset(utt_ids, text, dictionary, append_eos=False))

        logger.info("{} {} examples".format(data_json_path, len(src_datasets[-1])))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0
    assert len(src_datasets) == len(text_datasets) or len(text_datasets) == 0

    feat_dim = src_datasets[0].feat_dim

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
        text_dataset = text_datasets[0] if len(text_datasets) > 0 else None
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
        if len(text_datasets) > 0:
            text_dataset = ConcatDataset(text_datasets, sample_ratios)
        else:
            text_dataset = None

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    if lf_mmi:
        return AsrChainDataset(
            src_dataset,
            src_dataset.sizes,
            tgt_dataset,
            tgt_dataset_sizes,
            text=text_dataset,
            num_buckets=num_buckets,
            shuffle=shuffle,
            pad_to_multiple=pad_to_multiple,
        )
    else:
        return AsrXentDataset(
            src_dataset,
            src_dataset.sizes,
            tgt_dataset,
            tgt_dataset_sizes,
            text=text_dataset,
            num_buckets=num_buckets,
            shuffle=shuffle,
            pad_to_multiple=pad_to_multiple,
            seed=seed,
            chunk_width=chunk_width,
            chunk_left_context=chunk_left_context,
            chunk_right_context=chunk_right_context,
            label_delay=label_delay,
            random_chunking=(split == "train" and chunk_width is not None),
        )


@register_task("speech_recognition_hybrid", dataclass=SpeechRecognitionHybridConfig)
class SpeechRecognitionHybridTask(FairseqTask):
    """
    Hybrid speech recognition with lattice-free MMI or cross-entropy loss.
    Currently it dumps posteriors from neural networks' output on-the-fly or
    as an ark file for Kaldi to decode.

    Args:
        dictionary (~fairseq.data.AsrDictionary): dictionary for the final text

    .. note::

        The speech recognition with lattice-free MMI task is compatible with
        :mod:`speech-train`, and :mod:`dump-posteriors`. The results are not
        strictly reproducible (i.e., there is some randomness among different
        runs with the same exprimental setting) due to the use of `atomicAdd`
        while accumulating gradients w.r.t. pdf-ids in backprop of LF-MMI loss.
        See https://pytorch.org/docs/stable/notes/randomness.html for details.

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
    def build_dictionary(cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        """Disable this method
        """
        raise NotImplementedError

    def __init__(self, cfg: SpeechRecognitionHybridConfig, dictionary, feat_dim):
        super().__init__(cfg)
        self.dictionary = dictionary
        self.feat_dim = feat_dim
        self.feat_in_channels = cfg.feat_in_channels
        self.specaugment_config = cfg.specaugment_config
        self.num_targets = cfg.num_targets
        self.training_stage = (cfg.max_epoch > 0)  # a hack

        # the following attributes are related to state_prior estimate
        self.initial_state_prior = None
        if cfg.initial_state_prior_file is not None:  # only relevant for Xent training, used in models
            self.initial_state_prior = kaldi_io.read_vec_flt(cfg.initial_state_prior_file)
            self.initial_state_prior = torch.from_numpy(self.initial_state_prior)
            assert (
                self.initial_state_prior.size(0) == self.num_targets
            ), "length of initial_state_prior ({}) != num_targets ({})".format(
                self.initial_state_prior.size(0), self.num_targets
            )
        self.state_prior_update_interval = cfg.state_prior_update_interval
        if self.state_prior_update_interval is None and self.initial_state_prior is not None:
            logger.info("state prior will not be updated during training")
        self.state_prior_update_smoothing = cfg.state_prior_update_smoothing
        self.averaged_state_post = None  # state poterior will be saved here before commited as new state prior

        # the following 4 options are for chunk-wise training/test (including Xent and LF-MMI)
        self.chunk_width = cfg.chunk_width
        self.chunk_left_context = cfg.chunk_left_context
        self.chunk_right_context = cfg.chunk_right_context
        self.label_delay = cfg.label_delay  # only for chunk-wise Xent training

        torch.backends.cudnn.deterministic = True

    @classmethod
    def setup_task(cls, cfg: SpeechRecognitionHybridConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (SpeechRecognitionHybridConfig): configuration of this task
        """
        # load dictionaries
        dict_path = cfg.dict
        dictionary = (
            cls.load_dictionary(dict_path, non_lang_syms=cfg.non_lang_syms)
            if dict_path is not None
            else None
        )
        if dictionary is not None:
            logger.info("dictionary: {} types".format(len(dictionary)))

        # minimum code for loading data in order to obtain feat_dim
        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        data_path = paths[0]
        split = cfg.valid_subset.split(",")[0]  # valid set is usually much smaller than train set, so it's faster
        try:
            src_dataset = get_asr_dataset_from_json(data_path, split, dictionary, combine=False).src
        except FileNotFoundError:
            logger.warning(f"'{split}' set not found. Try to obtain feat_dim from '{cfg.gen_subset}'")
            src_dataset = get_asr_dataset_from_json(data_path, cfg.gen_subset, dictionary, combined=False).src
        if isinstance(src_dataset, ConcatDataset):
            feat_dim = src_dataset.datasets[0].feat_dim
        elif isinstance(src_dataset, BaseWrapperDataset):
            feat_dim = src_dataset.dataset.feat_dim
        else:
            feat_dim = src_dataset.feat_dim

        return cls(cfg, dictionary, feat_dim)

    def load_dataset(
        self,
        split: str,
        epoch: int = 1,
        combine: bool = False,
        task_cfg: FairseqDataclass = None,
        **kwargs,
    ):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
            epoch (int): epoch number determining which shard of training data to load
            combine (bool): combines a split segmented into pieces into one dataset
            task_cfg (FairseqDataclass): optional task configuration stored in the checkpoint that can be used
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
            self.dictionary,
            combine=combine,
            upsample_primary=self.cfg.upsample_primary,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != self.cfg.gen_subset),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
            lf_mmi=(self.cfg.criterion_name == "lattice_free_mmi"),
            seed=self.cfg.seed,
            specaugment_config=self.specaugment_config,
            chunk_width=None if self.training_stage
            and split in self.cfg.valid_subset.split(",")
            else self.chunk_width,
            chunk_left_context=self.chunk_left_context,
            chunk_right_context=self.chunk_right_context,
            label_delay=self.label_delay,
        )

    def build_generator(self, models, cfg: GenerationConfig):
        if cfg.score_reference:
            cfg.score_reference = False
            logger.warning(
                "--score-reference is not applicable to speech recognition, ignoring it."
            )
        from espresso.tools.generate_log_probs_for_decoding import GenerateLogProbsForDecoding

        apply_log_softmax = getattr(cfg, "apply_log_softmax", False)
        return GenerateLogProbsForDecoding(models, apply_log_softmax=apply_log_softmax)

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return AsrChainDataset(src_tokens, src_lengths)

    def inference_step(self, generator, models, sample):
        with torch.no_grad():
            return generator.generate(models, sample)

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        state_post = []
        for log in logging_outputs:
            post = log.get("state_post", None)
            if post is not None:
                state_post.append(post)
        if len(state_post) > 0:
            # collect state priors from all workers and do weighted average
            weights = state_post[0].new([log.get("ntokens", 0) for log in logging_outputs])
            weights = weights / weights.sum()  # N
            with torch.no_grad():
                stacked_state_post = torch.stack(state_post, dim=1)  # V x N
                self.averaged_state_post = stacked_state_post.mv(weights)  # V
        else:
            self.averaged_state_post = None

    def update_state_prior(self, model):
        if self.averaged_state_post is not None:
            assert hasattr(model, "update_state_prior")
            model.update_state_prior(self.averaged_state_post, self.state_prior_update_smoothing)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.AsrDictionary`."""
        # Note: padding idx for criterions would be self.target_dictionary.pad() if it
        # returns not None.
        return None
