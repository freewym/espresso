# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from argparse import Namespace
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor

import espresso.tools.utils as speech_utils
from espresso.models.transformer import (
    DEFAULT_MAX_SOURCE_POSITIONS,
    SpeechTransformerConfig,
    SpeechTransformerEncoderBase,
)
from espresso.modules.speech_convolutions import ConvBNReLU
from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.distributed import fsdp_wrap
from fairseq.models import (
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import DEFAULT_MIN_PARAMS_TO_WRAP, Linear

logger = logging.getLogger(__name__)


@register_model("speech_transformer_encoder_model", dataclass=SpeechTransformerConfig)
class SpeechTransformerEncoderModel(FairseqEncoderModel):
    def __init__(self, cfg, encoder):
        super().__init__(encoder)
        self.cfg = cfg
        self.num_updates = 0

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, SpeechTransformerConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        if cfg.encoder.layers_to_keep:
            cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(","))

        if cfg.max_source_positions is None:
            cfg.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS

        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing

        out_channels = speech_utils.eval_str_nested_list_or_tuple(
            cfg.encoder.conv_channels, type=int
        )
        kernel_sizes = speech_utils.eval_str_nested_list_or_tuple(
            cfg.encoder.conv_kernel_sizes, type=int
        )
        strides = speech_utils.eval_str_nested_list_or_tuple(
            cfg.encoder.conv_strides, type=int
        )
        logger.info(
            "input feature dimension: {}, channels: {}".format(
                task.feat_dim, task.feat_in_channels
            )
        )
        assert task.feat_dim % task.feat_in_channels == 0
        conv_layers = (
            ConvBNReLU(
                out_channels,
                kernel_sizes,
                strides,
                in_channels=task.feat_in_channels,
            )
            if out_channels is not None
            else None
        )

        transformer_encoder_input_size = task.feat_dim // task.feat_in_channels
        if conv_layers is not None:
            for stride in strides:
                if isinstance(stride, (list, tuple)):
                    assert len(stride) > 0
                    s = stride[1] if len(stride) > 1 else stride[0]
                else:
                    assert isinstance(stride, int)
                    s = stride
                transformer_encoder_input_size = (
                    transformer_encoder_input_size + s - 1
                ) // s
            transformer_encoder_input_size *= out_channels[-1]
        else:
            transformer_encoder_input_size = task.feat_dim

        encoder = cls.build_encoder(
            cfg,
            pre_encoder=conv_layers,
            input_size=transformer_encoder_input_size,
            vocab_size=(
                len(task.target_dictionary)
                if task.target_dictionary is not None
                else None
            ),
        )
        # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
        encoder = fsdp_wrap(encoder, min_num_params=1e8)
        return cls(cfg, encoder)

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates
        super().set_num_updates(num_updates)

    @classmethod
    def build_encoder(
        cls,
        cfg,
        pre_encoder=None,
        input_size=83,
        vocab_size=None,
    ):
        return SpeechTransformerEncoderForPrediction(
            cfg,
            pre_encoder=pre_encoder,
            input_size=input_size,
            vocab_size=vocab_size,
        )

    def output_lengths(self, in_lengths):
        return self.encoder.output_lengths(in_lengths)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        encoder_out = net_output["encoder_out"][0]
        if torch.is_tensor(encoder_out):
            logits = encoder_out.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError


class SpeechTransformerEncoderForPrediction(SpeechTransformerEncoderBase):
    """Transformer encoder for speech with an optional output layer for token prediction."""

    def __init__(
        self,
        cfg,
        return_fc=False,
        pre_encoder=None,
        input_size=83,
        vocab_size=None,
    ):
        super().__init__(
            cfg,
            return_fc=return_fc,
            pre_encoder=pre_encoder,
            input_size=input_size,
        )

        self.fc_out = (
            Linear(cfg.encoder.embed_dim, vocab_size)
            if vocab_size is not None
            else None
        )

    def forward(
        self,
        src_tokens,
        src_lengths,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        out = super().forward(
            src_tokens, src_lengths, return_all_hiddens=return_all_hiddens
        )

        if self.fc_out is not None:
            out["encoder_out"][0] = self.fc_out(out["encoder_out"][0])  # T x B x V

        return out
