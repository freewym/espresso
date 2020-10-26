# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import logging
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.lstm import Linear
from fairseq.modules import FairseqDropout
from omegaconf import DictConfig

import espresso.tools.utils as speech_utils


logger = logging.getLogger(__name__)


@register_model("speech_tdnn")
class SpeechTdnnEncoderModel(FairseqEncoderModel):
    def __init__(self, encoder, state_prior: Optional[torch.FloatTensor] = None):
        super().__init__(encoder)
        self.num_updates = 0
        self.state_prior = state_prior

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--dropout", type=float, metavar="D",
                            help="dropout probability")
        parser.add_argument("--hidden-sizes", type=str, metavar="EXPR",
                            help="list of hidden sizes for all Tdnn layers")
        parser.add_argument("--kernel-sizes", type=str, metavar="EXPR",
                            help="list of all Tdnn layer\'s kernel sizes")
        parser.add_argument("--strides", type=str, metavar="EXPR",
                            help="list of all Tdnn layer\'s strides")
        parser.add_argument("--dilations", type=str, metavar="EXPR",
                            help="list of all Tdnn layer\'s dilations")
        parser.add_argument("--num-layers", type=int, metavar="N",
                            help="number of Tdnn layers")
        parser.add_argument("--residual", type=lambda x: utils.eval_bool(x),
                            help="create residual connections for rnn encoder "
                            "layers (starting from the 2nd layer), i.e., the actual "
                            "output of such layer is the sum of its input and output")

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument("--dropout-in", type=float, metavar="D",
                            help="dropout probability for encoder\'s input")
        parser.add_argument("--dropout-out", type=float, metavar="D",
                            help="dropout probability for Tdnn layers\' output")
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        hidden_sizes = speech_utils.eval_str_nested_list_or_tuple(args.hidden_sizes, type=int)
        kernel_sizes = speech_utils.eval_str_nested_list_or_tuple(args.kernel_sizes, type=int)
        strides = speech_utils.eval_str_nested_list_or_tuple(args.strides, type=int)
        dilations = speech_utils.eval_str_nested_list_or_tuple(args.dilations, type=int)
        logger.info("input feature dimension: {}, output dimension: {}".format(task.feat_dim, task.num_targets))

        encoder = SpeechTdnnEncoder(
            input_size=task.feat_dim,
            output_size=task.num_targets,
            hidden_sizes=hidden_sizes,
            kernel_sizes=kernel_sizes,
            strides=strides,
            dilations=dilations,
            num_layers=args.num_layers,
            dropout_in=args.dropout_in,
            dropout_out=args.dropout_out,
            residual=args.residual,
            chunk_width=getattr(task, "chunk_width", None),
            chunk_left_context=getattr(task, "chunk_left_context", 0),
            training_stage=getattr(task, "training_stage", True),
        )
        return cls(encoder, state_prior=getattr(task, "initial_state_prior", None))

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates
        super().set_num_updates(num_updates)

    def output_lengths(self, in_lengths):
        return self.encoder.output_lengths(in_lengths)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        encoder_out = net_output.encoder_out
        if torch.is_tensor(encoder_out):
            logits = encoder_out.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    def get_logits(self, net_output):
        logits = net_output.encoder_out.transpose(0, 1).squeeze(2)  # T x B x 1 -> B x T
        return logits

    def update_state_prior(self, new_state_prior, factor=0.1):
        assert self.state_prior is not None
        self.state_prior = self.state_prior.to(new_state_prior)
        self.state_prior = (1. - factor) * self.state_prior + factor * new_state_prior
        self.state_prior = self.state_prior / self.state_prior.sum()  # re-normalize

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["state_prior"] = self.state_prior
        return state_dict

    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg: Optional[DictConfig] = None,
        args: Optional[Namespace] = None,
    ):
        state_dict_subset = state_dict.copy()
        self.state_prior = state_dict.get("state_prior", None)
        if "state_prior" in state_dict:
            self.state_prior = state_dict["state_prior"]
            del state_dict_subset["state_prior"]
        super().load_state_dict(
            state_dict_subset, strict=strict, model_cfg=model_cfg, args=args
        )


class TdnnBNReLU(nn.Module):
    """A block of Tdnn-BatchNorm-ReLU layers."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = dilation * (kernel_size - 1) // 2
        self.tdnn = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=self.padding, dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def output_lengths(self, in_lengths):
        out_lengths = (
            in_lengths + 2 * self.padding - self.dilation * (self.kernel_size - 1) +
            self.stride - 1
        ) // self.stride
        return out_lengths

    def forward(self, src, src_lengths):
        x = src.transpose(1, 2).contiguous()  # B x T x C -> B x C x T
        x = F.relu(self.bn(self.tdnn(x)))
        x = x.transpose(2, 1).contiguous()  # B x C x T -> B x T x C
        x_lengths = self.output_lengths(src_lengths)
        padding_mask = ~speech_utils.sequence_mask(x_lengths, x.size(1))
        if padding_mask.any():
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return x, x_lengths, padding_mask


class SpeechTdnnEncoder(FairseqEncoder):
    """Tdnn encoder."""
    def __init__(
        self, input_size, output_size, hidden_sizes=256, kernel_sizes=3, strides=1,
        dilations=3, num_layers=1, dropout_in=0.0, dropout_out=0.0, residual=False,
        chunk_width=None, chunk_left_context=0, training_stage=True,
    ):
        super().__init__(None)  # no src dictionary
        self.num_layers = num_layers
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes] * num_layers
        else:
            assert len(hidden_sizes) == num_layers
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * num_layers
        else:
            assert len(kernel_sizes) == num_layers
        if isinstance(strides, int):
            strides = [strides] * num_layers
        else:
            assert len(strides) == num_layers
        if isinstance(dilations, int):
            dilations = [dilations] * num_layers
        else:
            assert len(dilations) == num_layers
        self.dropout_in_module = FairseqDropout(
            dropout_in, module_name=self.__class__.__name__
        )
        self.dropout_out_module = FairseqDropout(
            dropout_out, module_name=self.__class__.__name__
        )
        self.residual = residual

        self.tdnn = nn.ModuleList([
            TdnnBNReLU(
                in_channels=input_size if layer == 0 else hidden_sizes[layer - 1],
                out_channels=hidden_sizes[layer], kernel_size=kernel_sizes[layer],
                stride=strides[layer], dilation=dilations[layer],
            )
            for layer in range(num_layers)
        ])

        receptive_field_radius = sum(layer.padding for layer in self.tdnn)
        assert (
            chunk_width is None
            or (chunk_width > 0 and chunk_left_context >= receptive_field_radius)
        )
        if (
            chunk_width is not None and chunk_width > 0
            and chunk_left_context > receptive_field_radius
        ):
            logger.warning(
                "chunk_{{left,right}}_context can be reduced to {}".format(receptive_field_radius)
            )
        self.out_chunk_begin = self.output_lengths(chunk_left_context + 1) - 1
        self.out_chunk_end = (
            self.output_lengths(chunk_left_context + chunk_width) if chunk_width is not None
            else None
        )
        self.training_stage = training_stage

        self.fc_out = Linear(hidden_sizes[-1], output_size, dropout=self.dropout_out_module.p)

    def output_lengths(self, in_lengths):
        out_lengths = in_lengths
        for layer in self.tdnn:
            out_lengths = layer.output_lengths(out_lengths)
        return out_lengths

    def forward(self, src_tokens, src_lengths: Tensor, **unused):
        x, x_lengths, encoder_padding_mask = self.extract_features(src_tokens, src_lengths)
        if (
            self.out_chunk_end is not None
            and (self.training or not self.training_stage)
        ):
            # determine which output frame to select for loss evaluation/test, assuming
            # all examples in a batch are of the same length for chunk-wise training/test
            x = x[self.out_chunk_begin: self.out_chunk_end]  # T x B x C -> W x B x C
            x_lengths = x_lengths.fill_(x.size(0))
            assert not encoder_padding_mask.any()
        x = self.output_layer(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask if encoder_padding_mask.any() else None,  # T x B
            encoder_embedding=None,
            encoder_states=None,
            src_tokens=None,
            src_lengths=x_lengths,  # B
        )

    def extract_features(self, src_tokens, src_lengths, **unused):
        x, x_lengths = src_tokens, src_lengths
        x = self.dropout_in_module(x)

        for i in range(len(self.tdnn)):
            if self.residual and i > 0:  # residual connection starts from the 2nd layer
                prev_x = x
            # apply Tdnn
            x, x_lengths, padding_mask = self.tdnn[i](x, x_lengths)
            x = self.dropout_out_module(x)
            x = x + prev_x if self.residual and i > 0 and x.size(1) == prev_x.size(1) else x

        x = x.transpose(0, 1)  # B x T x C -> T x B x C
        encoder_padding_mask = padding_mask.t()

        return x, x_lengths, encoder_padding_mask

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        return self.fc_out(features)  # T x B x C -> T x B x V

    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask
        src_lengths: Optional[Tensor] = encoder_out.src_lengths
        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(1, new_order)
        )
        new_src_lengths = (
            src_lengths
            if src_lengths is None
            else src_lengths.index_select(0, new_order)
        )
        return EncoderOut(
            encoder_out=encoder_out.encoder_out.index_select(1, new_order),
            encoder_padding_mask=new_encoder_padding_mask,
            encoder_embedding=None,
            encoder_states=None,
            src_tokens=None,
            src_lengths=new_src_lengths,
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


@register_model_architecture("speech_tdnn", "speech_tdnn")
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.0)
    args.hidden_sizes = getattr(args, "hidden_sizes", "640")
    args.kernel_sizes = getattr(args, "kernel_sizes", "[5, 3, 3, 3, 3]")
    args.strides = getattr(args, "strides", "1")
    args.dilations = getattr(args, "dilations", "[1, 1, 1, 3, 3]")
    args.num_layers = getattr(args, "num_layers", 5)
    args.residual = getattr(args, "residual", False)
    args.dropout_in = getattr(args, "dropout_in", args.dropout)
    args.dropout_out = getattr(args, "dropout_out", args.dropout)


@register_model_architecture("speech_tdnn", "speech_tdnn_wsj")
def tdnn_wsj(args):
    base_architecture(args)
