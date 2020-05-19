# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from fairseq import options
from fairseq.models import (
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.lstm import Linear

from espresso.models.speech_lstm import ConvBNReLU, SpeechLSTMEncoder
import espresso.tools.utils as speech_utils


DEFAULT_MAX_SOURCE_POSITIONS = 1e5


logger = logging.getLogger(__name__)


@register_model("speech_lstm_encoder_model")
class SpeechLSTMEncoderModel(FairseqEncoderModel):
    def __init__(self, encoder, state_prior: Optional[torch.FloatTensor] = None):
        super().__init__(encoder)
        self.state_prior = state_prior

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--dropout", type=float, metavar="D",
                            help="dropout probability")
        parser.add_argument("--encoder-conv-channels", type=str, metavar="EXPR",
                            help="list of encoder convolution's out channels")
        parser.add_argument("--encoder-conv-kernel-sizes", type=str, metavar="EXPR",
                            help="list of encoder convolution's kernel sizes")
        parser.add_argument("--encoder-conv-strides", type=str, metavar="EXPR",
                            help="list of encoder convolution's strides")
        parser.add_argument("--encoder-rnn-hidden-size", type=int, metavar="N",
                            help="encoder rnn's hidden size")
        parser.add_argument("--encoder-rnn-layers", type=int, metavar="N",
                            help="number of rnn encoder layers")
        parser.add_argument("--encoder-rnn-bidirectional",
                            type=lambda x: options.eval_bool(x),
                            help="make all rnn layers of encoder bidirectional")
        parser.add_argument("--encoder-rnn-residual",
                            type=lambda x: options.eval_bool(x),
                            help="create residual connections for rnn encoder "
                            "layers (starting from the 2nd layer), i.e., the actual "
                            "output of such layer is the sum of its input and output")

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument("--encoder-rnn-dropout-in", type=float, metavar="D",
                            help="dropout probability for encoder rnn's input")
        parser.add_argument("--encoder-rnn-dropout-out", type=float, metavar="D",
                            help="dropout probability for encoder rnn's output")
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)
        max_source_positions = getattr(args, "max_source_positions", DEFAULT_MAX_SOURCE_POSITIONS)

        out_channels = speech_utils.eval_str_nested_list_or_tuple(args.encoder_conv_channels, type=int)
        kernel_sizes = speech_utils.eval_str_nested_list_or_tuple(args.encoder_conv_kernel_sizes, type=int)
        strides = speech_utils.eval_str_nested_list_or_tuple(args.encoder_conv_strides, type=int)
        logger.info("input feature dimension: {}, channels: {}".format(task.feat_dim, task.feat_in_channels))
        assert task.feat_dim % task.feat_in_channels == 0
        conv_layers = ConvBNReLU(
            out_channels, kernel_sizes, strides, in_channels=task.feat_in_channels,
        ) if out_channels is not None else None

        rnn_encoder_input_size = task.feat_dim // task.feat_in_channels
        if conv_layers is not None:
            for stride in strides:
                if isinstance(stride, (list, tuple)):
                    assert len(stride) > 0
                    s = stride[1] if len(stride) > 1 else stride[0]
                else:
                    assert isinstance(stride, int)
                    s = stride
                rnn_encoder_input_size = (rnn_encoder_input_size + s - 1) // s
            rnn_encoder_input_size *= out_channels[-1]
        else:
            rnn_encoder_input_size = task.feat_dim

        encoder = SpeechChunkLSTMEncoder(
            conv_layers_before=conv_layers,
            input_size=rnn_encoder_input_size,
            hidden_size=args.encoder_rnn_hidden_size,
            num_layers=args.encoder_rnn_layers,
            dropout_in=args.encoder_rnn_dropout_in,
            dropout_out=args.encoder_rnn_dropout_out,
            bidirectional=args.encoder_rnn_bidirectional,
            residual=args.encoder_rnn_residual,
            num_targets=getattr(task, "num_targets", None),  # targets for encoder-only model
            chunk_width=getattr(task, "chunk_width", None),
            chunk_left_context=getattr(task, "chunk_left_context", 0),
            training_stage=getattr(task, "training_stage", True),
            max_source_positions=max_source_positions,
        )
        return cls(encoder, state_prior=getattr(task, "initial_state_prior", None))

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

    def update_state_prior(self, new_state_prior, factor=0.1):
        assert self.state_prior is not None
        self.state_prior = self.state_prior.to(new_state_prior)
        self.state_prior = (1. - factor) * self.state_prior + factor * new_state_prior
        self.state_prior = self.state_prior / self.state_prior.sum()  # re-normalize

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["state_prior"] = self.state_prior
        return state_dict

    def load_state_dict(self, state_dict, strict=True, args=None):
        state_dict_subset = state_dict.copy()
        self.state_prior = state_dict.get("state_prior", None)
        if "state_prior" in state_dict:
            self.state_prior = state_dict["state_prior"]
            del state_dict_subset["state_prior"]
        super().load_state_dict(state_dict_subset, strict=strict, args=args)


class SpeechChunkLSTMEncoder(SpeechLSTMEncoder):
    """LSTM encoder."""
    def __init__(
        self, conv_layers_before=None, input_size=83, hidden_size=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, bidirectional=False,
        residual=False, left_pad=False, padding_value=0.,
        num_targets=None, chunk_width=20, chunk_left_context=0, training_stage=True,
        max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS,
    ):
        super().__init__(
            conv_layers_before=conv_layers_before, input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, dropout_in=dropout_in,  dropout_out=dropout_in,
            bidirectional=bidirectional, residual=residual, left_pad=left_pad,
            padding_value=padding_value, max_source_positions=max_source_positions,
        )
        receptive_field_radius = sum(conv.padding[0] for conv in conv_layers_before.convolutions) \
            if conv_layers_before is not None else 0
        assert chunk_width is None or chunk_width > 0
        assert (conv_layers_before is None and chunk_left_context >= 0) or \
            (conv_layers_before is not None and chunk_left_context >= receptive_field_radius)
        self.out_chunk_begin = self.output_lengths(chunk_left_context + 1) - 1
        self.out_chunk_end = self.output_lengths(chunk_left_context + chunk_width) \
            if chunk_width is not None else None
        self.training_stage = training_stage

        # only for encoder-only model
        self.fc_out = Linear(self.output_units, num_targets, dropout=dropout_out) \
            if num_targets is not None else None

    def forward(
        self,
        src_tokens,
        src_lengths: Tensor,
        enforce_sorted: bool = True,
        **unused,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of
                shape `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of
                shape `(batch)`
            enforce_sorted (bool, optional): if True, `src_tokens` is
                expected to contain sequences sorted by length in a
                decreasing order. If False, this condition is not
                required. Default: True.
        """
        out = super().forward(src_tokens, src_lengths, enforce_sorted=enforce_sorted, **unused)
        x, encoder_padding_mask, x_lengths = out.encoder_out, out.encoder_padding_mask, out.src_lengths

        # determine which output frame to select for loss evaluation/test, assuming
        # all examples in a batch are of the same length for chunk-wise training/test
        if (
            self.out_chunk_end is not None
            and (self.training or not self.training_stage)
        ):
            x = x[self.out_chunk_begin: self.out_chunk_end]  # T x B x C -> W x B x C
            x_lengths = x_lengths.fill_(x.size(0))
            assert encoder_padding_mask is None

        if self.fc_out is not None:
            x = self.fc_out(x)  # T x B x C -> T x B x V

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask if encoder_padding_mask.any() else None,  # T x B
            encoder_embedding=None,
            encoder_states=None,
            src_tokens=None,
            src_lengths=x_lengths,  # B
        )


@register_model_architecture("speech_lstm_encoder_model", "speech_lstm_encoder_model")
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.4)
    args.encoder_conv_channels = getattr(
        args, "encoder_conv_channels", "[64, 64, 128, 128]",
    )
    args.encoder_conv_kernel_sizes = getattr(
        args, "encoder_conv_kernel_sizes", "[(3, 3), (3, 3), (3, 3), (3, 3)]",
    )
    args.encoder_conv_strides = getattr(
        args, "encoder_conv_strides", "[(1, 1), (2, 2), (1, 1), (2, 2)]",
    )
    args.encoder_rnn_hidden_size = getattr(args, "encoder_rnn_hidden_size", 320)
    args.encoder_rnn_layers = getattr(args, "encoder_rnn_layers", 3)
    args.encoder_rnn_bidirectional = getattr(args, "encoder_rnn_bidirectional", True)
    args.encoder_rnn_residual = getattr(args, "encoder_rnn_residual", False)
    args.encoder_rnn_dropout_in = getattr(args, "encoder_rnn_dropout_in", args.dropout)
    args.encoder_rnn_dropout_out = getattr(args, "encoder_rnn_dropout_out", args.dropout)


@register_model_architecture("speech_lstm_encoder_model", "speech_conv_lstm_encoder_model_wsj")
def encoder_conv_lstm_wsj(args):
    base_architecture(args)
