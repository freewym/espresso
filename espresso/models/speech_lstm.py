# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import II
from torch import Tensor

import espresso.tools.utils as speech_utils
from espresso.modules import ConvBNReLU, speech_attention
from espresso.tools.scheduled_sampling_rate_scheduler import (
    ScheduledSamplingRateScheduler,
)
from fairseq import checkpoint_utils, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import (
    FairseqDecoder,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.lstm import LSTM, Embedding, Linear, LSTMCell
from fairseq.modules import AdaptiveSoftmax, FairseqDropout

DEFAULT_MAX_SOURCE_POSITIONS = 1e5
DEFAULT_MAX_TARGET_POSITIONS = 1e5
ATTENTION_TYPE_CHOICES = ChoiceEnum(["bahdanau", "luong", "none"])


logger = logging.getLogger(__name__)


@dataclass
class SpeechLSTMModelConfig(FairseqDataclass):
    dropout: float = field(default=0.4, metadata={"help": "dropout probability"})
    encoder_conv_channels: str = field(
        default="[64, 64, 128, 128]",
        metadata={"help": "list of encoder convolution's out channels"},
    )
    encoder_conv_kernel_sizes: str = field(
        default="[(3, 3), (3, 3), (3, 3), (3, 3)]",
        metadata={"help": "list of encoder convolution's kernel sizes"},
    )
    encoder_conv_strides: str = field(
        default="[(1, 1), (2, 2), (1, 1), (2, 2)]",
        metadata={"help": "list of encoder convolution's out strides"},
    )
    encoder_rnn_hidden_size: int = field(
        default=320, metadata={"help": "encoder rnn's hidden size"}
    )
    encoder_rnn_layers: int = field(
        default=3, metadata={"help": "number of rnn encoder layers"}
    )
    encoder_rnn_bidirectional: bool = field(
        default=True, metadata={"help": "make all rnn layers of encoder bidirectional"}
    )
    encoder_rnn_residual: bool = field(
        default=False,
        metadata={
            "help": "create residual connections for rnn encoder "
            "layers (starting from the 2nd layer), i.e., the actual "
            "output of such layer is the sum of its input and output"
        },
    )
    encoder_multilayer_rnn_as_single_module: bool = field(
        default=False,
        metadata={
            "help": "if True use a single nn.Module.LSTM for multilayer LSTMs (faster and may fix a possible cuDNN error); "
            "otherwise use nn.ModuleList(for back-compatibility). Note: if True then encoder_rnn_residual is set to False"
        },
    )
    decoder_embed_path: Optional[str] = field(
        default=None, metadata={"help": "path to pre-trained decoder embedding"}
    )
    decoder_embed_dim: int = field(
        default=48, metadata={"help": "decoder embedding dimension"}
    )
    decoder_freeze_embed: bool = field(
        default=False, metadata={"help": "freeze decoder embeddings"}
    )
    decoder_hidden_size: int = field(
        default=320, metadata={"help": "decoder hidden size"}
    )
    decoder_layers: int = field(default=3, metadata={"help": "num decoder layers"})
    decoder_out_embed_dim: int = field(
        default=960, metadata={"help": "decoder output embedding dimension"}
    )
    decoder_rnn_residual: bool = field(
        default=True,
        metadata={
            "help": "create residual connections for rnn decoder "
            "layers (starting from the 2nd layer), i.e., the actual "
            "output of such layer is the sum of its input and output"
        },
    )
    attention_type: ATTENTION_TYPE_CHOICES = field(
        default="bahdanau",
        metadata={"help": "attention type ('bahdanau' or 'luong' or 'none')"},
    )
    attention_dim: int = field(default=320, metadata={"help": "attention dimension"})
    need_attention: bool = field(
        default=False,
        metadata={"help": "need to return attention tensor for the caller"},
    )
    adaptive_softmax_cutoff: Optional[str] = field(
        default=None,
        metadata={
            "help": "comma separated list of adaptive softmax cutoff points. "
            "Must be used with adaptive_loss criterion"
        },
    )
    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    pretrained_lm_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "path to load checkpoint from pretrained language model(LM), "
            "which will be present and kept fixed during training."
        },
    )
    # Granular dropout settings (if not specified these default to --dropout)
    encoder_rnn_dropout_in: Optional[float] = field(
        default=II("model.dropout"),
        metadata={"help": "dropout probability for encoder rnn's input"},
    )
    encoder_rnn_dropout_out: Optional[float] = field(
        default=II("model.dropout"),
        metadata={"help": "dropout probability for encoder rnn's output"},
    )
    decoder_dropout_in: Optional[float] = field(
        default=II("model.dropout"),
        metadata={"help": "dropout probability for decoder input embedding"},
    )
    decoder_dropout_out: Optional[float] = field(
        default=II("model.dropout"),
        metadata={"help": "dropout probability for decoder output"},
    )
    # config for scheduled sampling
    scheduled_sampling_probs: List[float] = field(
        default_factory=lambda: [1.0],
        metadata={
            "help": "scheduled sampling probabilities of sampling the truth "
            "labels for N epochs starting from --start-schedule-sampling-epoch; "
            "all later epochs using the last value in the list"
        },
    )
    start_scheduled_sampling_epoch: int = field(
        default=1,
        metadata={"help": "start scheduled sampling from the specified epoch"},
    )
    # options from other parts of the config
    max_source_positions: Optional[int] = II("task.max_source_positions")
    max_target_positions: Optional[int] = II("task.max_target_positions")
    tpu: bool = II("common.tpu")
    criterion_name: Optional[str] = II("criterion._name")


@register_model("speech_lstm", dataclass=SpeechLSTMModelConfig)
class SpeechLSTMModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder, pretrained_lm=None):
        super().__init__(encoder, decoder)
        self.num_updates = 0
        self.pretrained_lm = pretrained_lm
        if pretrained_lm is not None:
            assert isinstance(self.pretrained_lm, FairseqDecoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        max_source_positions = getattr(
            args, "max_source_positions", DEFAULT_MAX_SOURCE_POSITIONS
        )
        max_target_positions = getattr(
            args, "max_target_positions", DEFAULT_MAX_TARGET_POSITIONS
        )

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        # separate decoder input embeddings
        pretrained_decoder_embed = None
        if args.decoder_embed_path:
            pretrained_decoder_embed = load_pretrained_embedding_from_file(
                args.decoder_embed_path,
                task.target_dictionary,
                args.decoder_embed_dim,
            )
        # one last double check of parameter combinations
        if args.share_decoder_input_output_embed and (
            args.decoder_embed_dim != args.decoder_out_embed_dim
        ):
            raise ValueError(
                "--share-decoder-input-output-embed requires "
                "--decoder-embed-dim to match --decoder-out-embed-dim"
            )

        if args.decoder_freeze_embed:
            pretrained_decoder_embed.weight.requires_grad = False

        out_channels = speech_utils.eval_str_nested_list_or_tuple(
            args.encoder_conv_channels, type=int
        )
        kernel_sizes = speech_utils.eval_str_nested_list_or_tuple(
            args.encoder_conv_kernel_sizes, type=int
        )
        strides = speech_utils.eval_str_nested_list_or_tuple(
            args.encoder_conv_strides, type=int
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

        if args.encoder_multilayer_rnn_as_single_module and args.encoder_rnn_residual:
            args.encoder_rnn_residual = False
            logger.info(
                "--encoder-rnn-residual is set to False when --encoder-multilayer-rnn-as-single-module=True"
            )

        scheduled_sampling_rate_scheduler = ScheduledSamplingRateScheduler(
            args.scheduled_sampling_probs,
            args.start_scheduled_sampling_epoch,
        )

        encoder = SpeechLSTMEncoder(
            conv_layers_before=conv_layers,
            input_size=rnn_encoder_input_size,
            hidden_size=args.encoder_rnn_hidden_size,
            num_layers=args.encoder_rnn_layers,
            dropout_in=args.encoder_rnn_dropout_in,
            dropout_out=args.encoder_rnn_dropout_out,
            bidirectional=args.encoder_rnn_bidirectional,
            residual=args.encoder_rnn_residual,
            src_bucketed=(getattr(task.cfg, "num_batch_buckets", 0) > 0),
            max_source_positions=max_source_positions,
            multilayer_rnn_as_single_module=args.encoder_multilayer_rnn_as_single_module,
        )
        decoder = SpeechLSTMDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            encoder_output_units=encoder.output_units,
            attn_type=args.attention_type,
            attn_dim=args.attention_dim,
            need_attn=args.need_attention,
            residual=args.decoder_rnn_residual,
            pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args.share_decoder_input_output_embed,
            adaptive_softmax_cutoff=(
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion_name == "adaptive_loss"
                else None
            ),
            max_target_positions=max_target_positions,
            scheduled_sampling_rate_scheduler=scheduled_sampling_rate_scheduler,
        )
        pretrained_lm = None
        if args.pretrained_lm_checkpoint:
            logger.info(
                "loading pretrained LM from {}".format(args.pretrained_lm_checkpoint)
            )
            pretrained_lm = checkpoint_utils.load_model_ensemble(
                args.pretrained_lm_checkpoint, task=task
            )[0][0]
            pretrained_lm.make_generation_fast_()
            # freeze pretrained model
            for param in pretrained_lm.parameters():
                param.requires_grad = False
        return cls(encoder, decoder, pretrained_lm)

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        epoch=1,
    ):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            epoch=epoch,
        )
        return decoder_out

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates
        super().set_num_updates(num_updates)

    def max_positions(self):
        """Maximum length supported by the model."""
        return (
            self.encoder.max_positions(),
            self.decoder.max_positions()
            if self.pretrained_lm is None
            else min(self.decoder.max_positions(), self.pretrained_lm.max_positions()),
        )

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return (
            self.decoder.max_positions()
            if self.pretrained_lm is None
            else min(self.decoder.max_positions(), self.pretrained_lm.max_positions())
        )


class SpeechLSTMEncoder(FairseqEncoder):
    """LSTM encoder."""

    def __init__(
        self,
        conv_layers_before=None,
        input_size=83,
        hidden_size=512,
        num_layers=1,
        dropout_in=0.1,
        dropout_out=0.1,
        bidirectional=False,
        residual=False,
        left_pad=False,
        padding_value=0.0,
        src_bucketed=False,
        max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS,
        multilayer_rnn_as_single_module=False,
    ):
        super().__init__(None)  # no src dictionary
        self.conv_layers_before = conv_layers_before
        self.num_layers = num_layers
        self.dropout_in_module = FairseqDropout(
            dropout_in * 1.0, module_name=self.__class__.__name__
        )
        self.dropout_out_module = FairseqDropout(
            dropout_out * 1.0, module_name=self.__class__.__name__
        )
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.residual = residual
        self.max_source_positions = max_source_positions

        # enforce deterministic behavior (https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        self.multilayer_rnn_as_single_module = multilayer_rnn_as_single_module
        if self.multilayer_rnn_as_single_module:
            self.lstm = LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=self.dropout_out_module.p if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
        else:
            self.lstm = nn.ModuleList(
                [
                    LSTM(
                        input_size=input_size
                        if layer == 0
                        else 2 * hidden_size
                        if self.bidirectional
                        else hidden_size,
                        hidden_size=hidden_size,
                        bidirectional=bidirectional,
                    )
                    for layer in range(num_layers)
                ]
            )
        self.left_pad = left_pad
        self.padding_value = padding_value
        self.src_bucketed = src_bucketed

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def output_lengths(self, in_lengths):
        return (
            in_lengths
            if self.conv_layers_before is None
            else self.conv_layers_before.output_lengths(in_lengths)
        )

    def forward(
        self,
        src_tokens: Tensor,
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
        if self.left_pad:
            # nn.utils.rnn.pack_padded_sequence requires right-padding;
            # convert left-padding to right-padding
            src_tokens = speech_utils.convert_padding_direction(
                src_tokens,
                src_lengths,
                left_to_right=True,
            )

        if self.conv_layers_before is not None:
            x, src_lengths, padding_mask = self.conv_layers_before(
                src_tokens, src_lengths
            )
        else:
            x, padding_mask = (
                src_tokens,
                ~speech_utils.sequence_mask(src_lengths, src_tokens.size(1)),
            )

        bsz, seqlen = x.size(0), x.size(1)

        x = self.dropout_in_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        if self.multilayer_rnn_as_single_module:
            state_size = (
                (2 if self.bidirectional else 1) * self.num_layers,
                bsz,
                self.hidden_size,
            )
            h0, c0 = x.new_zeros(*state_size), x.new_zeros(*state_size)

            # pack embedded source tokens into a PackedSequence
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x,
                (
                    src_lengths.cpu()
                    if not self.src_bucketed
                    else src_lengths.new_full(
                        src_lengths.size(), x.size(0), device="cpu"
                    )
                ),
                enforce_sorted=enforce_sorted,
            )
            # apply LSTM
            packed_outs, (_, _) = self.lstm(packed_x, (h0, c0))
            # unpack outputs
            x, _ = nn.utils.rnn.pad_packed_sequence(
                packed_outs, padding_value=self.padding_value * 1.0
            )
        else:  # for back-compatibility
            state_size = 2 if self.bidirectional else 1, bsz, self.hidden_size
            h0, c0 = x.new_zeros(*state_size), x.new_zeros(*state_size)

            for i in range(len(self.lstm)):
                if (
                    self.residual and i > 0
                ):  # residual connection starts from the 2nd layer
                    prev_x = x
                # pack embedded source tokens into a PackedSequence
                packed_x = nn.utils.rnn.pack_padded_sequence(
                    x,
                    (
                        src_lengths.cpu()
                        if not self.src_bucketed
                        else src_lengths.new_full(
                            src_lengths.size(), x.size(0), device="cpu"
                        )
                    ),
                    enforce_sorted=enforce_sorted,
                )

                # apply LSTM
                packed_outs, (_, _) = self.lstm[i](packed_x, (h0, c0))

                # unpack outputs and apply dropout
                x, _ = nn.utils.rnn.pad_packed_sequence(
                    packed_outs, padding_value=self.padding_value * 1.0
                )
                if i < len(self.lstm) - 1:  # not applying dropout for the last layer
                    x = self.dropout_out_module(x)
                x = x + prev_x if self.residual and i > 0 else x
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        encoder_padding_mask = padding_mask.t()

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `foward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask]
            if encoder_padding_mask.any()
            else [],  # T x B
            "encoder_embedding": [],
            "encoder_states": [],
            "src_tokens": [],
            "src_lengths": [src_lengths],  # B
        }

    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(
                    1, new_order
                )  # note: transposed
            ]
        if len(encoder_out["src_lengths"]) == 0:
            new_src_lengths = []
        else:
            new_src_lengths = [
                (encoder_out["src_lengths"][0]).index_select(0, new_order)
            ]

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # T x B
            "encoder_embedding": [],
            "encoder_states": [],
            "src_tokens": [],
            "src_lengths": new_src_lengths,  # B
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions


class SpeechLSTMDecoder(FairseqIncrementalDecoder):
    """LSTM decoder."""

    def __init__(
        self,
        dictionary,
        embed_dim=512,
        hidden_size=512,
        out_embed_dim=512,
        num_layers=1,
        dropout_in=0.1,
        dropout_out=0.1,
        encoder_output_units=0,
        attn_type=None,
        attn_dim=0,
        need_attn=False,
        residual=False,
        pretrained_embed=None,
        share_input_output_embed=False,
        adaptive_softmax_cutoff=None,
        max_target_positions=DEFAULT_MAX_TARGET_POSITIONS,
        scheduled_sampling_rate_scheduler=None,
    ):
        super().__init__(dictionary)
        self.dropout_in_module = FairseqDropout(
            dropout_in * 1.0, module_name=self.__class__.__name__
        )
        self.dropout_out_module = FairseqDropout(
            dropout_out * 1.0, module_name=self.__class__.__name__
        )
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        if attn_type is None or str(attn_type).lower() == "none":
            # no attention, no encoder output needed (language model case)
            need_attn = False
            encoder_output_units = 0
        self.need_attn = need_attn
        self.residual = residual
        self.max_target_positions = max_target_positions
        self.num_layers = num_layers

        self.adaptive_softmax = None
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units

        self.layers = nn.ModuleList(
            [
                LSTMCell(
                    input_size=encoder_output_units
                    + (embed_dim if layer == 0 else hidden_size),
                    hidden_size=hidden_size,
                )
                for layer in range(num_layers)
            ]
        )

        if attn_type is None or str(attn_type).lower() == "none":
            self.attention = None
        elif str(attn_type).lower() == "bahdanau":
            self.attention = speech_attention.BahdanauAttention(
                hidden_size,
                encoder_output_units,
                attn_dim,
            )
        elif str(attn_type).lower() == "luong":
            self.attention = speech_attention.LuongAttention(
                hidden_size,
                encoder_output_units,
            )
        else:
            raise ValueError("unrecognized attention type.")

        if hidden_size + encoder_output_units != out_embed_dim:
            self.additional_fc = Linear(
                hidden_size + encoder_output_units, out_embed_dim
            )

        if adaptive_softmax_cutoff is not None:
            # setting adaptive_softmax dropout to dropout_out for now but can be redefined
            self.adaptive_softmax = AdaptiveSoftmax(
                num_embeddings,
                hidden_size,
                adaptive_softmax_cutoff,
                dropout=dropout_out,
            )
        elif not self.share_input_output_embed:
            self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)

        self.scheduled_sampling_rate_scheduler = scheduled_sampling_rate_scheduler

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        **kwargs,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - attention weights of shape `(batch, tgt_len, src_len)`
        """
        if self.training and self.scheduled_sampling_rate_scheduler is not None:
            epoch = kwargs.get("epoch", 1)
            sampling_prob = self.scheduled_sampling_rate_scheduler.step(epoch)
            if sampling_prob < 1.0:  # apply scheduled sampling
                return self._forward_with_scheduled_sampling(
                    prev_output_tokens,
                    sampling_prob,
                    encoder_out=encoder_out,
                    incremental_state={},  # use empty dict to preserve forward state
                )

        x, attn_scores = self.extract_features(
            prev_output_tokens,
            encoder_out,
            incremental_state,
        )
        return self.output_layer(x), attn_scores

    def _forward_with_scheduled_sampling(
        self,
        prev_output_tokens,
        sampling_prob,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        bsz, seqlen = prev_output_tokens.size()
        outs = []
        pred = None
        for step in range(seqlen):
            if step > 0:
                sampling_mask = torch.rand(
                    [bsz, 1],
                    device=prev_output_tokens.device,
                ).lt(sampling_prob)
                feed_tokens = torch.where(
                    sampling_mask,
                    prev_output_tokens[:, step : step + 1],
                    pred,
                )
            else:
                feed_tokens = prev_output_tokens[:, step : step + 1]  # B x 1
            x, _ = self.extract_features(feed_tokens, encoder_out, incremental_state)
            x = self.output_layer(x)  # B x 1 x V
            outs.append(x)
            pred = x.argmax(-1)  # B x 1
        x = torch.cat(outs, dim=1)  # B x T x V
        # ignore attention scores
        return x, None

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        **unused,
    ):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - attention weights of shape `(batch, tgt_len, src_len)`
        """
        # get outputs from encoder
        if encoder_out is not None:
            assert self.attention is not None
            encoder_outs = (
                encoder_out["encoder_out"][0]
                if len(encoder_out["encoder_out"]) > 0
                else torch.empty(0)
            )
            encoder_padding_mask = (
                encoder_out["encoder_padding_mask"][0]
                if len(encoder_out["encoder_padding_mask"]) > 0
                else None
            )
        else:
            encoder_outs = torch.empty(0)
            encoder_padding_mask = torch.empty(0)
        srclen = encoder_outs.size(0)

        if (
            incremental_state is not None
            and self._get_full_incremental_state_key("cached_state")
            in incremental_state
        ):
            prev_output_tokens = prev_output_tokens[:, -1:]

        bsz, seqlen = prev_output_tokens.size()

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = self.dropout_in_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        if (
            incremental_state is not None
            and self._get_full_incremental_state_key("cached_state")
            in incremental_state
        ):
            prev_hiddens, prev_cells, input_feed = self.get_cached_state(
                incremental_state
            )
        else:
            zero_state = x.new_zeros(bsz, self.hidden_size)
            prev_hiddens = [zero_state for i in range(self.num_layers)]
            prev_cells = [zero_state for i in range(self.num_layers)]
            input_feed = (
                x.new_zeros(bsz, self.encoder_output_units)
                if encoder_out is not None
                else None
            )

        attn_scores: Optional[Tensor] = (
            x.new_zeros(srclen, seqlen, bsz) if encoder_out is not None else None
        )
        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            if input_feed is not None:
                input = torch.cat((x[j, :, :], input_feed), dim=1)
            else:
                input = x[j, :, :]

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))
                if (
                    self.residual and i > 0
                ):  # residual connection starts from the 2nd layer
                    prev_layer_hidden = input[:, : hidden.size(1)]

                # compute and apply attention using the 1st layer's hidden state
                if encoder_out is not None:
                    if i == 0:
                        assert attn_scores is not None
                        context, attn_scores[:, j, :], _ = self.attention(
                            hidden,
                            encoder_outs,
                            encoder_padding_mask,
                        )

                    # hidden state concatenated with context vector becomes the
                    # input to the next layer
                    input = torch.cat((hidden, context), dim=1)
                else:
                    input = hidden
                input = self.dropout_out_module(input)
                if self.residual and i > 0:
                    if encoder_out is not None:
                        hidden_sum = input[:, : hidden.size(1)] + prev_layer_hidden
                        input = torch.cat(
                            (hidden_sum, input[:, hidden.size(1) :]), dim=1
                        )
                    else:
                        input = input + prev_layer_hidden

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # input feeding
            if input_feed is not None:
                input_feed = context

            # save final output
            outs.append(input)

        # Stack all the necessary tensors together and store
        prev_hiddens_tensor = torch.stack(prev_hiddens)
        prev_cells_tensor = torch.stack(prev_cells)
        cache_state = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_hiddens": prev_hiddens_tensor,
                "prev_cells": prev_cells_tensor,
                "input_feed": input_feed,
            },
        )
        self.set_incremental_state(incremental_state, "cached_state", cache_state)

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, -1)
        assert x.size(2) == self.hidden_size + self.encoder_output_units

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        if hasattr(self, "additional_fc") and self.adaptive_softmax is None:
            x = self.additional_fc(x)
            x = self.dropout_out_module(x)
        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and encoder_out is not None and self.need_attn:
            assert attn_scores is not None
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None

        return x, attn_scores

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.weight)
            else:
                return self.fc_out(features)
        else:
            return features

    def initialize_cached_state(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        bsz = prev_output_tokens.size(0)
        x = self.embed_tokens(prev_output_tokens)
        zero_states = x.new_zeros(self.num_layers, bsz, self.hidden_size)
        input_feed = (
            x.new_zeros(bsz, self.encoder_output_units)
            if encoder_out is not None
            else None
        )
        cache_state = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_hiddens": zero_states,
                "prev_cells": zero_states,
                "input_feed": input_feed,
            },
        )
        self.set_incremental_state(incremental_state, "cached_state", cache_state)

    def get_cached_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
    ) -> Tuple[List[Tensor], List[Tensor], Optional[Tensor]]:
        cached_state = self.get_incremental_state(incremental_state, "cached_state")
        assert cached_state is not None
        prev_hiddens_ = cached_state["prev_hiddens"]
        assert prev_hiddens_ is not None
        prev_cells_ = cached_state["prev_cells"]
        assert prev_cells_ is not None
        prev_hiddens = [prev_hiddens_[i] for i in range(self.num_layers)]
        prev_cells = [prev_cells_[j] for j in range(self.num_layers)]
        input_feed = cached_state[
            "input_feed"
        ]  # can be None for decoder-only language models

        return prev_hiddens, prev_cells, input_feed

    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        if (
            incremental_state is None
            or self._get_full_incremental_state_key("cached_state")
            not in incremental_state
        ):
            return
        prev_hiddens, prev_cells, input_feed = self.get_cached_state(incremental_state)
        prev_hiddens = [p.index_select(0, new_order) for p in prev_hiddens]
        prev_cells = [p.index_select(0, new_order) for p in prev_cells]
        if input_feed is not None:
            input_feed = input_feed.index_select(0, new_order)
        cached_state_new = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_hiddens": torch.stack(prev_hiddens),
                "prev_cells": torch.stack(prev_cells),
                "input_feed": input_feed,
            },
        )
        self.set_incremental_state(incremental_state, "cached_state", cached_state_new),
        return

    def masked_copy_cached_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        src_cached_state: Tuple[Optional[Union[List[torch.Tensor], torch.Tensor]]],
        mask: Tensor,
    ):
        if (
            incremental_state is None
            or self._get_full_incremental_state_key("cached_state")
            not in incremental_state
        ):
            assert src_cached_state is None or len(src_cached_state) == 0
            return
        prev_hiddens, prev_cells, input_feed = self.get_cached_state(incremental_state)
        src_prev_hiddens, src_prev_cells, src_input_feed = (
            src_cached_state[0],
            src_cached_state[1],
            src_cached_state[2],
        )

        def masked_copy_state(state: Optional[Tensor], src_state: Optional[Tensor]):
            if state is None:
                assert src_state is None
                return None
            else:
                assert (
                    state.size(0) == mask.size(0)
                    and src_state is not None
                    and state.size() == src_state.size()
                )
                state[mask, ...] = src_state[mask, ...]
                return state

        prev_hiddens = [
            masked_copy_state(p, src_p)
            for (p, src_p) in zip(prev_hiddens, src_prev_hiddens)
        ]
        prev_cells = [
            masked_copy_state(p, src_p)
            for (p, src_p) in zip(prev_cells, src_prev_cells)
        ]
        input_feed = masked_copy_state(input_feed, src_input_feed)

        cached_state_new = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_hiddens": torch.stack(prev_hiddens),
                "prev_cells": torch.stack(prev_cells),
                "input_feed": input_feed,
            },
        )
        self.set_incremental_state(incremental_state, "cached_state", cached_state_new)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.4)
    args.encoder_conv_channels = getattr(
        args,
        "encoder_conv_channels",
        "[64, 64, 128, 128]",
    )
    args.encoder_conv_kernel_sizes = getattr(
        args,
        "encoder_conv_kernel_sizes",
        "[(3, 3), (3, 3), (3, 3), (3, 3)]",
    )
    args.encoder_conv_strides = getattr(
        args,
        "encoder_conv_strides",
        "[(1, 1), (2, 2), (1, 1), (2, 2)]",
    )
    args.encoder_rnn_hidden_size = getattr(args, "encoder_rnn_hidden_size", 320)
    args.encoder_rnn_layers = getattr(args, "encoder_rnn_layers", 3)
    args.encoder_rnn_bidirectional = getattr(args, "encoder_rnn_bidirectional", True)
    args.encoder_rnn_residual = getattr(args, "encoder_rnn_residual", False)
    args.encoder_multilayer_rnn_as_single_module = getattr(
        args, "encoder_multilayer_rnn_as_single_module", False
    )
    if args.encoder_multilayer_rnn_as_single_module:
        args.encoder_rnn_residual = False
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 48)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_freeze_embed = getattr(args, "decoder_freeze_embed", False)
    args.decoder_hidden_size = getattr(args, "decoder_hidden_size", 320)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 960)
    args.decoder_rnn_residual = getattr(args, "decoder_rnn_residual", True)
    args.attention_type = getattr(args, "attention_type", "bahdanau")
    args.attention_dim = getattr(args, "attention_dim", 320)
    args.need_attention = getattr(args, "need_attention", False)
    args.encoder_rnn_dropout_in = getattr(args, "encoder_rnn_dropout_in", args.dropout)
    args.encoder_rnn_dropout_out = getattr(
        args, "encoder_rnn_dropout_out", args.dropout
    )
    args.decoder_dropout_in = getattr(args, "decoder_dropout_in", args.dropout)
    args.decoder_dropout_out = getattr(args, "decoder_dropout_out", args.dropout)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.pretrained_lm_checkpoint = getattr(args, "pretrained_lm_checkpoint", None)


@register_model_architecture("speech_lstm", "speech_conv_lstm_wsj")
def conv_lstm_wsj(args):
    base_architecture(args)


@register_model_architecture("speech_lstm", "speech_conv_lstm_librispeech")
def speech_conv_lstm_librispeech(args):
    args.dropout = getattr(args, "dropout", 0.3)
    args.encoder_rnn_hidden_size = getattr(args, "encoder_rnn_hidden_size", 1024)
    args.encoder_rnn_layers = getattr(args, "encoder_rnn_layers", 4)
    args.encoder_multilayer_rnn_as_single_module = getattr(
        args, "encoder_multilayer_rnn_as_single_module", True
    )
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_hidden_size = getattr(args, "decoder_hidden_size", 1024)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 3072)
    args.decoder_rnn_residual = getattr(args, "decoder_rnn_residual", True)
    args.attention_type = getattr(args, "attention_type", "bahdanau")
    args.attention_dim = getattr(args, "attention_dim", 512)
    base_architecture(args)


@register_model_architecture("speech_lstm", "speech_conv_lstm_swbd")
def speech_conv_lstm_swbd(args):
    args.dropout = getattr(args, "dropout", 0.5)
    args.encoder_rnn_hidden_size = getattr(args, "encoder_rnn_hidden_size", 640)
    args.encoder_rnn_layers = getattr(args, "encoder_rnn_layers", 4)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 640)
    args.decoder_hidden_size = getattr(args, "decoder_hidden_size", 640)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 1920)
    args.decoder_rnn_residual = getattr(args, "decoder_rnn_residual", True)
    args.attention_type = getattr(args, "attention_type", "bahdanau")
    args.attention_dim = getattr(args, "attention_dim", 640)
    base_architecture(args)
