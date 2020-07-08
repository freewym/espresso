# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, Optional

import torch
from torch import Tensor
import torch.nn as nn

from fairseq import options
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import (
    Linear,
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder,
)
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    TransformerDecoderLayer,
)
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

from espresso.models.speech_lstm import ConvBNReLU
from espresso.tools.scheduled_sampling_rate_scheduler import ScheduledSamplingRateScheduler
import espresso.tools.utils as speech_utils


DEFAULT_MAX_SOURCE_POSITIONS = 10240
DEFAULT_MAX_TARGET_POSITIONS = 1024


logger = logging.getLogger(__name__)


@register_model('speech_transformer')
class SpeechTransformerModel(TransformerModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_. It adds 2D convolutions before
    transformer layers in the encoder to process speech input.

    Args:
        encoder (SpeechTransformerEncoder): the encoder
        decoder (SpeechTransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    @classmethod
    def hub_models(cls):
        raise NotImplementedError

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.num_updates = 0

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        TransformerModel.add_args(parser)
        parser.add_argument("--encoder-conv-channels", type=str, metavar="EXPR",
                            help="list of encoder convolution\'s out channels")
        parser.add_argument("--encoder-conv-kernel-sizes", type=str, metavar="EXPR",
                            help="list of encoder convolution\'s kernel sizes")
        parser.add_argument("--encoder-conv-strides", type=str, metavar="EXPR",
                            help="list of encoder convolution\'s strides")
        parser.add_argument("--encoder-transformer-context", type=str, metavar="EXPR",
                            help="left/right context for time-restricted self-attention; "
                            "can be None or a tuple of two non-negative integers/None")
        parser.add_argument("--decoder-input-dim", type=int, metavar="N",
                            help="decoder input dimension (extra linear layer "
                                 "if different from decoder embed dim)")

        # Scheduled sampling options
        parser.add_argument("--scheduled-sampling-probs", type=lambda p: options.eval_str_list(p),
                            metavar="P_1,P_2,...,P_N", default=[1.0],
                            help="scheduled sampling probabilities of sampling the truth "
                            "labels for N epochs starting from --start-schedule-sampling-epoch; "
                            "all later epochs using P_N")
        parser.add_argument("--start-scheduled-sampling-epoch", type=int,
                            metavar="N", default=1,
                            help="start scheduled sampling from the specified epoch")
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        tgt_dict = task.target_dictionary

        decoder_embed_tokens = cls.build_embedding(
            args, tgt_dict, args.decoder_input_dim, args.decoder_embed_path
        )

        out_channels = speech_utils.eval_str_nested_list_or_tuple(args.encoder_conv_channels, type=int)
        kernel_sizes = speech_utils.eval_str_nested_list_or_tuple(args.encoder_conv_kernel_sizes, type=int)
        strides = speech_utils.eval_str_nested_list_or_tuple(args.encoder_conv_strides, type=int)
        logger.info("input feature dimension: {}, channels: {}".format(task.feat_dim, task.feat_in_channels))
        assert task.feat_dim % task.feat_in_channels == 0
        conv_layers = ConvBNReLU(
            out_channels, kernel_sizes, strides, in_channels=task.feat_in_channels,
        ) if out_channels is not None else None

        transformer_encoder_input_size = task.feat_dim // task.feat_in_channels
        if conv_layers is not None:
            for stride in strides:
                if isinstance(stride, (list, tuple)):
                    assert len(stride) > 0
                    s = stride[1] if len(stride) > 1 else stride[0]
                else:
                    assert isinstance(stride, int)
                    s = stride
                transformer_encoder_input_size = (transformer_encoder_input_size + s - 1) // s
            transformer_encoder_input_size *= out_channels[-1]
        else:
            transformer_encoder_input_size = task.feat_dim

        encoder_transformer_context = speech_utils.eval_str_nested_list_or_tuple(
            args.encoder_transformer_context, type=int,
        )
        if encoder_transformer_context is not None:
            assert len(encoder_transformer_context) == 2
            for i in range(2):
                assert (
                    encoder_transformer_context[i] is None
                    or (
                        isinstance(encoder_transformer_context[i], int)
                        and encoder_transformer_context[i] >= 0
                    )
                )

        scheduled_sampling_rate_scheduler = ScheduledSamplingRateScheduler(
            args.scheduled_sampling_probs, args.start_scheduled_sampling_epoch,
        )

        encoder = cls.build_encoder(
            args, conv_layers_before=conv_layers, input_size=transformer_encoder_input_size,
            transformer_context=encoder_transformer_context,
        )
        decoder = cls.build_decoder(
            args, tgt_dict, decoder_embed_tokens,
            scheduled_sampling_rate_scheduler=scheduled_sampling_rate_scheduler,
        )
        return cls(args, encoder, decoder)

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates
        super().set_num_updates(num_updates)

    @classmethod
    def build_encoder(cls, args, conv_layers_before=None, input_size=83, transformer_context=None):
        return SpeechTransformerEncoder(
            args, conv_layers_before=conv_layers_before, input_size=input_size,
            transformer_context=transformer_context,
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens, scheduled_sampling_rate_scheduler=None):
        return SpeechTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
            scheduled_sampling_rate_scheduler=scheduled_sampling_rate_scheduler,
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        epoch=1,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            epoch=epoch,
        )
        return decoder_out


class SpeechTransformerEncoder(TransformerEncoder):
    """
    Transformer encoder consisting of 2D convolution layers and
    *args.encoder_layers* layers. Each layer is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        conv_layers_before (~fairseq.speech_lstm.ConvBNReLU): convolutions before
            transformer layers
        input_size (int, optional): dimension of the input to the transformer
            before being projected to args.encoder_embed_dim
    """

    def __init__(self, args, conv_layers_before=None, input_size=83, transformer_context=None):
        super(TransformerEncoder, self).__init__(None)  # no src dictionary
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = args.encoder_embed_dim
        self.max_source_positions = args.max_source_positions

        self.embed_positions = None

        self.conv_layers_before = conv_layers_before
        self.fc0 = Linear(input_size, embed_dim) if input_size != embed_dim else None

        self.embed_positions = (
            PositionalEmbedding(
                self.output_lengths(args.max_source_positions),
                embed_dim,
                0,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.transformer_context = transformer_context

    def output_lengths(self, in_lengths):
        return in_lengths if self.conv_layers_before is None \
            else self.conv_layers_before.output_lengths(in_lengths)

    def get_attn_mask(self, in_lengths):
        """
        Create attention mask according to sequence lengths and transformer context.

        Args:
            in_lengths (LongTensor): lengths of each input sequence of shape `(batch)`

        Returns:
            attn_mask (ByteTensor|BoolTensor, optional): self-attention mask of shape
            `(tgt_len, src_len)`, where `tgt_len` is the length of output and `src_len`
            is the length of input, though here both are equal to `seq_len`.
            `attn_mask[tgt_i, src_j] = 1` means that when calculating the
            embedding for `tgt_i`, we exclude (mask out) `src_j`.
        """
        if (
            self.transformer_context is None
            or (self.transformer_context[0] is None and self.transformer_context[1] is None)
        ):
            return None
        max_len = in_lengths.data.max()
        all_ones = in_lengths.ones([max_len, max_len], dtype=torch.bool)
        # at this point left and right context cannot be both None
        if self.transformer_context[0] is None:  # mask is a triu matrix
            return all_ones.triu(self.transformer_context[1] + 1)
        if self.transformer_context[1] is None:  # mask is a tril matrix
            return all_ones.tril(-self.transformer_context[0] - 1)
        return (
            all_ones.triu(self.transformer_context[1] + 1) | all_ones.tril(-self.transformer_context[0] - 1)
        )

    def forward(self, src_tokens, src_lengths, return_all_hiddens: bool = False):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            namedtuple:
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
        if self.conv_layers_before is not None:
            x, src_lengths, encoder_padding_mask = self.conv_layers_before(src_tokens, src_lengths)
        else:
            x, encoder_padding_mask = src_tokens, \
                ~speech_utils.sequence_mask(src_lengths, src_tokens.size(1))

        x = self.dropout_module(x)
        if self.fc0 is not None:
            x = self.fc0(x)
            if self.embed_positions is not None:
                # 0s in `~encoder_padding_mask` are used as pad_idx for positional embeddings
                x = x + self.embed_positions((~encoder_padding_mask).int())
            if self.layernorm_embedding is not None:
                x = self.layernorm_embedding(x)
            x = self.dropout_module(x)
        elif self.embed_positions is not None:
            # 0s in `~encoder_padding_mask` are used as pad_idx for positional embeddings
            x = x + self.embed_positions((~encoder_padding_mask).int())
            if self.layernorm_embedding is not None:
                x = self.layernorm_embedding(x)

        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        attn_mask = self.get_attn_mask(src_lengths)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask, attn_mask=attn_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=None,
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions


class SpeechTransformerDecoder(TransformerDecoder):
    def __init__(
        self, args, dictionary, embed_tokens, no_encoder_attn=False,
        scheduled_sampling_rate_scheduler=None,
    ):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn)

        self.scheduled_sampling_rate_scheduler = scheduled_sampling_rate_scheduler
        for layer in self.layers:
            if isinstance(layer, TransformerDecoderLayer):
                layer.need_attn = False  # make validation fast

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        **kwargs,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (EncoderOut, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        if self.training and alignment_layer is None:  # no attention tensors during training to save memory
            alignment_layer = self.num_layers  # can be any value no less than this
        if self.training and self.scheduled_sampling_rate_scheduler is not None:
            epoch = kwargs.get("epoch", 1)
            sampling_prob = self.scheduled_sampling_rate_scheduler.step(epoch)
            if sampling_prob < 1.0:  # apply scheduled sampling
                assert not features_only
                return self._forward_with_scheduled_sampling(
                    prev_output_tokens, sampling_prob, encoder_out=encoder_out,
                    incremental_state={},  # use empty dict to preserve forward state
                    alignment_layer=alignment_layer,
                    alignment_heads=alignment_heads,
                    src_lengths=src_lengths,
                    return_all_hiddens=return_all_hiddens,
                )

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def _forward_with_scheduled_sampling(
        self,
        prev_output_tokens,
        sampling_prob,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        bsz, seqlen = prev_output_tokens.size()
        outs = []
        pred = None
        for step in range(seqlen):
            if step > 0:
                sampling_mask = torch.rand(
                    [bsz, 1], device=prev_output_tokens.device,
                ).lt(sampling_prob)
                feed_tokens = torch.where(
                    sampling_mask, prev_output_tokens[:, step:step + 1], pred,
                )
            else:
                feed_tokens = prev_output_tokens[:, step:step + 1]  # B x 1
            x, _ = self.extract_features(
                feed_tokens,
                encoder_out=encoder_out,
                incremental_state=incremental_state,
                alignment_layer=alignment_layer,
                alignment_heads=alignment_heads,
            )
            x = self.output_layer(x)  # B x 1 x V
            outs.append(x)
            pred = x.argmax(-1)  # B x 1
        x = torch.cat(outs, dim=1)  # B x T x V
        return x, None

    def masked_copy_incremental_state(self, incremental_state, another_cached_state, mask):
        raise NotImplementedError


@register_model_architecture("speech_transformer", "speech_transformer")
def base_architecture(args):
    args.encoder_conv_channels = getattr(
        args, "encoder_conv_channels", "[64, 64, 128, 128]",
    )
    args.encoder_conv_kernel_sizes = getattr(
        args, "encoder_conv_kernel_sizes", "[(3, 3), (3, 3), (3, 3), (3, 3)]",
    )
    args.encoder_conv_strides = getattr(
        args, "encoder_conv_strides", "[(1, 1), (2, 2), (1, 1), (2, 2)]",
    )
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.encoder_transformer_context = getattr(args, "encoder_transformer_context", None)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.2)
    args.activation_dropout = getattr(args, "activation_dropout", 0.2)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.2)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)


@register_model_architecture("speech_transformer", "speech_transformer_wsj")
def speech_transformer_wsj(args):
    base_architecture(args)


@register_model_architecture("speech_transformer", "speech_transformer_librispeech")
def speech_transformer_librispeech(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_transformer_context = getattr(args, "encoder_transformer_context", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.dropout = getattr(args, "dropout", 0.1)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    base_architecture(args)


@register_model_architecture("speech_transformer", "speech_transformer_swbd")
def speech_transformer_swbd(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_transformer_context = getattr(args, "encoder_transformer_context", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.attention_dropout = getattr(args, "attention_dropout", 0.25)
    args.activation_dropout = getattr(args, "activation_dropout", 0.25)
    args.dropout = getattr(args, "dropout", 0.25)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    base_architecture(args)
