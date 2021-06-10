# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
import torch.nn as nn

from fairseq import utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.distributed import fsdp_wrap
from fairseq.models import (
    register_model,
    register_model_architecture,
)
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
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from omegaconf import II

from espresso.modules import (
    ConvBNReLU,
    TransformerWithRelativePositionalEmbeddingDecoderLayer,
    TransformerWithRelativePositionalEmbeddingEncoderLayer,
)
from espresso.tools.scheduled_sampling_rate_scheduler import ScheduledSamplingRateScheduler
import espresso.tools.utils as speech_utils


DEFAULT_MAX_SOURCE_POSITIONS = 10240
DEFAULT_MAX_TARGET_POSITIONS = 1024


DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)


logger = logging.getLogger(__name__)


@dataclass
class SpeechTransformerModelConfig(FairseqDataclass):
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="relu", metadata={"help": "activation function to use"}
    )
    dropout: float = field(default=0.2, metadata={"help": "dropout probability"})
    attention_dropout: float = field(
        default=0.2, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.2, metadata={"help": "dropout probability after activation in FFN."}
    )
    encoder_embed_dim: int = field(
        default=256, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=1024, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers"}
    )
    encoder_attention_heads: int = field(
        default=4, metadata={"help": "num encoder attention heads"}
    )
    encoder_normalize_before: bool = field(
        default=True, metadata={"help": "apply layernorm before each encoder block"}
    )
    encoder_learned_pos: bool = field(
        default=False, metadata={"help": "use learned positional embeddings in the encoder"}
    )
    encoder_relative_positional_embeddings: bool = field(
        default=False,
        metadata={"help": "if set, uses relative positional embeddings (inside self attention) for encoder"}
    )
    decoder_embed_path: Optional[str] = field(
        default=None, metadata={"help": "path to pre-trained decoder embedding"}
    )
    decoder_embed_dim: Optional[int] = field(
        default=II("model.encoder_embed_dim"), metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: Optional[int] = field(
        default=II("model.encoder_ffn_embed_dim"), metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=6, metadata={"help": "num decoder layers"})
    decoder_attention_heads: int = field(
        default=4, metadata={"help": "num decoder attention heads"}
    )
    decoder_learned_pos: bool = field(
        default=False, metadata={"help": "use learned positional embeddings in the decoder"}
    )
    decoder_relative_positional_embeddings: bool = field(
        default=False,
        metadata={"help": "if set, uses relative positional embeddings (inside self attention) for decoder"}
    )
    decoder_normalize_before: bool = field(
        default=True, metadata={"help": "apply layernorm before each decoder block"}
    )
    decoder_output_dim: Optional[int] = field(
        default=II("model.decoder_embed_dim"),
        metadata={"help": "decoder output dimension (extra linear layer if different from decoder embed dim)"}
    )
    decoder_input_dim: Optional[int] = field(
        default=II("model.decoder_embed_dim"),
        metadata={"help": "decoder input dimension (extra linear layer if different from decoder embed dim)"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    adaptive_input: bool = field(
        default=False, metadata={"help": "if set, uses adaptive input"}
    )
    tie_adaptive_weights: bool = field(
        default=False,
        metadata={
            "help": "if set, ties the weights of adaptive softmax and adaptive input"
        },
    )
    tie_adaptive_proj: bool = field(
        default=False,
        metadata={
            "help": "if set, ties the projection weights of adaptive softmax and adaptive input"
        },
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={"help": "if set, disables positional embeddings (outside self attention)"}
    )
    adaptive_softmax_cutoff: Optional[str] = field(
        default=None,
        metadata={
            "help": "comma separated list of adaptive softmax cutoff points. "
            "Must be used with adaptive_loss criterion"
        },
    )
    adaptive_softmax_dropout: float = field(
        default=0,
        metadata={"help": "sets adaptive softmax dropout for the tail projections"},
    )
    layernorm_embedding: bool = field(
        default=False, metadata={"help": "add layernorm to embedding"}
    )
    no_scale_embedding: bool = field(
        default=False, metadata={"help": "if True, dont scale embeddings"}
    )
    checkpoint_activations: bool = field(
        default=False,
        metadata={
            "help": "checkpoint activations at each layer, which saves GPU "
            "memory usage at the cost of some additional compute"
        },
    )
    offload_activations: bool = field(
        default=False,
        metadata={"help": "move checkpointed activations to CPU after they are used."},
    )
    # config for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
    no_cross_attention: bool = field(
        default=False, metadata={"help": "do not perform cross-attention"}
    )
    cross_self_attention: bool = field(
        default=False, metadata={"help": "perform cross+self-attention"}
    )
    # config for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
    encoder_layerdrop: float = field(
        default=0.0, metadata={"help": "LayerDrop probability for encoder"}
    )
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "LayerDrop probability for decoder"}
    )
    encoder_layers_to_keep: Optional[str] = field(
        default=None,
        metadata={
            "help": "which encoder layers to *keep* when pruning as a comma-separated list"
        },
    )
    decoder_layers_to_keep: Optional[str] = field(
        default=None,
        metadata={
            "help": "which decoder layers to *keep* when pruning as a comma-separated list"
        },
    )
    # config for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
    quant_noise_pq: float = field(
        default=0.0,
        metadata={"help": "iterative PQ quantization noise at training time"},
    )
    quant_noise_pq_block_size: int = field(
        default=8,
        metadata={"help": "block size of quantization noise at training time"},
    )
    quant_noise_scalar: float = field(
        default=0.0,
        metadata={
            "help": "scalar quantization noise and scalar quantization at training time"
        },
    )
    # config for Fully Sharded Data Parallel (FSDP) training
    min_params_to_wrap: int = field(
        default=DEFAULT_MIN_PARAMS_TO_WRAP,
        metadata={
            "help": (
                "minimum number of params for a layer to be wrapped with FSDP() when "
                "training with --ddp-backend=fully_sharded. Smaller values will "
                "improve memory efficiency, but may make torch.distributed "
                "communication less efficient due to smaller input sizes. This option "
                "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
                "--offload-activations are passed."
            )
        }
    )
    # config for "BASE Layers: Simplifying Training of Large, Sparse Models"
    base_layers: Optional[int] = field(
        default=0, metadata={"help": "number of BASE layers in total"}
    )
    base_sublayers: Optional[int] = field(
        default=1, metadata={"help": "number of sublayers in each BASE layer"}
    )
    base_shuffle: Optional[bool] = field(
        default=False, metadata={"help": "shuffle tokens between workers before computing assignment"}
    )
    # config specific for SpeechTransformerModel
    encoder_conv_channels: str = field(
        default="[64, 64, 128, 128]", metadata={"help": "list of encoder convolution\'s out channels"}
    )
    encoder_conv_kernel_sizes: str = field(
        default="[(3, 3), (3, 3), (3, 3), (3, 3)]", metadata={"help": "list of encoder convolution\'s kernel sizes"}
    )
    encoder_conv_strides: str = field(
        default="[(1, 1), (2, 2), (1, 1), (2, 2)]", metadata={"help": "list of encoder convolution\'s out strides"}
    )
    encoder_transformer_context: Optional[str] = field(
        default=None,
        metadata={
            "help": "left/right context for time-restricted self-attention; "
            "can be None or a tuple of two non-negative integers/None"
        }
    )
    # config for scheduled sampling
    scheduled_sampling_probs: List[float] = field(
        default_factory=lambda: [1.0],
        metadata={
            "help": "scheduled sampling probabilities of sampling the truth "
            "labels for N epochs starting from --start-schedule-sampling-epoch; "
            "all later epochs using the last value in the list"
        }
    )
    start_scheduled_sampling_epoch: int = field(
        default=1, metadata={"help": "start scheduled sampling from the specified epoch"}
    )
    # options from other parts of the config
    max_source_positions: Optional[int] = II("task.max_source_positions")
    max_target_positions: Optional[int] = II("task.max_target_positions")
    tpu: bool = II("common.tpu")


@register_model("speech_transformer", dataclass=SpeechTransformerModelConfig)
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

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            from fairseq.dataclass.utils import gen_parser_from_dataclass
            # do not set defaults so that settings defaults from various architectures still works
            gen_parser_from_dataclass(parser, dc(), delete_default=True)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

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
        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing

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
        min_params_to_wrap = getattr(
            args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
        )
        # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
        encoder = fsdp_wrap(encoder, min_num_params=min_params_to_wrap)
        decoder = fsdp_wrap(decoder, min_num_params=min_params_to_wrap)
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
        conv_layers_before (~espresso.modules.ConvBNReLU): convolutions before
            transformer layers
        input_size (int, optional): dimension of the input to the transformer
            before being projected to args.encoder_embed_dim
    """

    def __init__(self, args, conv_layers_before=None, input_size=83, transformer_context=None):
        self.args = args
        super(TransformerEncoder, self).__init__(None)  # no src dictionary
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = args.encoder_embed_dim
        self.max_source_positions = args.max_source_positions

        self.conv_layers_before = conv_layers_before
        self.fc0 = Linear(input_size, embed_dim) if input_size != embed_dim else None

        if not args.no_token_positional_embeddings and args.encoder_relative_positional_embeddings:
            logger.info("disabled encoder's absolute positional embeddings as encoder_relative_positional_embeddings is True.")
        self.embed_positions = (
            PositionalEmbedding(
                self.output_lengths(self.max_source_positions),
                embed_dim,
                0,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings and not args.encoder_relative_positional_embeddings
            else None
        )

        export = getattr(args, "export", False)
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim, export=export)
        else:
            self.layernorm_embedding = None

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
            self.layer_norm = LayerNorm(embed_dim, export=export)
        else:
            self.layer_norm = None

        self.transformer_context = transformer_context

    def build_encoder_layer(self, args):
        orig_max_source_positions = args.max_source_positions
        args.max_source_positions = self.output_lengths(args.max_source_positions)
        layer = TransformerWithRelativePositionalEmbeddingEncoderLayer(args)
        args.max_source_positions = orig_max_source_positions
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint
            else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def output_lengths(self, in_lengths):
        return (
            in_lengths if self.conv_layers_before is None
            else self.conv_layers_before.output_lengths(in_lengths)
        )

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
            src_lengths (torch.LongTensor): lengths of each source sentence of
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
        return self.forward_scriptable(src_tokens, src_lengths, return_all_hiddens)

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        src_tokens,
        src_lengths,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
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
        if self.conv_layers_before is not None:
            x, src_lengths, encoder_padding_mask = self.conv_layers_before(src_tokens, src_lengths)
        else:
            x, encoder_padding_mask = (
                src_tokens,
                ~speech_utils.sequence_mask(src_lengths, src_tokens.size(1))
            )
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

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

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        attn_mask = self.get_attn_mask(src_lengths)

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask=encoder_padding_mask if has_pads else None, attn_mask=attn_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask] if encoder_padding_mask.any()
            else [],  # B x T
            "encoder_embedding": [],
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions


class SpeechTransformerDecoder(TransformerDecoder):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
        scheduled_sampling_rate_scheduler=None,
    ):
        is_no_token_positional_embeddings_changed = False
        if not args.no_token_positional_embeddings and args.decoder_relative_positional_embeddings:
            args.no_token_positional_embeddings = True
            is_no_token_positional_embeddings_changed = True
            logger.info("disabled decoder's absolute positional embeddings as decoder_relative_positional_embeddings is True.")
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn, output_projection=output_projection)
        if is_no_token_positional_embeddings_changed:
            args.no_token_positional_embeddings = not args.no_token_positional_embeddings

        self.scheduled_sampling_rate_scheduler = scheduled_sampling_rate_scheduler
        for layer in self.layers:
            if isinstance(layer, TransformerWithRelativePositionalEmbeddingDecoderLayer):
                layer.need_attn = False  # make validation fast

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = TransformerWithRelativePositionalEmbeddingDecoderLayer(args, no_encoder_attn)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint
            else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
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
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

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
                    full_context_alignment=full_context_alignment,
                    alignment_layer=alignment_layer,
                    alignment_heads=alignment_heads,
                    src_lengths=src_lengths,
                    return_all_hiddens=return_all_hiddens,
                )

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
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
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
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
                full_context_alignment=full_context_alignment,
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
    args.encoder_relative_positional_embeddings = getattr(args, "encoder_relative_positional_embeddings", False)
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
    args.decoder_relative_positional_embeddings = getattr(args, "decoder_relative_positional_embeddings", False)
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
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)


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
