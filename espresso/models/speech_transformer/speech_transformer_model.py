# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import logging
from typing import List, Optional

from fairseq import utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.distributed import fsdp_wrap
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    DEFAULT_MIN_PARAMS_TO_WRAP,
    TransformerEncoder,
    TransformerModel,
)
from omegaconf import II

from espresso.models import speech_transformer
from espresso.modules import ConvBNReLU
from espresso.tools.scheduled_sampling_rate_scheduler import ScheduledSamplingRateScheduler
import espresso.tools.utils as speech_utils


DEFAULT_MAX_SOURCE_POSITIONS = 10240
DEFAULT_MAX_TARGET_POSITIONS = 1024


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
        return speech_transformer.SpeechTransformerEncoder(
            args, conv_layers_before=conv_layers_before, input_size=input_size,
            transformer_context=transformer_context,
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens, scheduled_sampling_rate_scheduler=None):
        return speech_transformer.SpeechTransformerDecoder(
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
