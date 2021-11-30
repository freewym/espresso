# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from espresso.models.transformer import (
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
    SpeechTransformerConfig,
)
from espresso.models.transformer.speech_transformer_base import (
    SpeechTransformerModelBase,
)
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models import register_model, register_model_architecture

logger = logging.getLogger(__name__)


@register_model("speech_transformer")
class SpeechTransformerModel(SpeechTransformerModelBase):
    """
    This is the legacy implementation of the speech transformer model that
    uses argparse for configuration.
    """

    @classmethod
    def hub_models(cls):
        raise NotImplementedError

    def __init__(self, args, encoder, decoder):
        cfg = SpeechTransformerConfig.from_namespace(args)
        super().__init__(cfg, encoder, decoder)
        self.args = args

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        # do not set defaults so that settings defaults from various architectures still works
        gen_parser_from_dataclass(
            parser, SpeechTransformerConfig(), delete_default=True, with_prefix=""
        )

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

        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing

        args.min_params_to_wrap = getattr(
            args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
        )
        cfg = SpeechTransformerConfig.from_namespace(args)
        return super().build_model(cfg, task)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        return super().build_embedding(
            SpeechTransformerConfig.from_namespace(args), dictionary, embed_dim, path
        )

    @classmethod
    def build_encoder(
        cls, args, conv_layers_before=None, input_size=83, transformer_context=None
    ):
        return super().build_encoder(
            SpeechTransformerConfig.from_namespace(args),
            conv_layers_before=conv_layers_before,
            input_size=input_size,
            transformer_context=transformer_context,
        )

    @classmethod
    def build_decoder(
        cls, args, tgt_dict, embed_tokens, scheduled_sampling_rate_scheduler=None
    ):
        return super().build_decoder(
            SpeechTransformerConfig.from_namespace(args),
            tgt_dict,
            embed_tokens,
            scheduled_sampling_rate_scheduler=scheduled_sampling_rate_scheduler,
        )


# architectures


def base_architecture(args):
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
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.encoder_relative_positional_embeddings = getattr(
        args, "encoder_relative_positional_embeddings", False
    )
    args.encoder_transformer_context = getattr(
        args, "encoder_transformer_context", None
    )
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.decoder_relative_positional_embeddings = getattr(
        args, "decoder_relative_positional_embeddings", False
    )
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
    args.encoder_transformer_context = getattr(
        args, "encoder_transformer_context", None
    )
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
    args.encoder_transformer_context = getattr(
        args, "encoder_transformer_context", None
    )
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
