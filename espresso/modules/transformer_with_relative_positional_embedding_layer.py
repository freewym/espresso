# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.models.transformer import TransformerConfig
from fairseq.modules import MultiheadAttention
from fairseq.modules.transformer_layer import (
    TransformerDecoderLayerBase,
    TransformerEncoderLayerBase,
)


class TransformerWithRelativePositionalEmbeddingEncoderLayerBase(
    TransformerEncoderLayerBase
):
    """Encoder layer block with optional relative positional embedding."""

    def __init__(self, cfg):
        super().__init__(cfg)

    def build_self_attention(self, embed_dim, cfg):
        relative_pos_embedding_type = None
        max_relative_pos = None
        if cfg.encoder.relative_positional_embeddings:
            if cfg.encoder.learned_pos:
                relative_pos_embedding_type = "learned"
                max_relative_pos = cfg.max_source_positions
            else:
                relative_pos_embedding_type = "sinusoidal"

        return MultiheadAttention(
            embed_dim,
            cfg.encoder.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            relative_pos_embedding_type=relative_pos_embedding_type,
            max_relative_pos=max_relative_pos,
        )


# backward compatible with the legacy argparse format
class TransformerWithRelativePositionalEmbeddingEncoderLayer(
    TransformerWithRelativePositionalEmbeddingEncoderLayerBase
):
    def __init__(self, args):
        super().__init__(TransformerConfig.from_namespace(args))
        self.args = args

    def build_self_attention(self, embed_dim, args):
        return super().build_self_attention(
            embed_dim, TransformerConfig.from_namespace(args)
        )


class TransformerWithRelativePositionalEmbeddingDecoderLayerBase(
    TransformerDecoderLayerBase
):
    """Decoder layer block with optional relative positional embedding."""

    def __init__(
        self, cfg, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(
            cfg,
            no_encoder_attn=no_encoder_attn,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

    def build_self_attention(
        self, embed_dim, cfg, add_bias_kv=False, add_zero_attn=False
    ):
        relative_pos_embedding_type = None
        max_relative_pos = None
        if cfg.decoder.relative_positional_embeddings:
            if cfg.decoder.learned_pos:
                relative_pos_embedding_type = "learned"
                max_relative_pos = cfg.max_target_positions
            else:
                relative_pos_embedding_type = "sinusoidal"

        return MultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            dropout=cfg.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not cfg.cross_self_attention,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            relative_pos_embedding_type=relative_pos_embedding_type,
            max_relative_pos=max_relative_pos,
        )

    def build_encoder_attention(self, embed_dim, cfg):
        return MultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            kdim=cfg.encoder.embed_dim,
            vdim=cfg.encoder.embed_dim,
            dropout=cfg.attention_dropout,
            encoder_decoder_attention=True,
            relaxed_attention_weight=cfg.decoder.relaxed_attention_weight,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

# backward compatible with the legacy argparse format
class TransformerWithRelativePositionalEmbeddingDecoderLayer(
    TransformerWithRelativePositionalEmbeddingDecoderLayerBase
):
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(
            TransformerConfig.from_namespace(args),
            no_encoder_attn=no_encoder_attn,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.args = args

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return super().build_self_attention(
            embed_dim,
            TransformerConfig.from_namespace(args),
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

    def build_encoder_attention(self, embed_dim, args):
        return super().build_encoder_attention(
            embed_dim,
            TransformerConfig.from_namespace(args),
        )
