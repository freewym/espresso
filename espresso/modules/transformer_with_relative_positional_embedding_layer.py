# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from espresso.modules import RelativePositionalEmbedding
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

    def __init__(
        self, cfg, positional_embedding: Optional[RelativePositionalEmbedding] = None
    ):
        # a workaround to avoid being registered within this class
        # `positional_embedding` will be registered in :class:`~MultiheadAttention`
        self.positional_embedding = [positional_embedding]
        super().__init__(cfg)

    def build_self_attention(self, embed_dim, cfg):
        return MultiheadAttention(
            embed_dim,
            cfg.encoder.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            positional_embedding=self.positional_embedding[0],
        )


# backward compatible with the legacy argparse format
class TransformerWithRelativePositionalEmbeddingEncoderLayer(
    TransformerWithRelativePositionalEmbeddingEncoderLayerBase
):
    def __init__(
        self, args, positional_embedding: Optional[RelativePositionalEmbedding] = None
    ):
        super().__init__(
            TransformerConfig.from_namespace(args),
            positional_embedding=positional_embedding,
        )
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
        self,
        cfg,
        no_encoder_attn=False,
        add_bias_kv=False,
        add_zero_attn=False,
        positional_embedding: Optional[RelativePositionalEmbedding] = None,
    ):
        # a workaround to avoid being registered within this class.
        # `positional_embedding` will be registered in :class:`~MultiheadAttention`
        self.positional_embedding = [positional_embedding]
        super().__init__(
            cfg,
            no_encoder_attn=no_encoder_attn,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

    def build_self_attention(
        self, embed_dim, cfg, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            dropout=cfg.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not cfg.cross_self_attention,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            positional_embedding=self.positional_embedding[0],
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
        self,
        args,
        no_encoder_attn=False,
        add_bias_kv=False,
        add_zero_attn=False,
        positional_embedding: Optional[RelativePositionalEmbedding] = None,
    ):
        super().__init__(
            TransformerConfig.from_namespace(args),
            no_encoder_attn=no_encoder_attn,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            positional_embedding=positional_embedding,
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
