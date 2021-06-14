# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.modules import MultiheadAttention, TransformerDecoderLayer, TransformerEncoderLayer


class TransformerWithRelativePositionalEmbeddingEncoderLayer(TransformerEncoderLayer):
    """Encoder layer block with optional relative positional embedding.
    """

    def __init__(self, args):
        super().__init__(args)

    def build_self_attention(self, embed_dim, args):
        relative_pos_embedding_type = None
        max_relative_pos = None
        if args.encoder_relative_positional_embeddings:
            if args.encoder_learned_pos:
                relative_pos_embedding_type = "learned"
                max_relative_pos = args.max_source_positions
            else:
                relative_pos_embedding_type = "sinusoidal"

        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            relative_pos_embedding_type=relative_pos_embedding_type,
            max_relative_pos=max_relative_pos,
        )


class TransformerWithRelativePositionalEmbeddingDecoderLayer(TransformerDecoderLayer):
    """Decoder layer block with optional relative positional embedding.
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(args, no_encoder_attn=no_encoder_attn, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        relative_pos_embedding_type = None
        max_relative_pos = None
        if args.decoder_relative_positional_embeddings:
            if args.decoder_learned_pos:
                relative_pos_embedding_type = "learned"
                max_relative_pos = args.max_target_positions
            else:
                relative_pos_embedding_type = "sinusoidal"

        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            relative_pos_embedding_type=relative_pos_embedding_type,
            max_relative_pos=max_relative_pos,
        )
