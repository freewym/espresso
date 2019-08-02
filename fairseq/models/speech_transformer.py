# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.modules import LayerNorm

from .speech_lstm import ConvBNReLU

from .transformer import (
    Embedding,
    Linear,
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder,
    TransformerEncoderLayer,
)

import speech_tools.utils as speech_utils

DEFAULT_MAX_SOURCE_POSITIONS = 9999
DEFAULT_MAX_TARGET_POSITIONS = 999


@register_model('speech_transformer')
class SpeechTransformerModel(TransformerModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    @classmethod
    def hub_models(cls):
        raise NotImplementedError

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        TransformerModel.add_args(parser)
        parser.add_argument('--encoder-conv-channels', type=str, metavar='EXPR',
                            help='list of encoder convolution\'s out channels')
        parser.add_argument('--encoder-conv-kernel-sizes', type=str, metavar='EXPR',
                            help='list of encoder convolution\'s kernel sizes')
        parser.add_argument('--encoder-conv-strides', type=str, metavar='EXPR',
                            help='list of encoder convolution\'s strides')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        dict = task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        decoder_embed_tokens = build_embedding(
            dict, args.decoder_embed_dim, args.decoder_embed_path
        )

        def eval_str_nested_list_or_tuple(x, type=int):
            if x is None:
                return None
            if isinstance(x, str):
                x = eval(x)
            if isinstance(x, list):
                return list(
                    map(lambda s: eval_str_nested_list_or_tuple(s, type), x))
            elif isinstance(x, tuple):
                return tuple(
                    map(lambda s: eval_str_nested_list_or_tuple(s, type), x))
            else:
                try:
                    return type(x)
                except:
                    raise ValueError

        out_channels = eval_str_nested_list_or_tuple(args.encoder_conv_channels,
            type=int)
        kernel_sizes = eval_str_nested_list_or_tuple(
            args.encoder_conv_kernel_sizes, type=int)
        strides = eval_str_nested_list_or_tuple(args.encoder_conv_strides,
            type=int)
        print('| input feature dimension: {}, channels: {}'.format(task.feat_dim,
            task.feat_in_channels))
        assert task.feat_dim % task.feat_in_channels == 0
        conv_layers = ConvBNReLU(out_channels, kernel_sizes, strides,
            in_channels=task.feat_in_channels) if not out_channels is None else None

        transformer_encoder_input_size = task.feat_dim // task.feat_in_channels
        if conv_layers is not None:
            for stride in strides:
                if isinstance(stride, (list, tuple)):
                    assert len(stride) > 0
                    s = stride[1] if len(stride) > 1 else stride[0]
                else:
                    assert isinstance(stride, int)
                    s = stride
                transformer_encoder_input_size = \
                    (transformer_encoder_input_size + s - 1) // s
            transformer_encoder_input_size *= out_channels[-1]
        
        encoder = cls.build_encoder(args, conv_layers_before=conv_layers,
            input_size=transformer_encoder_input_size)
        decoder = cls.build_decoder(args, dict, decoder_embed_tokens)
        return SpeechTransformerModel(encoder, decoder)

    @classmethod
    def build_encoder(cls, args, conv_layers_before=None, input_size=83):
        return SpeechTransformerEncoder(args,
            conv_layers_before=conv_layers_before, input_size=input_size)

    @classmethod
    def build_decoder(cls, args, dict, embed_tokens):
        return SpeechTransformerDecoder(args, dict, embed_tokens)


class SpeechTransformerEncoder(TransformerEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        conv_layers_before (~fairseq.speech_lstm.ConvBNReLU): convolutions before
            transformer layers
        input_size (int, optional): dimension of the input to the transformer
            before being projected to args.encoder_embed_dim
    """

    def __init__(self, args, conv_layers_before=None, input_size=83):
        super(TransformerEncoder, self).__init__(None)  # no src dictionary
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout

        self.conv_layers_before = conv_layers_before
        self.fc0 = Linear(input_size, args.encoder_embed_dim) \
            if input_size != args.encoder_embed_dim else None
        self.max_source_positions = args.max_source_positions

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

    def output_lengths(self, in_lengths):
        return in_lengths if self.conv_layers_before is None \
            else self.conv_layers_before.output_lengths(in_lengths)

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        if self.conv_layers_before is not None:
            x, src_lengths, encoder_padding_mask = self.conv_layers_before(src_tokens,
                src_lengths)
        else:
            x, encoder_padding_mask = src_tokens, \
                ~speech_utils.sequence_mask(src_lengths, src_tokens.size(1))

        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.fc0 is not None:
            x = self.fc0(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.layer_norm:
            x = self.layer_norm(x)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        for i in range(len(self.layers)):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(state_dict, "{}.layers.{}".format(name, i))

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class SpeechTransformerDecoder(TransformerDecoder):
    def masked_copy_incremental_state(self, incremental_state, another_cached_state, mask):
        pass

@register_model_architecture('speech_transformer', 'speech_transformer')
def base_architecture(args):
    args.encoder_conv_channels = getattr(args, 'encoder_conv_channels',
        '[64, 64, 128, 128]')
    args.encoder_conv_kernel_sizes = getattr(args, 'encoder_conv_kernel_sizes',
        '[(3, 3), (3, 3), (3, 3), (3, 3)]')
    args.encoder_conv_strides = getattr(args, 'encoder_conv_strides',
        '[(1, 1), (2, 2), (1, 1), (2, 2)]')
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)


@register_model_architecture('speech_transformer', 'speech_transformer_librispeech')
def speech_transformer_librispeech(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 1)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 1)
    args.dropout = getattr(args, 'dropout', 0.3)
    base_architecture(args)
