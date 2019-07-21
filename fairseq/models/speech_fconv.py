# Copyright (c) 2019-present, Yiming Wang
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.modules import GradMultiply

from .speech_lstm import ConvBNReLU

from .fconv import (
    ConvTBC,
    FConvModel,
    FConvEncoder,
    FConvDecoder,
    Linear,
    extend_conv_spec,
)

import speech_tools.utils as speech_utils


@register_model('speech_fconv')
class SpeechFConvModel(FConvModel):
    """
    A fully convolutional model, i.e. a convolutional encoder and a
    convolutional decoder, as described in `"Convolutional Sequence to Sequence
    Learning" (Gehring et al., 2017) <https://arxiv.org/abs/1705.03122>`_.

    Args:
        encoder (FConvEncoder): the encoder
        decoder (FConvDecoder): the decoder

    The Convolutional model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.fconv_parser
        :prog:
    """

    @classmethod
    def hub_models(cls):
        raise NotImplementedError

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        FConvModel.add_args(parser)
        parser.add_argument('--encoder-conv-channels', type=str, metavar='EXPR',
                            help='list of encoder convolution\'s out channels')
        parser.add_argument('--encoder-conv-kernel-sizes', type=str, metavar='EXPR',
                            help='list of encoder convolution\'s kernel sizes')
        parser.add_argument('--encoder-conv-strides', type=str, metavar='EXPR',
                            help='list of encoder convolution\'s strides')
        parser.add_argument('--decoder-positional-embed', action='store_true',
                            help='use decoder positional embeddings')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        decoder_embed_dict = None
        if args.decoder_embed_path:
            decoder_embed_dict = utils.parse_embedding(args.decoder_embed_path)
            utils.print_embed_overlap(decoder_embed_dict, task.target_dictionary)

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

        fconv_encoder_input_size = task.feat_dim // task.feat_in_channels
        if conv_layers is not None:
            for stride in strides:
                if isinstance(stride, (list, tuple)):
                    assert len(stride) > 0
                    s = stride[1] if len(stride) > 1 else stride[0]
                else:
                    assert isinstance(stride, int)
                    s = stride
                fconv_encoder_input_size = (fconv_encoder_input_size + s - 1) // s
            fconv_encoder_input_size *= out_channels[-1]
        
        encoder = SpeechFConvEncoder(
            conv_layers_before=conv_layers,
            input_size=fconv_encoder_input_size,
            embed_dim=args.encoder_embed_dim,
            convolutions=eval(args.encoder_layers),
            dropout=args.dropout,
        )
        decoder = SpeechFConvDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            embed_dict=decoder_embed_dict,
            convolutions=eval(args.decoder_layers),
            out_embed_dim=args.decoder_out_embed_dim,
            attention=eval(args.decoder_attention),
            dropout=args.dropout,
            max_positions=args.max_target_positions,
            share_embed=args.share_input_output_embed,
            positional_embeddings=args.decoder_positional_embed,
        )
        return SpeechFConvModel(encoder, decoder)


class SpeechFConvEncoder(FConvEncoder):
    """
    Convolutional encoder consisting of `len(convolutions)` layers.

    Args:
        conv_layers_before (~fairseq.speech_lstm.ConvBNReLU): convolutions befoe
            fconv layers
        input_size (int, optional): dimension of the input to the transformer
            before being projected to embed_dim
        embed_dim (int, optional): embedding dimension
        max_positions (int, optional): maximum supported input sequence length
        convolutions (list, optional): the convolutional layer structure. Each
            list item `i` corresponds to convolutional layer `i`. Layers are
            given as ``(out_channels, kernel_width, [residual])``. Residual
            connections are added between layers when ``residual=1`` (which is
            the default behavior).
        dropout (float, optional): dropout to be applied before each conv layer
    """

    def __init__(
        self, conv_layers_before=None, input_size=83, embed_dim=512,
        convolutions=((512, 3),) * 20, dropout=0.1,
    ):
        super(FConvEncoder, self).__init__(None)  # no src dictionary
        self.dropout = dropout
        self.num_attention_layers = None

        self.conv_layers_before = conv_layers_before
        self.fc0 = Linear(input_size, embed_dim, dropout=dropout) \
            if input_size != embed_dim else None
        
        convolutions = extend_conv_spec(convolutions)
        in_channels = convolutions[0][0]
        self.fc1 = Linear(embed_dim, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.residuals = []

        layer_in_channels = [in_channels]
        for _, (out_channels, kernel_size, residual) in enumerate(convolutions):
            if residual == 0:
                residual_dim = out_channels
            else:
                residual_dim = layer_in_channels[-residual]
            self.projections.append(Linear(residual_dim, out_channels)
                                    if residual_dim != out_channels else None)
            if kernel_size % 2 == 1:
                padding = kernel_size // 2
            else:
                padding = 0
            self.convolutions.append(
                ConvTBC(in_channels, out_channels * 2, kernel_size,
                        dropout=dropout, padding=padding)
            )
            self.residuals.append(residual)
            in_channels = out_channels
            layer_in_channels.append(out_channels)
        self.fc2 = Linear(in_channels, embed_dim)

    def output_lengths(self, in_lengths):
        return in_lengths if self.conv_layers_before is None \
            else self.conv_layers_before.output_lengths(in_lengths)

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of shape
                `(batch)`

        Returns:
            dict:
                - **encoder_out** (tuple): a tuple with two elements, where the
                  first element is the last encoder layer's output and the
                  second element is the same quantity summed with the input
                  embedding (used for attention). The shape of both tensors is
                  `(batch, src_len, embed_dim)`.
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        if self.conv_layers_before is not None:
            x, src_lengths, encoder_padding_mask = self.conv_layers_before(src_tokens,
                src_lengths)
        else:
            x, encoder_padding_mask = src_tokens, \
                ~speech_utils.sequence_mask(src_lengths, src_tokens.size(1))

        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.fc0 is not None:
            x = self.fc0(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        input_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        encoder_padding_mask = encoder_padding_mask.t()  # -> T x B
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        residuals = [x]
        # temporal convolutions
        for proj, conv, res_layer in zip(self.projections, self.convolutions, self.residuals):
            if res_layer > 0:
                residual = residuals[-res_layer]
                residual = residual if proj is None else proj(residual)
            else:
                residual = None

            if encoder_padding_mask is not None:
                x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)

            x = F.dropout(x, p=self.dropout, training=self.training)
            if conv.kernel_size[0] % 2 == 1:
                # padding is implicit in the conv
                x = conv(x)
            else:
                padding_l = (conv.kernel_size[0] - 1) // 2
                padding_r = conv.kernel_size[0] // 2
                x = F.pad(x, (0, 0, 0, 0, padding_l, padding_r))
                x = conv(x)
            x = F.glu(x, dim=2)

            if residual is not None:
                x = (x + residual) * math.sqrt(0.5)
            residuals.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # project back to size of embedding
        x = self.fc2(x)

        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.t()  # -> B x T
            x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)

        # scale gradients (this only affects backward, not forward)
        x = GradMultiply.apply(x, 1.0 / (2.0 * self.num_attention_layers))

        # add output to input embedding for attention
        y = (x + input_embedding) * math.sqrt(0.5)

        return {
            'encoder_out': (x, y),
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)


class SpeechFConvDecoder(FConvDecoder):
    def masked_copy_incremental_state(self, incremental_state, another_state, mask):
        state = utils.get_incremental_state(self, incremental_state, 'encoder_out')
        if state is None:
            assert another_state is None
            return

        def mask_copy_state(state, another_state):
            if isinstance(state, list):
                assert isinstance(another_state, list) and len(state) == len(another_state)
                return [mask_copy_state(state_i, another_state_i) \
                    for state_i, another_state_i in zip(state, another_state)]
            if state is not None:
                assert state.size(0) == mask.size(0) and another_state is not None and \
                    state.size() == another_state.size()
                for _ in range(1, len(state.size())):
                    mask_unsqueezed = mask.unsqueeze(-1)
                return torch.where(mask_unsqueezed, state, another_state)
            else:
                assert another_state is None
                return None

        new_state = tuple(map(mask_copy_state, state, another_state))
        utils.set_incremental_state(self, incremental_state, 'encoder_out', new_state)


@register_model_architecture('speech_fconv', 'speech_fconv')
def base_architecture(args):
    args.encoder_conv_channels = getattr(args, 'encoder_conv_channels',
        '[64, 64, 128, 128]')
    args.encoder_conv_kernel_sizes = getattr(args, 'encoder_conv_kernel_sizes',
        '[(3, 3), (3, 3), (3, 3), (3, 3)]')
    args.encoder_conv_strides = getattr(args, 'encoder_conv_strides',
        '[(1, 1), (2, 2), (1, 1), (2, 2)]')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', '[(512, 3)] * 20')
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(512, 3)] * 20')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.decoder_attention = getattr(args, 'decoder_attention', 'True')
    args.share_input_output_embed = getattr(args, 'share_input_output_embed', False)
    args.decoder_positional_embed = getattr(args, 'decoder_positional_embed', False)


@register_model_architecture('speech_fconv', 'speech_fconv_librispeech')
def speech_fconv_librispeech(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_layers = getattr(args, 'encoder_layers', '[(256, 3)] * 4')
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(256, 3)] * 3')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    base_architecture(args)
