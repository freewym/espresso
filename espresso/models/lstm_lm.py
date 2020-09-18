# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from omegaconf import II
from typing import Optional

from fairseq import utils
from fairseq.dataclass.utils import FairseqDataclass, gen_parser_from_dataclass
from fairseq.models import (
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.lstm import Embedding

from espresso.models.speech_lstm import SpeechLSTMDecoder
from espresso.tasks.speech_recognition import SpeechRecognitionEspressoTask


DEFAULT_MAX_TARGET_POSITIONS = 1e5


@dataclass
class LSTMLanguageModelEspressoConfig(FairseqDataclass):
    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    decoder_embed_dim: int = field(
        default=48, metadata={"help": "decoder embedding dimension"}
    )
    decoder_embed_path: Optional[str] = field(
        default=None, metadata={"help": "path to pre-trained decoder embedding"}
    )
    decoder_freeze_embed: bool = field(
        default=False, metadata={"help": "freeze decoder embeddings"}
    )
    decoder_hidden_size: int = field(
        default=650, metadata={"help": "decoder hidden size"}
    )
    decoder_layers: int = field(
        default=2, metadata={"help": "number of decoder layers"}
    )
    decoder_out_embed_dim: int = field(
        default=650, metadata={"help": "decoder output embedding dimension"}
    )
    decoder_rnn_residual: lambda x: utils.eval_bool(x) = field(
        default=False,
        metadata={
            "help": "create residual connections for rnn decoder layers "
            "(starting from the 2nd layer), i.e., the actual output of such "
            "layer is the sum of its input and output"
        },
    )
    adaptive_softmax_cutoff: Optional[str] = field(
        default=None,
        metadata={
            "help": "comma separated list of adaptive softmax cutoff points. "
            "Must be used with adaptive_loss criterion"
        },
    )
    share_embed: lambda x: utils.eval_bool(x) = field(
        default=False, metadata={"help": "share input and output embeddings"}
    )
    is_wordlm: bool = field(
        default=False,
        metadata={
            "help": "whether it is word LM or subword LM. Only relevant for ASR decoding "
            "with LM, and it determines how the underlying decoder instance gets the "
            "dictionary from the task instance when calling cls.build_model()"
        },
    )
    # Granular dropout settings (if not specified these default to --dropout)
    decoder_dropout_in: float = field(
        default=0.1,
        metadata={"help": "dropout probability for decoder input embedding"}
    )
    decoder_dropout_out: float = field(
        default=0.1,
        metadata={"help": "dropout probability for decoder output"}
    )
    # TODO common var add to parent
    add_bos_token: bool = II("task.add_bos_token")
    tokens_per_sample: int = II("task.tokens_per_sample")
    max_target_positions: Optional[int] = II("task.max_target_positions")
    # TODO common var add to parent
    tpu: bool = II("params.common.tpu")


@register_model("lstm_lm_espresso")
class LSTMLanguageModelEspresso(FairseqLanguageModel):
    def __init__(self, decoder, args):
        super().__init__(decoder)
        self.is_wordlm = args.is_wordlm

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--dropout", type=float, metavar="D",
                            help="dropout probability")
        parser.add_argument("--decoder-embed-dim", type=int, metavar="N",
                            help="decoder embedding dimension")
        parser.add_argument("--decoder-embed-path", type=str, metavar="STR",
                            help="path to pre-trained decoder embedding")
        parser.add_argument("--decoder-freeze-embed", action="store_true",
                            help="freeze decoder embeddings")
        parser.add_argument("--decoder-hidden-size", type=int, metavar="N",
                            help="decoder hidden size")
        parser.add_argument("--decoder-layers", type=int, metavar="N",
                            help="number of decoder layers")
        parser.add_argument("--decoder-out-embed-dim", type=int, metavar="N",
                            help="decoder output embedding dimension")
        parser.add_argument("--decoder-rnn-residual",
                            type=lambda x: utils.eval_bool(x),
                            help="create residual connections for rnn decoder "
                            "layers (starting from the 2nd layer), i.e., the actual "
                            "output of such layer is the sum of its input and output")
        parser.add_argument("--adaptive-softmax-cutoff", metavar="EXPR",
                            help="comma separated list of adaptive softmax cutoff points. "
                                 "Must be used with adaptive_loss criterion")
        parser.add_argument("--share-embed",
                            type=lambda x: utils.eval_bool(x),
                            help="share input and output embeddings")
        parser.add_argument("--is-wordlm", action="store_true",
                            help="whether it is word LM or subword LM. Only "
                            "relevant for ASR decoding with LM, and it determines "
                            "how the underlying decoder instance gets the dictionary "
                            "from the task instance when calling cls.build_model()")

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument("--decoder-dropout-in", type=float, metavar="D",
                            help="dropout probability for decoder input embedding")
        parser.add_argument("--decoder-dropout-out", type=float, metavar="D",
                            help="dropout probability for decoder output")
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_lm_architecture(args)

        if getattr(args, "max_target_positions", None) is not None:
            max_target_positions = args.max_target_positions
        else:
            max_target_positions = getattr(args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS)

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        if args.is_wordlm and hasattr(task, "word_dictionary"):
            dictionary = task.word_dictionary
        elif isinstance(task, SpeechRecognitionEspressoTask):
            dictionary = task.target_dictionary
        else:
            dictionary = task.source_dictionary

        # separate decoder input embeddings
        pretrained_decoder_embed = None
        if args.decoder_embed_path:
            pretrained_decoder_embed = load_pretrained_embedding_from_file(
                args.decoder_embed_path,
                dictionary,
                args.decoder_embed_dim
            )
        # one last double check of parameter combinations
        if args.share_embed and (
                args.decoder_embed_dim != args.decoder_out_embed_dim):
            raise ValueError(
                "--share-embed requires "
                "--decoder-embed-dim to match --decoder-out-embed-dim"
            )

        if args.decoder_freeze_embed:
            pretrained_decoder_embed.weight.requires_grad = False

        decoder = SpeechLSTMDecoder(
            dictionary=dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attn_type=None,
            encoder_output_units=0,
            pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args.share_embed,
            adaptive_softmax_cutoff=(
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == "adaptive_loss" else None
            ),
            max_target_positions=max_target_positions,
        )
        return cls(decoder, args)


@register_model_architecture("lstm_lm_espresso", "lstm_lm_espresso")
def base_lm_architecture(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 48)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_freeze_embed = getattr(args, "decoder_freeze_embed", False)
    args.decoder_hidden_size = getattr(args, "decoder_hidden_size", 650)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 650)
    args.decoder_rnn_residual = getattr(args, "decoder_rnn_residual", False)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.decoder_dropout_in = getattr(args, "decoder_dropout_in", args.dropout)
    args.decoder_dropout_out = getattr(args, "decoder_dropout_out", args.dropout)
    args.share_embed = getattr(args, "share_embed", False)
    args.is_wordlm = getattr(args, "is_wordlm", False)


@register_model_architecture("lstm_lm_espresso", "lstm_lm_wsj")
def lstm_lm_wsj(args):
    base_lm_architecture(args)


@register_model_architecture("lstm_lm_espresso", "lstm_lm_librispeech")
def lstm_lm_librispeech(args):
    args.dropout = getattr(args, "dropout", 0.0)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 800)
    args.decoder_hidden_size = getattr(args, "decoder_hidden_size", 800)
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 800)
    args.share_embed = getattr(args, "share_embed", True)
    base_lm_architecture(args)


@register_model_architecture("lstm_lm_espresso", "lstm_lm_swbd")
def lstm_lm_swbd(args):
    args.dropout = getattr(args, "dropout", 0.3)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1800)
    args.decoder_hidden_size = getattr(args, "decoder_hidden_size", 1800)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 1800)
    args.share_embed = getattr(args, "share_embed", True)
    base_lm_architecture(args)


@register_model_architecture("lstm_lm_espresso", "lstm_wordlm_wsj")
def lstm_wordlm_wsj(args):
    args.dropout = getattr(args, "dropout", 0.35)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1200)
    args.decoder_hidden_size = getattr(args, "decoder_hidden_size", 1200)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 1200)
    args.share_embed = getattr(args, "share_embed", True)
    args.is_wordlm = True
    base_lm_architecture(args)
