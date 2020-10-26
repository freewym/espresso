# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch

from fairseq import tokenizer, utils
from fairseq.data import TruncatedDictionary
from fairseq.tasks import register_task
from fairseq.tasks.language_modeling import LanguageModelingTask, LanguageModelingConfig

from espresso.data import AsrDictionary


logger = logging.getLogger(__name__)


@dataclass
class LanguageModelingForASRConfig(LanguageModelingConfig):
    dict: Optional[str] = field(default=None, metadata={"help": "path to the dictionary"})


@register_task("language_modeling_for_asr", dataclass=LanguageModelingForASRConfig)
class LanguageModelingForASRTask(LanguageModelingTask):
    """
    Train a language model.

    Args:
        dictionary (~fairseq.data.AsrDictionary): the dictionary for the input of
            the language model
        output_dictionary (~fairseq.data.AsrDictionary): the dictionary for the
            output of the language model. In most cases it will be the same as
            *dictionary*, but could possibly be a more limited version of the
            dictionary (if ``--output-dictionary-size`` is used).
        targets (List[str]): list of the target types that the language model
            should predict.  Can be one of "self", "future", and "past".
            Defaults to "future".

    .. note::

        The language modeling task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate`, :mod:`fairseq-interactive` and
        :mod:`fairseq-eval-lm`.

    The language modeling task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.language_modeling_for_asr_parser
        :prog:
    """

    def __init__(self, args, dictionary, output_dictionary=None, targets=None):
        super().__init__(args, dictionary, output_dictionary, targets=targets)
        torch.backends.cudnn.deterministic = True
        # Compansate for the removal of :func:`torch.rand()` from
        # :func:`fairseq.distributed_utils.distributed_init()` by fairseq,
        # to make previous experiments reproducible.
        torch.rand(1)

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        return AsrDictionary.load(filename)

    @classmethod
    def build_dictionary(cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        """Build the dictionary

        Args:
            filenames (list): list of filenames
            workers (int): number of concurrent workers
            threshold (int): defines the minimum word count
            nwords (int): defines the total number of words in the final dictionary,
                including special symbols
            padding_factor (int): can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        d = AsrDictionary()
        for filename in filenames:
            AsrDictionary.add_file_to_dictionary(filename, d, tokenizer.tokenize_line, workers)
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    @classmethod
    def setup_dictionary(cls, args, **kwargs):
        dictionary = None
        output_dictionary = None
        if args.data:
            paths = utils.split_paths(args.data)
            assert len(paths) > 0
            dict_path = (
                os.path.join(paths[0], "dict.txt") if args.dict is None
                else args.dict
            )
            dictionary = AsrDictionary.load(dict_path)
            logger.info("dictionary: {} types".format(len(dictionary)))
            output_dictionary = dictionary
            if args.output_dictionary_size >= 0:
                output_dictionary = TruncatedDictionary(
                    dictionary, args.output_dictionary_size
                )
        return (dictionary, output_dictionary)
