# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq.data import Dictionary, encoders
from fairseq.file_io import PathManager

# will automatically load modules defined from there
from espresso.data import encoders as encoders_espresso


class AsrDictionary(Dictionary):
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        bos="<s>",
        pad="<pad>",
        eos="</s>",
        unk="<unk>",
        space="<space>",
        extra_special_symbols=None,
    ):
        self.unk_word, self.bos_word, self.pad_word, self.eos_word, self.space_word = \
            unk, bos, pad, eos, space
        self.symbols = []
        self.count = []
        self.indices = {}
        # no bos added to the dictionary
        self.pad_index = self.add_symbol(pad, n=0)
        self.eos_index = self.add_symbol(eos, n=0)
        self.unk_index = self.add_symbol(unk, n=0)
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s, n=0)
        self.nspecial = len(self.symbols)
        self.non_lang_syms = None
        self.tokenizer = None
        self.bpe = None

    def bos(self):
        """Disallow beginning-of-sentence symbol"""
        raise NotImplementedError

    def space(self):
        """Helper to get index of space symbol"""
        return self.space_index

    @classmethod
    def load(cls, f, f_non_lang_syms=None):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```

        Identifies the space symbol if it exists, by obtaining its index
        (space_index=-1 if no space symbol)

        Loads non_lang_syms from another text file, if it exists, with one
        symbol per line
        """
        d = super().load(f)
        d.space_index = d.indices.get(d.space_word, -1)

        if f_non_lang_syms is not None:
            assert isinstance(f_non_lang_syms, str)
            try:
                with PathManager.open(f_non_lang_syms, "r", encoding="utf-8") as fd:
                    non_lang_syms = [x.rstrip() for x in fd.readlines()]
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(
                    "Incorrect encoding detected in {}, please "
                    "rebuild the dataset".format(f)
                )

            for sym in non_lang_syms:
                assert d.index(sym) != d.unk(), \
                    "{} in {} is not in the dictionary".format(sym, f_non_lang_syms)
            d.non_lang_syms = non_lang_syms

        return d

    def dummy_sentence(self, length):
        # sample excluding special symbols
        t = torch.Tensor(length).uniform_(self.nspecial, len(self)).long()
        t[-1] = self.eos()
        return t

    def build_tokenizer(self, args):
        self.tokenizer = encoders.build_tokenizer(args)

    def build_bpe(self, args):
        if args.bpe == "characters_asr":
            self.bpe = encoders.build_bpe(
                args, space_symbol=self.space_word, ends_with_space=True,
                non_lang_syms=self.non_lang_syms,
            )
        else:
            self.bpe = encoders.build_bpe(args)

    def wordpiece_encode(self, x):
        if self.tokenizer is not None:
            x = self.tokenizer.encode(x)
        if self.bpe is not None:
            x = self.bpe.encode(x)
        return x

    def wordpiece_decode(self, x):
        if self.bpe is not None:
            x = self.bpe.decode(x)
        if self.tokenizer is not None:
            x = self.tokenizer.decode(x)
        return x
