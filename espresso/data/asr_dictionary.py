# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq.data import Dictionary, data_utils
from fairseq.file_io import PathManager
from fairseq.tokenizer import tokenize_line


class AsrDictionary(Dictionary):
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        pad="<pad>",
        eos="</s>",
        unk="<unk>",
        bos="<s>",
        space="<space>",
        extra_special_symbols=None,
    ):
        self.unk_word, self.pad_word, self.eos_word, self.bos_word, self.space_word = \
            unk, pad, eos, bos, space
        self.symbols = []
        self.count = []
        self.indices = {}
        self.pad_index = self.add_symbol(pad, n=0)
        self.eos_index = self.add_symbol(eos, n=0)
        self.unk_index = self.add_symbol(unk, n=0)
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s, n=0)
        self.nspecial = len(self.symbols)
        self.non_lang_syms = None

    def string(self, tensor, bpe_symbol=None, escape_unk=False):
        """Helper for converting a tensor of token indices to a string.

        Can optionally remove BPE symbols or escape <unk> words.

        We overwrite this since we would like to also ignore <pad>.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return "\n".join(self.string(t, bpe_symbol, escape_unk) for t in tensor)

        def token_string(i):
            if i == self.unk():
                return self.unk_string(escape_unk)
            else:
                return self[i]

        if hasattr(self, "bos_index"):
            sent = " ".join(
                token_string(i)
                for i in tensor
                if (i != self.eos()) and (i != self.bos()) and (i != self.pad())
            )
        else:
            sent = " ".join(
                token_string(i)
                for i in tensor
                if (i != self.eos()) and (i != self.pad())
            )
        return data_utils.process_bpe_symbol(sent, bpe_symbol)

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

    def tokens_to_sentence(
        self, line, line_tokenizer=tokenize_line, use_unk_sym=True, bpe_symbol=None,
    ):
        if bpe_symbol is not None:
            return data_utils.process_bpe_symbol(line, bpe_symbol)
        # use_unk_sym=False when we want to restore original transcripts from
        # token sequences, e.g., obtain reference to compute WER
        tokens = line_tokenizer(line)
        sent = ""
        for token in tokens:
            if token == self.space_word:
                sent += " "
            elif use_unk_sym and self.index(token) == self.unk_index:
                sent += self.unk_word
            elif token != self.pad_word and token != self.eos_word:
                sent += token
        return sent.strip()
