# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List, Optional

from fairseq.data.encoders import register_bpe
from fairseq.dataclass import FairseqDataclass

from espresso.tools.utils import tokenize


@dataclass
class CharactersAsrConfig(FairseqDataclass):
    pass


@register_bpe("characters_asr", dataclass=CharactersAsrConfig)
class CharactersAsr(object):
    def __init__(
        self, cfg, space_symbol="<space>", ends_with_space=True,
        non_lang_syms: Optional[List[str]] = None,
    ):
        self.space_symbol = space_symbol
        self.ends_with_space = ends_with_space
        self.non_lang_syms = non_lang_syms

    def encode(self, x: str) -> str:
        y = tokenize(x, space=self.space_symbol, non_lang_syms=self.non_lang_syms)
        if self.ends_with_space:
            return y + " " + self.space_symbol
        else:
            return y

    def decode(self, x: str) -> str:
        return x.replace(" ", "").replace(self.space_symbol, " ").strip()
