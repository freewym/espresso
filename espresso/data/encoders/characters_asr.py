# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional

from fairseq.data.encoders import register_bpe

from espresso.tools.utils import tokenize


@register_bpe('characters_asr')
class CharactersAsr(object):

    @staticmethod
    def add_args(parser):
        pass

    def __init__(
        self, args, space_symbol="<space>", ends_with_space=True,
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
