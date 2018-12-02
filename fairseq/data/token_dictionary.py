# Copyright (c) 2018-present, Yiming Wang
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from fairseq.data import Dictionary

import torch


class TokenDictionary(Dictionary):
    """A mapping from symbols to consecutive integers"""
    def __init__(self, pad='<pad>', eos='</s>', unk='<unk>', space='<space>'):
        self.unk_word, self.pad_word, self.eos_word, self.space_word = \
            unk, pad, eos, space
        self.symbols = []
        self.count = []
        self.indices = {}
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        self.space_index = self.add_symbol(space)
        self.nspecial = len(self.symbols)

    def string(self, tensor, bpe_symbol=None, escape_unk=False):
        """Helper for converting a tensor of token indices to a string.

        Can optionally remove BPE symbols or escape <unk> words.

        We overwrite this since we would like to also ignore <pad>.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return '\n'.join(self.string(t) for t in tensor)

        def token_string(i):
            if i == self.unk():
                return self.unk_string(escape_unk)
            else:
                return self[i]

        sent = ' '.join(token_string(i) for i in tensor if i != self.eos() and \
            i != self.pad())
        if bpe_symbol is not None:
            sent = (sent + ' ').replace(bpe_symbol, '').rstrip()
        return sent
    
    def space(self):
        """Helper to get index of space symbol"""
        return self.space_index

    def dummy_sentence(self, length):
        # sample starting from space
        t = torch.Tensor(length).uniform_(self.nspecial - 1, len(self)).int()
        t[-1] = self.eos()
        return t
