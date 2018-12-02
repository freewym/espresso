# Copyright (c) 2018-present, Yiming Wang
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import re

import torch


class Tokenizer:

    @staticmethod
    def tokenize(sent, space='<space>', non_lang_syms=None):
        sent = ' '.join(sent.strip().split())

        match_pos = []
        if non_lang_syms is not None:
            assert isinstance(non_lang_syms, list)
            prog = re.compile('|'.join(map(re.escape, non_lang_syms)))
            matches = prog.finditer(sent)
            for match in matches:
                match_pos.append([match.start(), match.end()])

        tokens = []
        i = 0
        for (start_pos, end_pos) in match_pos:
            tokens.extend([token for token in sent[i:start_pos]])
            tokens.append(sent[start_pos:end_pos])
            i = end_pos
        tokens.extend([token for token in sent[i:]])

        tokens = [space if token == ' ' else token for token in tokens]
        return ' '.join(tokens)
    
    @staticmethod
    def tokens_to_index_tensor(line, dict, append_eos=True):
        tokens = line.strip().split()
        ntokens = len(tokens)
        ids = torch.IntTensor(ntokens + 1 if append_eos else ntokens)

        for i, token in enumerate(tokens):
            ids[i] = dict.index(token)
        if append_eos:
            ids[ntokens] = dict.eos_index
        return ids

    @staticmethod
    def tokens_to_sentence(line, dict):
        tokens = line.strip().split()
        sent = ""
        for token in tokens:
            if token == dict.space_word:
                sent += " "
            elif dict.index(token) == dict.unk():
                sent += dict.unk_word
            elif token != dict.pad_word and token != dict.eos_word:
                sent += token
        return sent.strip()

def collate_frames(values, pad_value=0.0, left_pad=False):
    """Convert a list of 2d tensor into a padded 3d tensor."""
    assert values[0].dim() == 2, "expected 2, got " + str(values[0].dim)
    length = max(v.size(0) for v in values)
    dim = values[0].size(1)
    res = values[0].new(len(values), length, dim).fill_(pad_value)

    for i, v in enumerate(values):
        dst = res[i][length - v.size(0):, :] if left_pad \
            else res[i][:v.size(0), :]
        assert dst.numel() == v.numel()
        dst.copy_(v)
    return res
