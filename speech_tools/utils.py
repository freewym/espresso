# Copyright (c) 2018-present, Yiming Wang
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import re
import numpy as np
from collections import Counter

import torch

from fairseq.utils import buffered_arange


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

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    else:
        assert utils.item(sequence_length.data.max()) <= utils.item(max_len)
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long().to(device=sequence_length.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
    return seq_range_expand < seq_length_expand

def covnert_padding_direction(src_frames, src_lengths, right_to_left=False,
    left_to_right=False):
    """Counterpart of :func:`~fairseq.utils.convert_padding_direction`,
    operating on 3d tensors of size B x T x C.
    """
    assert right_to_left ^ left_to_right
    assert src_frames.size(0) == src_lengths.size(0)
    pad_mask = sequence_mask(src_lengths, max_len=src_frames.size(1))
    if not pad_mask.any():
        # no padding, return early
        return src_frames
    if left_to_right and not pad_mask[:, 0].any():
        # already right padded
        return src_frames
    if right_to_left and not pad_mask[:, -1].any():
        # already left padded
        return src_frames
    max_len = src_frames.size(1)
    range = buffered_arange(max_len).type_as(src_frames).expand_as(src_frames)
    num_pads = pad_mask.long().sum(dim=1, keepdim=True)
    if right_to_left:
        index = torch.remainder(range - num_pads, max_len)
    else:
        index = torch.remainder(range + num_pads, max_len)
    return src_frames.gather(1, index)

def edit_distance(ref, hyp):
    """This function is to calculate the edit distance of reference sentence and
    the hypothesis sentence using dynamic programming, and also backtrace to get
    a list of edit steps.

    Args:
        ref: list of words obtained by splitting reference sentence string
        hyp: list of words obtained by splitting hypothesis sentence string

    Return:
        dist: edit distance matrix of size len(ref) x len(hyp)
        steps: list of edit steps
        counter: object of collections.Counter containing counts of
            reference words ('words'), number of correct words ('corr'),
            substitutions ('sub'), insertions ('ins'), deletions ('del').


    """

    assert isinstance(ref, list) and isinstance(hyp, list)

    dist = numpy.zeros((len(ref) + 1, len(hyp) + 1), dtype=numpy.uint32)
    for i in range(len(ref) + 1):
        for j in range(len(hyp) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                dist[i][j] = dist[i - 1][j - 1]
            else:
                substitute = dist[i - 1][j - 1] + 1
                insert = dist[i][j - 1] + 1
                delete = dist[i - 1][j] + 1
                dist[i][j] = min(substitute, insert, delete)

    i = len(ref)
    j = len(hyp)
    steps = []
    while True:
        if i == 0 and j == 0:
            break
        elif i >= 1 and j >= 1 and dist[i][j] == dist[i - 1][j - 1]
            assert ref[i - 1] == hyp[j - 1]
            steps.append('corr')
            i = i - 1
            j = j - 1
        elif i >= 1 and j >= 1 and dist[i][j] == dist[i - 1][j - 1] + 1:
            steps.append('sub')
            i = i - 1
            j = j - 1
        elif j >= 1 and dist[i][j] == dist[i][j - 1] + 1:
            steps.append('ins')
            j = j - 1
        else:
            assert i >= 1 and dist[i][j] == dist[i - 1][j] + 1
            steps.append('del')
            i = i - 1
    steps = steps[::-1]

    counter = Counter({'words', len(ref)})
    counter.update(steps)

    return dist, steps, counter

def aligned_print(ref, hyp, steps):
    """This funcition is to print the result of comparing reference and
    hypothesis sentences in an aligned way.

    Args:
        ref: list of words obtained by splitting reference sentence string
        hyp: list of words obtained by splitting hypothesis sentence string
        steps: list of edit steps with elements 'corr', 'sub', 'ins' or 'del'.
    """

    assert isinstance(ref, list) and isinstance(hyp, list)
    assert isinstance(steps, list)

    out_str = 'REF: '
    for i in range(len(steps)):
        delim = ' ' if i < len(steps) - 1 else '\n'
        if steps[i] == 'sub':
            ref_idx =  i - steps[:i].count('ins')
            hyp_idx = i - steps[:i].count('del')
            if len(ref[ref_idx]) < len(hyp[hyp_idx]):
                out_str += ref[ref_idx] +
                    ' ' * (len(hyp[hyp_idx])-len(ref[ref_idx])) + delim
            else:
                out_str += ref[ref_idx] + delim
        elif steps[i] == 'ins':
            idx = i - steps[:i].count('del')
            out_str += ' ' * len(hyp[idx] + delim
        else:
            assert steps[i] == 'del' or steps[i] == 'corr'
            idx = i - steps[:i].count('ins')
            out_str += ref[idx] + delim

    out_str += 'HYP: '
    for i in range(len(steps)):
        delim = ' ' if i < len(steps) - 1 else '\n'
        if steps[i] == 'sub':
            ref_idx =  i - steps[:i].count('ins')
            hyp_idx = i - steps[:i].count('del')
            if len(ref[ref_idx]) > len(hyp[hyp_idx]):
                out_str += hyp[hyp_idx] +
                    ' ' * (len(ref[ref_idx])-len(hyp[hyp_idx])) + delim
            else:
                out_str += hyp[hyp_idx] + delim
        elif steps[i] == 'del':
            idx = i - steps[:i].count('ins')
            out_str += ' ' * len(ref[idx] + delim
        else:
            assert steps[i] == 'ins' or steps[i] == 'corr'
            idx = i - steps[:i].count('del')
            out_str += hyp[idx] + delim

    out_str += 'STP: '
    for i in range(len(steps)):
        delim = ' ' if i < len(steps) - 1 else '\n'
        if steps[i] == 'sub':
            ref_idx =  i - steps[:i].count('ins')
            hyp_idx = i - steps[:i].count('del')
            if len(ref[ref_idx]) > len(hyp[hyp_idx]):
                out_str += 'S' + ' ' * (len(ref[ref_idx]) - 1) + delim
            else:
                out_str += 'S' + ' ' * (len(hyp[hyp_idx]) - 1) + delim
        elif steps[i] == 'ins':
            idx = i - steps[:i].count('del')
            out_str += 'I' + ' ' * (len(hyp[idx]) - 1) + delim
        else:
            assert steps[i] == 'del' or steps[i] == 'corr'
            idx = i - steps[:i].count('ins')
            sym = 'D' if step[i] == 'del' else ' '
            out_str += sym + ' ' * (len(ref[idx]) - 1) + delim

    counter = Counter(steps)
    wer = float(counter['sub'] + counter['ins'] + counter['del']) / len(ref) \
        * 100
    out_str += 'WER: ' + '{:.2f}%'.format(wer) + '\n'

    return out_str
