# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os, re
import numpy as np
from collections import Counter

import torch

from fairseq import utils


def tokenize(sent, space='<space>', non_lang_syms=None):
    assert isinstance(sent, str)
    sent = ' '.join(sent.strip().split())

    match_pos = []
    if non_lang_syms is not None:
        assert isinstance(non_lang_syms, list)
        if len(non_lang_syms) > 0:
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
        assert sequence_length.data.max().item() <= utils.item(max_len)
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).to(device=sequence_length.device,
        dtype=sequence_length.dtype)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
    return seq_range_expand < seq_length_expand

def convert_padding_direction(src_frames, src_lengths, right_to_left=False,
    left_to_right=False):
    """Counterpart of :func:`~fairseq.utils.convert_padding_direction`,
    operating on 3d tensors of size B x T x C. Note that this function is unware
    of whether it has already been right padded or left padded (since any real
    value is legal for non-padded elements), so be clear of the actual padding
    direction before calling this function.
    """
    assert right_to_left ^ left_to_right
    assert src_frames.size(0) == src_lengths.size(0)
    max_len = src_frames.size(1)
    if not src_lengths.eq(max_len).any():
        # no padding, return early
        return src_frames
    range = utils.buffered_arange(max_len).unsqueeze(-1).expand_as(src_frames)
    num_pads = (max_len - src_lengths.type_as(range)).unsqueeze(-1).unsqueeze(-1)
    if right_to_left:
        index = torch.remainder(range - num_pads, max_len)
    else:
        index = torch.remainder(range + num_pads, max_len)
    return src_frames.gather(1, index)

def plot_attention(attention, hypo_sent, utt_id, save_dir):
    """This function plots the attention for an example and save the plot in
    save_dir with <utt_id>.pdf as its filename.
    """
    try:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            """This function requires matplotlib.
            Please install it to generate plots, or unset --print-alignment.
            If you are on a cluster where you do not have admin rights you could
            try using virtualenv.""")

    attn = attention.data.numpy()
    plt.matshow(attn)
    plt.title(hypo_sent, fontsize=8)
    filename = os.path.join(save_dir, utt_id + '.pdf')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

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

    dist = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.uint32)
    for i in range(len(ref) + 1):
        for j in range(len(hyp) + 1):
            if i == 0:
                dist[0][j] = j
            elif j == 0:
                dist[i][0] = i
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
        elif i >= 1 and j >= 1 and dist[i][j] == dist[i - 1][j - 1] and \
            ref[i - 1] == hyp[j - 1]:
            steps.append('corr')
            i, j = i - 1, j - 1
        elif i >= 1 and j >= 1 and dist[i][j] == dist[i - 1][j - 1] + 1:
            assert ref[i - 1] != hyp[j - 1]
            steps.append('sub')
            i, j = i - 1, j - 1
        elif j >= 1 and dist[i][j] == dist[i][j - 1] + 1:
            steps.append('ins')
            j = j - 1
        else:
            assert i >= 1 and dist[i][j] == dist[i - 1][j] + 1
            steps.append('del')
            i = i - 1
    steps = steps[::-1]

    counter = Counter({'words': len(ref), 'corr': 0, 'sub': 0, 'ins': 0,
        'del': 0})
    counter.update(steps)

    return dist, steps, counter

def aligned_print(ref, hyp, steps):
    """This funcition is to print the result of comparing reference and
    hypothesis sentences in an aligned way.

    Args:
        ref: list of words obtained by splitting reference sentence string
        hyp: list of words obtained by splitting hypothesis sentence string
        steps: list of edit steps with elements 'corr', 'sub', 'ins' or 'del'.

    Return:
        out_str: aligned reference and hypothesis string with edit steps.
    """

    assert isinstance(ref, list) and isinstance(hyp, list)
    assert isinstance(steps, list)

    if len(steps) == 0:  # in case both ref and hyp are empty
        assert len(ref) == 0 and len(hyp) == 0
        out_str = 'REF: \nHYP: \nSTP: \nWER: {:.2f}%\n\n'.format(0.)
        return out_str

    out_str = 'REF: '
    for i in range(len(steps)):
        delim = ' ' if i < len(steps) - 1 else '\n'
        if steps[i] == 'sub':
            ref_idx =  i - steps[:i].count('ins')
            hyp_idx = i - steps[:i].count('del')
            if len(ref[ref_idx]) < len(hyp[hyp_idx]):
                out_str += ref[ref_idx] + \
                    ' ' * (len(hyp[hyp_idx]) - len(ref[ref_idx])) + delim
            else:
                out_str += ref[ref_idx] + delim
        elif steps[i] == 'ins':
            idx = i - steps[:i].count('del')
            out_str += ' ' * len(hyp[idx]) + delim
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
                out_str += hyp[hyp_idx] + \
                    ' ' * (len(ref[ref_idx]) - len(hyp[hyp_idx])) + delim
            else:
                out_str += hyp[hyp_idx] + delim
        elif steps[i] == 'del':
            idx = i - steps[:i].count('ins')
            out_str += ' ' * len(ref[idx]) + delim
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
            sym = 'D' if steps[i] == 'del' else ' '
            out_str += sym + ' ' * (len(ref[idx]) - 1) + delim

    counter = Counter(steps)
    wer = float(counter['sub'] + counter['ins'] + counter['del']) / len(ref) \
        * 100 if len(ref) > 0 else 0.
    out_str += 'WER: ' + '{:.2f}%'.format(wer) + '\n'
    out_str += '\n'

    return out_str

def lexical_prefix_tree(word_dict, subword_dict, subword_tokenizer=None):
    """Build a lexical prefix tree for words.

    Args:
        word_dict: an instance of :class:`fairseq.data.TokenDictionary`.
        subword_dict: an instance of :class:`fairseq.data.TokenDictionary`.
        subword_tokenizer (callable): a function that takes a word string as its
            only one argument, and returns a list of subwords as a result of
            tokenization.

    Return:
        root (Node): the root of the prefix tree, where each node has the fields:
            ('children': Dict[int,Node], 'word_idx': int, 'word_set': Tuple[int]).
            'children' is subword_idx -> node, and 'word_set' is (first-1, last),
            where [first, last] is the range of the word indexes (inclusive) in
            the word dictionary who share the same prefix at that node.
            We assume words in the word dictionary are in lexical order.
    """

    class Node(object):
        def __init__(self, children={}, word_idx=-1, word_set=None):
            self.children = children
            self.word_idx = word_idx
            self.word_set = word_set

    special_symbols = [word_dict.pad(), word_dict.eos(), word_dict.unk()]
    assert 0 in special_symbols # to ensure widx - 1 >= 0
    root = Node({}, -1, None)
    for widx in range(len(word_dict)):
        if widx not in special_symbols: # skip <pad>, <eos>, <unk>
            # tokenize a word into a list of subwords
            subwords = subword_tokenizer(word_dict[widx]) \
                if subword_tokenizer is not None else list(word_dict[widx])
            if any(subword_dict.index(s) == subword_dict.unk() for s in subwords):
                # skip words containing any unknown subwords
                continue
            children = root.children
            for i, s in enumerate(subwords):
                sidx = subword_dict.index(s)
                if sidx not in children: # make a new node
                    children[sidx] = Node({}, -1, (widx - 1, widx))
                else:
                    children[sidx].word_set = (
                        min(children[sidx].word_set[0], widx - 1),
                        max(children[sidx].word_set[1], widx)
                    )
                if i == len(subwords) - 1: # if word end, set word_idx
                    children[sidx].word_idx = widx
                children = children[sidx].children # move to children
    return root
