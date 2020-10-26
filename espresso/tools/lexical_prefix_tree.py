# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List

from espresso.data import AsrDictionary


def lexical_prefix_tree(
    word_dict: AsrDictionary,
    subword_dict: AsrDictionary,
    subword_tokenizer: Callable[[str], List[str]] = None
):
    """Build a lexical prefix tree for words.

    Args:
        word_dict: an instance of :class:`fairseq.data.AsrDictionary`.
        subword_dict: an instance of :class:`fairseq.data.AsrDictionary`.
        subword_tokenizer (callable): a function that takes a word string as its
            only one argument, and returns a list of subwords as a result of
            tokenization.

    Return:
        root (Node): the root of the prefix tree, where each node has the fields:
            ("children": Dict[int,Node], "word_idx": int, "word_set": Tuple[int]).
            "children" is subword_idx -> node, and "word_set" is (first-1, last),
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
    assert 0 in special_symbols  # to ensure widx - 1 >= 0
    root = Node({}, -1, None)
    for widx in range(len(word_dict)):
        if widx not in special_symbols:  # skip <pad>, <eos>, <unk>
            # tokenize a word into a list of subwords
            subwords = (
                subword_tokenizer(word_dict[widx])
                if subword_tokenizer is not None
                else list(word_dict[widx])
            )
            if any(subword_dict.index(s) == subword_dict.unk() for s in subwords):
                # skip words containing any unknown subwords
                continue
            children = root.children
            for i, s in enumerate(subwords):
                sidx = subword_dict.index(s)
                if sidx not in children:  # make a new node
                    children[sidx] = Node({}, -1, (widx - 1, widx))
                else:
                    children[sidx].word_set = (
                        min(children[sidx].word_set[0], widx - 1),
                        max(children[sidx].word_set[1], widx)
                    )
                if i == len(subwords) - 1:  # if word end, set word_idx
                    children[sidx].word_idx = widx
                children = children[sidx].children  # move to children
    return root
