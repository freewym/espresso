# Copyright (c) Tongfei Chen, Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os, re
import numpy as np
from typing import *

import torch

from espresso.data import TokenDictionary
from espresso.tools.utils import lexical_prefix_tree


class TensorizedPrefixTree:
    """
    A tensorized lexical prefix tree designed for maximum parallelism in ASR decoding.
    """

    def __init__(self,
                 children: np.ndarray,  # NodeId[NodeId, NumChildren]
                 prev_subword_idx: np.ndarray,  # SubWordId[NodeId]
                 word_idx: np.ndarray,  # WordId[NodeId]; -1 means non-terminal node
                 word_set_idx: np.ndarray  # WordId[NodeId, 2 = (first-1, last)]
                 ):
        self.children = children
        self.prev_subword_idx = prev_subword_idx
        self.word_idx = word_idx
        self.word_set_idx = word_set_idx

        self.none_id = 0
        self.root_id = 1

    def max_out_degree(self) -> int:
        return self.children.shape[1]

    def to_cuda(self, device):
        self.children = self.children.to(device=device)
        self.prev_subword_idx = self.prev_subword_idx.to(device=device)
        self.word_idx = self.word_idx.to(device=device)
        self.word_set_idx = self.word_set_idx.to(device=device)

    @staticmethod
    def build(
            word_dict: TokenDictionary,
            subword_dict: TokenDictionary,
            subword_tokenizer: Callable[[str], List[str]] = None
    ):
        """
        Builds a tensorized lexical prefix tree for words.
        """

        root = lexical_prefix_tree(
            word_dict=word_dict,
            subword_dict=subword_dict,
            subword_tokenizer=subword_tokenizer
        )  # build traditional tree data structure by reusing existing routines

        # Performs pre-order traversal of this tree to assign an index for each node
        max_num_children = 0
        nodes = [None]  # nodes[0] is a dummy node for OOV
        node_to_id_dict = {}
        stack = [root]

        while len(stack) > 0:
            curr = stack.pop()
            node_id = len(nodes)
            nodes.append(curr)
            node_to_id_dict[curr] = node_id
            if len(curr.children) > max_num_children:
                max_num_children = len(curr.children)

            # Guarantee that the children are traversed ascendingly according to the subword index
            for _, next_node in sorted(curr.children.items(), key=lambda t: t[0], reverse=True):
                stack.append(next_node)

        # Construct the tree
        num_nodes = len(nodes)
        children = np.full([num_nodes, max_num_children], 0, dtype=np.int64)
        prev_subword_idx = np.full([num_nodes], subword_dict.pad(), dtype=np.int64)
        word_idx = np.full([num_nodes], -1, dtype=np.int64)
        word_set_idx = np.full([num_nodes, 2], word_dict.pad(), dtype=np.int64)

        for node_id in range(1, len(nodes)):  # skip 0, which is `None`
            node = nodes[node_id]
            # Guarantee that the children are traversed ascendingly according to the subword index
            for i, (subword_id, child) in enumerate(sorted(node.children.items(), key=lambda t: t[0])):
                child_node_id = node_to_id_dict[child]
                children[node_id, i] = child_node_id
                prev_subword_idx[child_node_id] = subword_id

            word_idx[node_id] = node.word_idx
            if node.word_set is not None:
                word_set_idx[node_id] = node.word_set
            else:
                word_set_idx[node_id] = [0, len(word_dict) - 1]

        return TensorizedPrefixTree(
            children=torch.from_numpy(children),
            prev_subword_idx=torch.from_numpy(prev_subword_idx),
            word_idx=torch.from_numpy(word_idx),
            word_set_idx=torch.from_numpy(word_set_idx)
        )
