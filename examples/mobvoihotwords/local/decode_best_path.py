#!/usr/bin/env python3
# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import sys

import torch


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("mobvoihotwords.decode_best_path")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Decode by finding the best path"
    )
    # fmt: off
    parser.add_argument("--beam", type=float, default=10.0, help="decoding beam")
    parser.add_argument("--word-symbol-table", type=str, help="path to the HCL fst file (torch_saved)", required=True)
    parser.add_argument("decoding_graph", type=str, default="data", help="path to the decoding graph")
    parser.add_argument("net_output", type=str, help="path to the network output file for acoustic scores")
    parser.add_argument("hyp_file", type=str, help="path to the resulting hypotheses file")
    # fmt: on

    return parser


def main(args):
    try:
        import k2
    except ImportError:
        raise ImportError("Please install k2 by `pip install k2`")
    try:
        import kaldi_io
    except ImportError:
        raise ImportError("Please install kaldi_io by `pip install kaldi_io`")

    symbol_table = k2.SymbolTable.from_file(args.word_symbol_table)
    graph = k2.Fsa.from_dict(torch.load(args.args.decoding_graph))
    graph.scores.requires_grad_(False)

    num_processed = 0
    with open(args.net_output, "r", encoding="utf-8") as f_in, open(args.hyp_file, "r", encoding="utf-8") as f_out:
        for line in f_in:
            utt_id, rxfile = line.strip().split(maxsplit=1)
            net_output = torch.from_numpy(kaldi_io.read_mat(rxfile)).unsqueeze(0)  # 1 x T x V
            supervision_segments = net_output.new_tensor([0, 0, net_output.size(0)], dtype=torch.int).unsqueeze(0)  # 1 x 3
            dense_fsa_vec = k2.DenseFsaVec(net_output, supervision_segments)
            graph = graph.to(dense_fsa_vec.device)
            graph_unrolled = k2.intersect_dense_pruned(
                graph, dense_fsa_vec, search_beam=args.beam, output_beam=15.0, min_active_states=0, max_active_states=10000
            )
            best_path = k2.shortest_path(graph_unrolled, use_double_scores=False)
            if isinstance(best_path[0].aux_labels, torch.Tensor):
                aux_labels = best_paths[0].aux_labels
            else:
                # it's a ragged tensor
                aux_labels = best_path[0].aux_labels.values()
            aux_labels = aux_labels[aux_labels > 0]
            aux_labels = aux_labels.tolist()
            hyp = [symbol_table.get(x) for x in aux_labels]
            print(utt_id, hyp, file=f_out)
            num_processed += 1

    logger.info(f"Processed {num_processed} utterances")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
