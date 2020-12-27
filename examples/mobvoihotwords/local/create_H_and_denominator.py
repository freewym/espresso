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
logger = logging.getLogger("mobvoihotwords.create_H_and_denominator")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate graphs for training"
    )
    # fmt: off
    parser.add_argument("--hmm-paths", nargs="+", help="list of HMM paths (in openfst text format)", required=True)
    parser.add_argument("--phone-lm-fsa-path", type=str, help="path to the phone LM fsa (in openfst text format)", required=True)
    parser.add_argument("--out-dir", type=str, default="data", help="directory to save output graphs")
    # fmt: on

    return parser


def main(args):
    try:
        import k2
    except ImportError:
        raise ImportError("Please install k2 by `pip install k2`")

    hmms = []
    for hmm in args.hmm_paths:
        with open(hmm, "r", encoding="utf-8") as f:
            hmms.append(k2.Fsa.from_openfst(f.read(), acceptor=False))
        hmms[-1] = k2.arc_sort(hmms[-1])
    hmm_vec = k2.create_fsa_vec(hmms)
    # temporarily +1 to make label 0 of HMMs (means "blank") be treated as normal label
    hmm_vec.labels = torch.where(hmm_vec.labels >= 0, hmm_vec.labels + 1, hmm_vec.labels)
    H = k2.closure(k2.union(hmm_vec))
    H_inv = k2.arc_sort(k2.invert(H))
    save_path = os.path.join(args.out_dir, "H.pt")  # H's pdf-ids are offset by +1
    torch.save(H.as_dict(), save_path)
    logger.info(f"saved H as {save_path}")

    with open(args.phone_lm_fsa_path, "r", encoding="utf-8") as f:
        phone_lm = k2.arc_sort(k2.Fsa.from_openfst(f.read(), acceptor=True))

    den_graph = k2.invert(k2.connect(k2.intersect(H_inv, phone_lm)))
    den_graph = k2.connect(k2.remove_epsilons_iterative_tropical(den_graph))
    den_graph = k2.arc_sort(k2.connect(k2.determinize(den_graph)))
    assert (den_graph.labels == 0).int().sum().item() == 0
    den_graph.labels = torch.where(den_graph.labels > 0, den_graph.labels - 1, den_graph.labels)  # restore the label indices
    del den_graph.aux_labels
    save_path = os.path.join(args.out_dir, "denominator.pt")
    torch.save(den_graph.as_dict(), save_path)
    logger.info(f"saved the denominator graph as {save_path}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
