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
logger = logging.getLogger("mobvoihotwords.generate_graphs")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate graphs for training"
    )
    # fmt: off
    parser.add_argument("--hmm-paths", nargs="+", help="list of HMM paths", required=True)
    parser.add_argument("--lexicon-fst-path", type=str, help="path to the lexicon fst", required=True)
    parser.add_argument("--phone-lm-fsa-path", type=str, help="path to the phone LM fsa", required=True)
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
    H = k2.union(hmm_vec)
    H_inv = k2.arc_sort(H.invert_())

    with open(args.lexicon_fst_path, "r", encoding="utf-8") as f:
        L = k2.Fsa.from_openfst(f.read(), acceptor=False)
    L = k2.arc_sort(L.invert_()).invert_()  # sort on olabels

    with open(args.phone_lm_fst_path, "r", encoding="utf-8") as f:
        phone_lm = k2.Fsa.from_openfst(f.read(), acceptor=True)
    phone_lm = k2.arc_sort(phone_lm)

    # emulate composition
    if hasattr(L, "aux_symbols"):
        setattr(L, "temp_symbols", L.aux_symbols)
        delattr(L, "aux_symbols")
    HL = k2.intersect(H_inv, L)
    if hasattr(L, "temp_symbols"):
        setattr(L, "aux_symbols", L.temp_symbols)
        delattr(L, "temp_symbols")
    HL = k2.arc_sort(HL)
    save_path = os.path.join(args.out_dir, "HL.pt")
    torch.save(HL.as_dict(), save_path)
    logger.info(f"saved the HL fst as {save_path}")

    den_graph = k2.intersect(H_inv, phone_lm).invert_()
    den_graph = k2.arc_sort(den_graph)
    save_path = os.path.join(args.out_dir, "denominator.pt")
    torch.save(den_graph.as_dict(), save_path)
    logger.info(f"saved the denominator graph as {save_path}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
