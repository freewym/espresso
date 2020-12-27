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
logger = logging.getLogger("mobvoihotwords.create_decoding_graph")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Create the decoding graph for decoding"
    )
    # fmt: off
    parser.add_argument("--H-path", type=str, help="path to the H fst file (torch_saved). Note: pdf-ids are offset by +1", required=True)
    parser.add_argument("--L-path", type=str, help="path to the L fst file (openfst text format or torch saved)", required=True)
    parser.add_argument("--G-path", type=str, help="path to the LM fsa (openfst text format or torch saved)", required=True)
    parser.add_argument(
        "--first-phone-disambig-id", type=int, default=999,
        help="An integer ID corresponding to the first disambiguation symbol in the phonetic alphabet"
    )
    parser.add_argument(
        "--first-word-disambig-id", type=int, default=999999,
        help="An integer ID corresponding to the first disambiguation symbol in the words vocabulary"
    )
    parser.add_argument("out_dir", type=str, help="directory to save the decoding graph")
    # fmt: on

    return parser


def main(args):
    try:
        import k2
    except ImportError:
        raise ImportError("Please install k2 by `pip install k2`")

    H_inv = k2.arc_sort(k2.invert(k2.Fsa.from_dict(torch.load(args.H_path))))

    if args.L_path[-3:] == ".pt":
        L_inv = k2.arc_sort(k2.Fsa.from_dict(torch.load(args.L_path)).invert_())
    else:
        with open(args.L_path, "r", encoding="utf-8") as f:
            L_inv = k2.arc_sort(k2.Fsa.from_openfst(f.read(), acceptor=False).invert_())

    if args.G_path[-3:] == ".pt":
        G = k2.arc_sort(k2.Fsa.from_dict(torch.load(args.G_path)))
    else:
        with open(args.G_path, "r", encoding="utf-8") as f:
            G = k2.arc_sort(k2.Fsa.from_openfst(f.read(), acceptor=True))

    LG = k2.connect(k2.intersect(G, L_inv)).invert_()
    LG = k2.connect(k2.determinize(LG))
    LG.labels[LG.labels >= args.first_phone_disambig_id] = 0
    if isinstance(LG.aux_labels, torch.Tensor):
        LG.aux_labels[LG.aux_labels >= args.first_word_disambig_id] = 0
    else:
        LG.aux_labels.values()[LG.aux_labels.values() >= args.first_word_disambig_id] = 0
    LG = k2.arc_sort(k2.connect(k2.remove_epsilons_iterative_tropical(LG)))

    LG.temp_labels = LG.aux_labels
    del LG.aux_labels
    HLG_inv = k2.connect(k2.intersect(LG, H_inv))
    HLG_inv.labels = HLG_inv.temp_labels
    del HLG_inv.temp_labels
    HLG = k2.invert(HLG_inv)

    HLG = k2.connect(k2.remove_epsilons_iterative_tropical(HLG))
    #HLG = k2.connect(k2.determinize(HLG))
    HLG.labels = torch.where(HLG.labels > 0, HLG.labels - 1, HLG.labels)
    HLG.aux_labels = k2.ragged.remove_values_eq(HLG.aux_labels, 0)

    os.makedirs(args.out_dir, exist_ok=True)
    save_path = os.path.join(args.out_dir, "HCLG.pt")
    torch.save(HLG.as_dict(), save_path)
    logger.info(f"saved the decoding graph as {save_path}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
