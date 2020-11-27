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
logger = logging.getLogger("mobvoihotwords.evaluate")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate by calculating detection metrics"
    )
    # fmt: off
    parser.add_argument("--wake-word", type=str, help="wake word to be treated as positive", required=True)
    parser.add_argument("supervsion_file", type=str, help="path to the supervision set file")
    parser.add_argument("hyp_file", type=str, help="path to the resulting hypotheses file")
    parser.add_argument("result_file", type=str, help="path to the result file")
    # fmt: on

    return parser


def main(args):
    try:
        from lhotse import SupervisionSet
    except ImportError:
        raise ImportError("Please install Lhotse by `pip install lhotse`")

    supervisions = SupervisionSet.from_json(args.recording_file)  # one and only one supervision segment per recording
    neg_dur = sum(sup.duration for sup in supervisions if sup.text != args.wake_word)
    ref = [(sup.recording_id, sup.text) for sup in supervisions]

    hyp = {}
    with open(args.hyp_file, "r", encoding="utf-8") as f:
        for line in f:
            split_line = line.strip().split(maxsplit=1)
            hyp[split_line[0]] = split_line[1] if len(split_line) == 2 else ""

    if len(ref) != len(hyp):
        logger.warning("The lengths of reference and hypothesis do not match. ref: {} vs hyp: {}.".format(len(ref), len(hyp)))

    TP = TN = FP = FN = 0.0
    for i in range(len(ref)):
        if ref[i][0] not in hyp:
            logger.warning("reference {} does not exist in hypothesis.".format(ref[i][0]))
            continue
        if ref[i][1] == args.wake_word:
            if args.wake_word in hyp[ref[i][0]]:
                TP += 1.0
            else:
                FN += 1.0
        else:
            if args.wake_word in hyp[ref[i][0]]:
                FP += 1.0
            else:
                TN += 1.0
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    false_positive_rate = FP / (FP + TN) if FP + TN > 0 else 0.0
    false_negative_rate = FN / (FN + TP) if FN + TP > 0 else 0.0
    false_alarms_per_hour = FP / (neg_dur / 3600) if neg_dur > 0.0 else 0.0

    with open(args.result_file, "w", encoding="utf-8") as f:
        print(
            "precision: {:.5f}  recall: {:.5f}  FPR: {:.5f}  FNR: {:.5f}  FP per hour: {:.5f}  total: {:d}".format(
                precision, recall, false_positive_rate, false_negative_rate, false_alarms_per_hour, TP + TN + FP + FN
            ),
            file=f
        )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
