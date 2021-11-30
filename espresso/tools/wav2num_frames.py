#!/usr/bin/env python3
# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import re
import sys
from concurrent.futures.thread import ThreadPoolExecutor

from tqdm import tqdm

from espresso.tools.utils import compute_num_frames_from_feat_or_waveform

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr,
)
logger = logging.getLogger(__file__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Compute num_frames from a raw waveform file and write them to stdout"
    )
    # fmt: off
    parser.add_argument(
        "file", type=str, nargs="?",
        help="input file where each line has the format '<utt-id> <file-path>' or '<utt-id> <command>'"
    )
    parser.add_argument("--num-workers", type=int, default=20, help="number of workers for multi-thread processing")
    # fmt: on

    return parser


def process(line):
    utt_id, rxfile = line.rstrip().split(None, 1)
    assert (
        re.search(r"\.ark:\d+$", rxfile.strip()) is None
    ), "Please provide raw waveform files"
    num_frames = compute_num_frames_from_feat_or_waveform(rxfile)
    return utt_id + " " + str(num_frames)


def main(args):
    logger.info(f"Computing num_frames directly from wavforms in {args.file}")
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = []
        with (open(args.file, "r", encoding="utf-8") if args.file else sys.stdin) as f:
            for line in f:
                futures.append(ex.submit(process, line))

        for future in tqdm(futures, desc="Processing", leave=False):
            result = future.result()
            print(result)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
