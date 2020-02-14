#!/usr/bin/env python3
# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        description="Wrap all related files of a dataset into a single json file"
    )
    # fmt: off
    parser.add_argument("--feat-files", nargs="+", required=True,
                        help="path(s) to scp feature file(s)")
    parser.add_argument("--text-files", nargs="+", default=None,
                        help="path(s) to text file(s)")
    parser.add_argument("--utt2num-frames-files", nargs="+", default=None,
                        help="path(s) to utt2num_frames file(s)")
    parser.add_argument("--output", required=True, type=argparse.FileType("w"),
                        help="path to save json output")
    args = parser.parse_args()
    # fmt: on

    wrapped = {"feat_files": args.feat_files}
    if args.text_files is not None:
        wrapped["text_files"] = args.text_files
    if args.utt2num_frames_files is not None:
        wrapped["utt2num_frames_files"] = args.utt2num_frames_files
    json.dump(wrapped, args.output, indent=4)


if __name__ == "__main__":
    main()
