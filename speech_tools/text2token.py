#!/usr/bin/env python3
# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys

from utils import tokenize


def get_parser():
    parser = argparse.ArgumentParser(
        description='Convert transcripts into tokens and write them to stdout')
    # fmt: off
    parser.add_argument('--skip-ncols', default=0, type=int,
                        help='skip first n columns')
    parser.add_argument('--space', default='<space>', type=str,
                        help='space symbol')
    parser.add_argument('--non-lang-syms', default=None, type=str,
                        help='path to a file listing non-linguistic symbols, '
                        'e.g., <NOISE> etc. One entry per line.')
    parser.add_argument('text', type=str, nargs='?',
                        help='input text')
    # fmt: on

    return parser


def main(args):
    nls = None
    if args.non_lang_syms is not None:
        with open(args.non_lang_syms, 'r', encoding='utf-8') as f:
            nls = [x.rstrip() for x in f.readlines()]
    with (open(args.text, 'r', encoding='utf-8') if args.text else sys.stdin) as f:
        for line in f:
            entry = line.rstrip().split()
            tokenized = tokenize(' '.join(entry[args.skip_ncols:]),
                space=args.space, non_lang_syms=nls)
            if args.skip_ncols > 0:
                print(' '.join(entry[:args.skip_ncols]) + ' ' + tokenized)
            else:
                print(tokenized)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
