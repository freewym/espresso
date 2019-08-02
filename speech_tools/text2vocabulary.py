#!/usr/bin/env python3
# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
from collections import Counter


def get_parser():
    parser = argparse.ArgumentParser(
        description='Create a vocabulary from text files')
    # fmt: off
    parser.add_argument('--skip-ncols', default=0, type=int,
                        help='skip first n columns')
    parser.add_argument('--cutoff', default=0, type=int,
                        help='cut-off frequency')
    parser.add_argument('--vocabsize', default=20000, type=int,
                        help='vocabulary size')
    parser.add_argument('--exclude', type=str, default=None,
                        help='space separated, list of excluding words, '
                        'e.g., <unk> <eos> etc.')
    parser.add_argument('--valid-text', type=str, default=None,
                        help='path to the validation text')
    parser.add_argument('--test-text', type=str, default=None,
                        help='path to the test text')
    parser.add_argument('text_files', nargs='*',
                        help='input text files')
    # fmt: on

    return parser


def main(args):
    exclude = args.exclude.split(' ') if args.exclude is not None else []
    if len(args.text_files) == 0:
        args.text_files.append('-')

    counter = Counter()
    for fn in args.text_files:
        with (open(fn, 'r', encoding='utf-8') if fn != '-' else sys.stdin) as f:
            for line in f:
                tokens = line.rstrip().split()[args.skip_ncols:]
                tokens = [tok for tok in tokens if tok not in exclude]
                counter.update(tokens)

    total_count = sum(counter.values())
    most_common = counter.most_common(args.vocabsize)
    cutoff_point = 0
    invocab_count = 0
    for elem in most_common:
        if elem[1] < args.cutoff:
            break
        invocab_count += elem[1]
        cutoff_point += 1
    cutoff_freq = most_common[cutoff_point - 1][1]
    most_common = most_common[:cutoff_point]

    oov_rate = 1. - float(invocab_count) / total_count
    print('training set:', file=sys.stderr)
    print('  total #tokens={:d}'.format(total_count), file=sys.stderr)
    print('  OOV rate={:.2f}%'.format(oov_rate * 100), file=sys.stderr)
    print('  cutoff frequency={:d}'.format(cutoff_freq), file=sys.stderr)

    # words in vocabulary are lexically sorted
    for w, c in sorted(most_common, key=lambda x: x[0]):
        print('{} {:d}'.format(w, c))

    vocab_set = set(list(zip(*most_common))[0])
    if args.valid_text is not None:
        total_count = 0
        invocab_count = 0
        with open(args.valid_text, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.rstrip().split()[args.skip_ncols:]
                tokens = [tok for tok in tokens if tok not in exclude]
                total_count += len(tokens)
                invocab_count += len([tok for tok in tokens if tok in vocab_set])
        oov_rate = 1. - float(invocab_count) / total_count
        print('validation set:', file=sys.stderr)
        print('  total #tokens={:d}'.format(total_count), file=sys.stderr)
        print('  OOV rate={:.2f}%'.format(oov_rate * 100), file=sys.stderr)

    if args.test_text is not None:
        total_count = 0
        invocab_count = 0
        with open(args.test_text, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.rstrip().split()[args.skip_ncols:]
                tokens = [tok for tok in tokens if tok not in exclude]
                total_count += len(tokens)
                invocab_count += len([tok for tok in tokens if tok in vocab_set])
        oov_rate = 1. - float(invocab_count) / total_count
        print('test set:', file=sys.stderr)
        print('  total #tokens={:d}'.format(total_count), file=sys.stderr)
        print('  OOV rate={:.2f}%'.format(oov_rate * 100), file=sys.stderr)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args) 
