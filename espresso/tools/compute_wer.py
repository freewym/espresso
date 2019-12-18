#!/usr/bin/env python3
# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import re
import sys
from collections import Counter

from espresso.tools.utils import edit_distance


def get_parser():
    parser = argparse.ArgumentParser(
        description='Compute WER from text')
    # fmt: off
    parser.add_argument('--non-lang-syms', default=None, type=str,
                        help='path to a file listing non-linguistic symbols, '
                        'e.g., <NOISE> etc. One entry per line.')
    parser.add_argument('--wer-output-filter', default=None, type=str,
                        help='path to wer_output_filter file for WER evaluation')
    parser.add_argument('ref_text', type=str,
                        help='path to the reference text file')
    parser.add_argument('hyp_text', type=str,
                        help='path to the hypothesis text file')

    # fmt: on

    return parser


def main(args):
    non_lang_syms = []
    if args.non_lang_syms is not None:
        with open(args.non_lang_syms, 'r', encoding='utf-8') as f:
            non_lang_syms = [x.rstrip() for x in f.readlines()]

    word_filters = []
    if args.wer_output_filter is not None:
        with open(args.wer_output_filter, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#!') or line == '':
                    continue
                elif line.startswith('s/'):
                    m = re.match(r's/(\S+)/(\w*)/g', line)
                    assert m is not None
                    word_filters.append([m.group(1), m.group(2)])
                elif line.startswith('s:'):
                    m = re.match(r's:(\S+):(\w*):g', line)
                    assert m is not None
                    word_filters.append([m.group(1), m.group(2)])
                else:
                    print(
                        'Unsupported pattern: "{}", ignored'.format(line),
                        file=sys.stderr,
                    )

    refs = {}
    with open(args.ref_text, 'r', encoding='utf-8') as f:
        for line in f:
            utt_id, text = line.strip().split(None, 1)
            assert utt_id not in refs, utt_id
            refs[utt_id] = text

    wer_counter = Counter()
    with open(args.hyp_text, 'r', encoding='utf-8') as f:
        for line in f:
            utt_id, text = line.strip().split(None, 1)
            assert utt_id in refs, utt_id
            ref, hyp = refs[utt_id], text

            # filter words according to word_filters (support re.sub only)
            for pattern, repl in word_filters:
                ref = re.sub(pattern, repl, ref)
                hyp = re.sub(pattern, repl, hyp)

            # filter out any non_lang_syms from ref and hyp
            ref_list = [x for x in ref.split() if x not in non_lang_syms]
            hyp_list = [x for x in hyp.split() if x not in non_lang_syms]

            _, _, counter = edit_distance(ref_list, hyp_list)
            wer_counter += counter

    assert wer_counter['words'] > 0
    wer = float(
        wer_counter['sub'] + wer_counter['ins'] + wer_counter['del']
    ) / wer_counter['words'] * 100
    sub = float(wer_counter['sub']) / wer_counter['words'] * 100
    ins = float(wer_counter['ins']) / wer_counter['words'] * 100
    dlt = float(wer_counter['del']) / wer_counter['words'] * 100

    print('WER={:.2f}%, Sub={:.2f}%, Ins={:.2f}%, Del={:.2f}%, #words={:d}'.format(
        wer, sub, ins, dlt, wer_counter['words']))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
