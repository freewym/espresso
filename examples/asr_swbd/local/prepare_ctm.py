#!/usr/bin/env python3
# Copyright (c) Hang Lyu, Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This script is use to transform the word level results to ctm format
# The start_time and end_time of each word is fake
# The format of input is like "en_4156-A_030185-030248 oh yeah"
# The format of output is like "en_4156 A start_time duration oh", and so on

import argparse
import re


def get_parser():
    parser = argparse.ArgumentParser(
        description="""Transform the word level results to ctm format""")
    # fmt: off
    parser.add_argument('ori_result', type=str,
                        help='The input: original decoded results.')
    parser.add_argument('ctm_result', type=str,
                        help='The output: transformed results in ctm format.')
    # fmt: on

    return parser


def main(args):
    # read in word level results
    with open(args.ori_result, 'r', encoding='utf-8') as f:
        content = f.readlines()
    # convert each line
    split_content = []  # store ctm results
    for i, line in enumerate(content):
        elements = line.strip().split()

        # The first field contains the information of the utterance
        utt_info = elements[0]
        infos = re.split('[-_]', utt_info)
        utt_id = infos[0] + '_' + infos[1]
        channel = infos[2]
        start_time = round((int(infos[3]) / 100.0), 2)
        end_time = round((int(infos[4]) / 100.0), 2)

        # generate ctm format results for each word
        time_diff = int(infos[4]) - int(infos[3])
        time_step = round((float(time_diff) / (len(elements) - 1) / 100), 2) \
            if len(elements) > 1 else 0
        for j, word in enumerate(elements):
            start_time_cur = start_time + time_step * (j - 1)
            duration = 0.0
            if j == 0:
                continue
            elif j == len(elements) - 1:
                duration = end_time - start_time_cur
                split_content.append(
                    ' '.join([utt_id, channel, str(round(start_time_cur, 2)),
                              str(round(duration, 2)), word])
                )
            else:
                duration = time_step
                split_content.append(
                    ' '.join([utt_id, channel, str(round(start_time_cur, 2)),
                              str(round(duration, 2)), word])
                )
        if j == 0:
            split_content.append(
                ' '.join([utt_id, channel, str(round(start_time, 2)),
                          str(round(time_diff, 2)), '[noise]'])
            )

    with open(args.ctm_result, 'w', encoding='utf-8') as f:
        for line in split_content:
            f.write(line + '\n')


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
