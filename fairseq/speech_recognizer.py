# Copyright (c) 2018-present, Yiming Wang
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch

from fairseq import utils

from fairseq.sequence_generator import SequenceGenerator


class SpeechRecognizer(SequenceGenerator):
    def generate_batched_itr(
        self, data_itr, beam_size=None, maxlen_a=0.0, maxlen_b=None,
        cuda=False, timer=None, prefix_size=0,
    ):
        """Iterate over a batched dataset and yield individual transcription.

        Args:
            maxlen_a/b (int, optional): generate sequences of maximum length
                ``ax + b``, where ``x`` is the source sentence length.
            cuda (bool, optional): use GPU for generation
            timer (StopwatchMeter, optional): time generations
            prefix_size (int, optional): prefill the generation with the gold
                prefix up to this length.
        """
        if maxlen_b is None:
            maxlen_b = self.maxlen

        for sample in data_itr:
            s = utils.move_to_cuda(sample) if cuda else sample
            if 'net_input' not in s:
                continue
            input = s['net_input']
            # model.forward normally channels prev_output_tokens into the decoder
            # separately, but SequenceGenerator directly calls model.encoder
            encoder_input = {
                k: v for k, v in input.items()
                if k != 'prev_output_tokens'
            }
            srclen = encoder_input['src_tokens'].size(1)
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(
                    encoder_input,
                    beam_size=beam_size,
                    maxlen=int(maxlen_a*srclen + maxlen_b),
                    prefix_tokens=s['target'][:, :prefix_size] if prefix_size > 0 else None,
                )
            if timer is not None:
                timer.stop(sum(len(h[0]['tokens']) for h in hypos))
            for i, id in enumerate(s['id'].data):
                utt_id = s['utt_id'][i]
                # remove padding
                ref = utils.strip_pad(s['target'].data[i, :], self.pad) if s['target'] is not None else None
                yield id, utt_id, ref, hypos[i]

