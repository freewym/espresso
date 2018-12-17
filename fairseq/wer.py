# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import Counter

import speech_tools.utils as speech_utils


class Scorer(object):
    def __init__(self, dict):
        self.dict = dict
        self.reset()

    def reset(self):
        self.char_counter = Counter()
        self.word_counter = Counter()
        self.results = ''
        self.aligned_results = ''

    def add_prediction(self, pred, utt_id=None):
        if not isinstance(pred, str):
            raise TypeError('pred must be a string(got {})'.format(type(pred)))
        if utt_id is not None and not isinstance(utt_id, str):
            raise TypeError('utt_id must be a string(got {}) if not None'
                .format(type(utt_id)))

        pred_words= speech_utils.Tokenizer.tokens_to_sentence(pred, self.dict)
        if utt_id is not None:
            self.results += utt_id + '\n'
        self.results += pred_words + '\n'

    def add(self, ref, pred, utt_id=None):
        if not isinstance(ref, str):
            raise TypeError('ref must be a string (got {})'.format(type(ref)))
        if not isinstance(pred, str):
            raise TypeError('pred must be a string(got {})'.format(type(pred)))
        if utt_id is not None and not isinstance(utt_id, str):
            raise TypeError('utt_id must be a string(got {}) if not None'
                .format(type(utt_id)))

        # char level counts
        _, _, counter = speech_utils.edit_distance(ref.strip().split(),
            pred.strip().split())
        self.char_counter += counter

        # word level counts
        ref_words = speech_utils.Tokenizer.tokens_to_sentence(ref, self.dict)
        pred_words= speech_utils.Tokenizer.tokens_to_sentence(pred, self.dict)
        ref_word_list, pred_word_list = ref_words.split(), pred_words.split()
        _, steps, counter = speech_utils.edit_distance(ref_word_list,
            pred_word_list)
        self.word_counter += counter
        if utt_id is not None:
            self.aligned_results += utt_id + '\n'
        self.aligned_results += speech_utils.aligned_print(ref_word_list,
            pred_word_list, steps)

    def cer(self):
        assert self.char_counter['words'] > 0
        cer = float(self.char_counter['sub'] + self.char_counter['ins'] + \
            self.char_counter['del']) / self.char_counter['words'] * 100
        sub = float(self.char_counter['sub']) / self.char_counter['words'] * 100
        ins = float(self.char_counter['ins']) / self.char_counter['words'] * 100
        dlt = float(self.char_counter['del']) / self.char_counter['words'] * 100
        return cer, sub, ins, dlt

    def wer(self):
        assert self.word_counter['words'] > 0
        wer = float(self.word_counter['sub'] + self.word_counter['ins'] + \
            self.word_counter['del']) / self.word_counter['words'] * 100
        sub = float(self.word_counter['sub']) / self.word_counter['words'] * 100
        ins = float(self.word_counter['ins']) / self.word_counter['words'] * 100
        dlt = float(self.word_counter['del']) / self.word_counter['words'] * 100
        return wer, sub, ins, dlt

    def acc_word_error(self):
        return self.word_counter['sub'] + self.word_counter['ins'] + \
            self.word_counter['del']

    def acc_word_count(self):
        return self.word_counter['words']

    @property
    def results(self):
        return self.results
    
    @property
    def aligned_results(self):
        return self.aligned_results

