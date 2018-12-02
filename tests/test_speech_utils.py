# Copyright (c) 2018-present, Yiming Wang
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import unittest
import string
import numpy as np
import os

import torch

from fairseq.data import TokenDictionary

import speech_tools.utils as utils


class TestSpeechUtils(unittest.TestCase):

    @staticmethod
    def make_dictionary(vocab, non_lang_syms=[]):
        """construct dictionary."""
        assert isinstance(vocab, list) and isinstance(non_lang_syms, list)
        d = TokenDictionary()
        for token in vocab:
            d.add_symbol(token)
        for token in non_lang_syms:
            d.add_symbol(token)
        d.finalize(padding_factor=1) # don't add extra padding symbols
        return d

    @staticmethod
    def generate_text(vocab, oovs=[], non_lang_syms=[], seed=0):
        """generate text of one synthetic sentence."""
        assert isinstance(vocab, list) and isinstance(oovs, list) and \
            isinstance(non_lang_syms, list)
        np.random.seed(seed)
        sent_len = np.random.randint(2, 30)
        sent = '' 
        for _ in range(sent_len):
            if len(non_lang_syms) > 0 and np.random.randint(0, 20) == 0:
                word = non_lang_syms[np.random.randint(0, len(non_lang_syms))]
            else:
                word = ''
                word_len = np.random.randint(2, 11)
                for _ in range(word_len):
                    if len(oovs) > 0 and np.random.randint(0, 20) == 0:
                        word += oovs[np.random.randint(0, len(oovs))]
                    else:
                        word += vocab[np.random.randint(0, len(vocab))]
            sent += word + ' '

        sent = ' '.join(sent.strip().split(' '))
        return sent

    def setUp(self):
        self.vocab = list(string.ascii_lowercase)
        self.oovs = list(string.ascii_uppercase)
        self.non_lang_syms = ['<noise>', '<spnoise>', '<sil>']
        self.num_sentences = 100
        self.dict = self.make_dictionary(self.vocab,
            non_lang_syms=self.non_lang_syms,
        )
        self.text = [self.generate_text(self.vocab, self.oovs,
            self.non_lang_syms, seed=i) for i in range(self.num_sentences)]

    def test_speech_tokenizer(self):
        for i, sent in enumerate(self.text):
            print('test sentence {}:'.format(i))
            print(sent)
            tokens = utils.Tokenizer.tokenize(sent, \
                space=self.dict.space_word, non_lang_syms=self.non_lang_syms)

            # test Tokenizer.tokenize() with Tokenizer.tokens_to_index_tensor()
            tensor = utils.Tokenizer.tokens_to_index_tensor(tokens, self.dict, \
                append_eos=True)
            reconstructed_tokens = self.dict.string(tensor)
            expected_tokens = ' '.join(
                [token if self.dict.index(token) != self.dict.unk() else \
                    self.dict.unk_word for token in tokens.split(' ')]
            )
            self.assertEqual(reconstructed_tokens, expected_tokens)

            # test Tokenizer.tokenize() with Tokenizer.tokens_to_sentence()
            reconstructed_sent = utils.Tokenizer.tokens_to_sentence(tokens,
                self.dict)
            expected_sent = []
            words = sent.split(' ')
            for w in words:
                if w not in self.non_lang_syms:
                    new_word = ''.join(
                        [self.dict.unk_word if c in self.oovs else c for c in w]
                    )
                    expected_sent.append(new_word)
                else:
                    expected_sent.append(w)
            expected_sent = ' '.join(expected_sent)
            self.assertEqual(reconstructed_sent, expected_sent)


if __name__ == "__main__":
    unittest.main()
