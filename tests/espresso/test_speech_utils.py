# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import string
import numpy as np
from collections import Counter

import torch

from espresso.data import AsrDictionary

import espresso.tools.utils as utils


class TestSpeechUtils(unittest.TestCase):

    @staticmethod
    def make_dictionary(vocab, non_lang_syms=[]):
        """construct dictionary."""
        assert isinstance(vocab, list) and isinstance(non_lang_syms, list)
        d = AsrDictionary()
        for token in vocab:
            d.add_symbol(token)
        d.add_symbol('<space>')
        for token in non_lang_syms:
            d.add_symbol(token)
        d.finalize(padding_factor=1)  # don't add extra padding symbols
        d.space_index = d.indices.get('<space>', -1)
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
        self.dict = self.make_dictionary(
            self.vocab,
            non_lang_syms=self.non_lang_syms,
        )
        self.text = [self.generate_text(
            self.vocab, self.oovs, self.non_lang_syms, seed=i,
        ) for i in range(self.num_sentences)]

    def test_speech_tokenizer(self):
        for i, sent in enumerate(self.text):
            print('test sentence {}:'.format(i))
            print(sent)
            tokens = utils.tokenize(
                sent, space=self.dict.space_word,
                non_lang_syms=self.non_lang_syms,
            )

            # test :func:`~speech_tools.utils.tokenize` with
            # :func:`~AsrDictionary.encode_line`
            tensor = self.dict.encode_line(
                tokens, add_if_not_exist=False, append_eos=True,
            )
            reconstructed_tokens = self.dict.string(tensor)
            expected_tokens = ' '.join(
                [token if self.dict.index(token) != self.dict.unk() else
                    self.dict.unk_word for token in tokens.split(' ')]
            )
            self.assertEqual(reconstructed_tokens, expected_tokens)

            # test :func:`~speech_tools.utils.tokenize` with
            # :func:`~AsrDictionary.tokens_to_sentence`
            reconstructed_sent = self.dict.tokens_to_sentence(tokens)
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

    def test_collate_frames(self):
        vals = [
            torch.tensor([4.5, 2.3, 1.2]).unsqueeze(-1).expand(-1, 10),
            torch.tensor([6.7, 9.8]).unsqueeze(-1).expand(-1, 10),
            torch.tensor([7.7, 5.4, 6.2, 8.0]).unsqueeze(-1).expand(-1, 10),
            torch.tensor([1.5]).unsqueeze(-1).expand(-1, 10)]
        expected_res1 = torch.tensor([
            [4.5, 2.3, 1.2, 0.0],
            [6.7, 9.8, 0.0, 0.0],
            [7.7, 5.4, 6.2, 8.0],
            [1.5, 0.0, 0.0, 0.0]]).unsqueeze(-1).expand(-1, -1, 10)
        expected_res2 = torch.tensor([
            [0.0, 4.5, 2.3, 1.2],
            [0.0, 0.0, 6.7, 9.8],
            [7.7, 5.4, 6.2, 8.0],
            [0.0, 0.0, 0.0, 1.5]]).unsqueeze(-1).expand(-1, -1, 10)

        res = utils.collate_frames(vals, pad_value=0.0, left_pad=False)
        self.assertTensorEqual(res, expected_res1)

        res = utils.collate_frames(vals, pad_value=0.0, left_pad=True)
        self.assertTensorEqual(res, expected_res2)

    def test_sequence_mask(self):
        seq_len = torch.tensor([1, 4, 0, 3]).int()
        expected_mask = torch.tensor([
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 0]]).bool()
        expected_mask2 = torch.tensor([
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0]]).bool()

        generated_mask = utils.sequence_mask(seq_len)
        generated_mask2 = utils.sequence_mask(seq_len, max_len=5)

        self.assertTensorEqual(generated_mask, expected_mask)
        self.assertTensorEqual(generated_mask2, expected_mask2)

    def test_convert_padding_direction(self):
        t1 = torch.tensor([
            [4.5, 2.3, 1.2, 0.0],
            [6.7, 9.8, 0.0, 0.0],
            [7.7, 5.4, 6.2, 8.0],
            [1.5, 0.0, 0.0, 0.0]]).unsqueeze(-1).expand(-1, -1, 10)
        t2 = torch.tensor([
            [0.0, 4.5, 2.3, 1.2],
            [0.0, 0.0, 6.7, 9.8],
            [7.7, 5.4, 6.2, 8.0],
            [0.0, 0.0, 0.0, 1.5]]).unsqueeze(-1).expand(-1, -1, 10)
        seq_len = torch.tensor([3, 2, 4, 1]).int()

        t1_to_t2 = utils.convert_padding_direction(
            t1, seq_len, right_to_left=True,
        )
        self.assertTensorEqual(t1_to_t2, t2)

        t2_to_t1 = utils.convert_padding_direction(
            t2, seq_len, left_to_right=True,
        )
        self.assertTensorEqual(t2_to_t1, t1)

    def test_edit_distance(self):
        ref, hyp = [], []
        dist, steps, counter = utils.edit_distance(ref, hyp)
        self.assertEqual(
            counter,
            Counter({'words': 0, 'corr': 0, 'sub': 0, 'ins': 0, 'del': 0}),
        )
        self.assertEqual(steps, [])

        ref, hyp = ['a', 'b', 'c'], []
        dist, steps, counter = utils.edit_distance(ref, hyp)
        self.assertEqual(
            counter,
            Counter({'words': 3, 'corr': 0, 'sub': 0, 'ins': 0, 'del': 3}),
        )
        self.assertEqual(steps, ['del', 'del', 'del'])

        ref, hyp = ['a', 'b', 'c'], ['a', 'b', 'c']
        dist, steps, counter = utils.edit_distance(ref, hyp)
        self.assertEqual(
            counter,
            Counter({'words': 3, 'corr': 3, 'sub': 0, 'ins': 0, 'del': 0}),
        )
        self.assertEqual(steps, ['corr', 'corr', 'corr'])

        ref, hyp = ['a', 'b', 'c'], ['d', 'b', 'c', 'e', 'f']
        dist, steps, counter = utils.edit_distance(ref, hyp)
        self.assertEqual(
            counter,
            Counter({'words': 3, 'corr': 2, 'sub': 1, 'ins': 2, 'del': 0}),
        )
        self.assertEqual(steps, ['sub', 'corr', 'corr', 'ins', 'ins'])

        ref, hyp = ['b', 'c', 'd', 'e', 'f', 'h'], \
            ['d', 'b', 'c', 'e', 'f', 'g']
        dist, steps, counter = utils.edit_distance(ref, hyp)
        self.assertEqual(
            counter,
            Counter({'words': 6, 'corr': 4, 'sub': 1, 'ins': 1, 'del': 1}),
        )
        self.assertEqual(
            steps,
            ['ins', 'corr', 'corr', 'del', 'corr', 'corr', 'sub'],
        )

    def assertTensorEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        if (t1.dtype == torch.short or t1.dtype == torch.int or
            t1.dtype == torch.long or t1.dtype == torch.uint8 or
            t1.dtype == torch.bool) and \
            (t2.dtype == torch.short or t2.dtype == torch.int or
             t2.dtype == torch.long or t2.dtype == torch.uint8 or
             t2.dtype == torch.bool):
            self.assertEqual(t1.ne(t2).long().sum(), 0)
        else:
            self.assertEqual(t1.allclose(t2, rtol=1e-05, atol=1e-08), True)


if __name__ == "__main__":
    unittest.main()
