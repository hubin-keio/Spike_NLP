"""
Test tokenizer
"""

import unittest
import torch
from pnlp.embedding.tokenizer import ProteinTokenizer, index_to_token, token_to_index

class test_ProteinTokenizer(unittest.TestCase):
    def setUp(self):
        self.batch_seqs = []
        with open('test_spike_seq.txt', 'r') as fh:
            for seq in fh.readlines():
                self.batch_seqs.append(seq.rstrip())

    def test_padding(self):
        max_len = 1500
        mask_prob = 0.0
        longest = max([len(seq) for seq in self.batch_seqs])
        tokenizer = ProteinTokenizer(max_len, mask_prob)
        padded_tokens, mask_idx = tokenizer(self.batch_seqs)
        self.assertEqual(padded_tokens.size(), (len(self.batch_seqs), longest))
        self.assertIn(token_to_index['<PAD>'], padded_tokens[:,-5:])

    def test_padding_masking(self):
        max_len = 1500
        mask_prob = 0.15
        longest = max([len(seq) for seq in self.batch_seqs])
        tokenizer = ProteinTokenizer(max_len, mask_prob)
        padded_tokens, mask_idx = tokenizer(self.batch_seqs)
        self.assertEqual(padded_tokens.size(), (len(self.batch_seqs), longest))
        self.assertIn(token_to_index['<PAD>'], padded_tokens)
        self.assertIn(token_to_index['<MASK>'], padded_tokens)

    def test_truncated_seq(self):
        max_len = 500
        mask_prob = 0.15
        
        tokenizer = ProteinTokenizer(max_len, mask_prob)
        tokenized_seqs, _ = tokenizer(self.batch_seqs)
        self.assertEqual(index_to_token[int(tokenized_seqs[0,-1])], '<TRUNCATED>')        
    

if __name__ == '__main__':
    unittest.main()
