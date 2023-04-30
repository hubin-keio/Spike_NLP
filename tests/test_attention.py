"""Test Attention Model"""

import unittest
import torch
from pnlp.embedding.tokenizer import ProteinTokenizer, token_to_index
from pnlp.embedding.nlp_embedding import NLPEmbedding
from pnlp.model.attention import Attention, MultiHeadedAttention


class TestAttention(unittest.TestCase):
    def setUp(self):
        self.embedding_dim = 24
        self.dropout=0.1
        self.max_len = 1500
        self.mask_prob = 0.15

        self.tokenizer = ProteinTokenizer(self.max_len, self.mask_prob)
        self.embedder = NLPEmbedding(self.embedding_dim, self.max_len, self.dropout)

        self.vocab_size = len(token_to_index)
        self.padding_idx = token_to_index['<PAD>']

        self.batch_seqs = []
        with open('test_spike_seq.txt', 'r') as fh:
            for seq in fh.readlines():
                self.batch_seqs.append(seq.rstrip())
        self.longest = max([len(seq) for seq in self.batch_seqs])

    def test_single_head_attention(self):
        tokenized_seqs, _ = self.tokenizer(self.batch_seqs)
        embedded_seq, mask = self.embedder(tokenized_seqs)
        query = key = value = embedded_seq


        print(f'query, key, and value dimensions: {query.shape}')
        print(f'query.size(-1) is {query.size(-1)}')
        print(f'mask tensor shape: {mask.shape}')

        single_at = Attention()
        x, y = single_at(query, key, value)
        print(y)

        single_at_masked = Attention()
        x, y = single_at_masked(query, key, value)
        print(y)

    def test_multi_head_attention(self):
        tokenized_seqs, _ = self.tokenizer(self.batch_seqs)
        embedded_seq, mask = self.embedder(tokenized_seqs)
        query = key = value = embedded_seq

        n_head = 12
        n_linear = 5

        multi_at = MultiHeadedAttention(n_head, self.embedding_dim, n_linear, self.dropout)
        multi_at(query, key, value, mask=mask)


if __name__ == '__main__':
    unittest.main()
