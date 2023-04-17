"""
Test embedding
"""
import unittest
from pnlp.embedding import nlp_embedding

class TestNLPEmbedding(unittest.TestCase):
    def setUp(self):
        self.embedding_dim = 10
        self.dropout = 0.1
        self.max_len = 500
        self.mask_prob = 0.15
        self.batch_seqs = []

        with open('test_spike_seq.txt', 'r') as fh:
            for seq in fh.readlines():
                self.batch_seqs.append(seq.rstrip())

    def test_truncated_sequences(self):
        embedder = nlp_embedding.NLPEmbedding(self.embedding_dim, self.dropout, self.max_len, self.mask_prob)

        padded_seqs = embedder.batch_pad(self.batch_seqs)
        (padded_masked_seqs, masked_idx) = embedder.batch_mask(padded_seqs)

        self.assertEqual(padded_masked_seqs.shape, (len(self.batch_seqs), self.max_len))
        self.assertEqual(embedder.index_to_token[int(padded_masked_seqs[0,-1])], '<TRUNCATED>')

        for idx in masked_idx:
            self.assertEqual(int(padded_masked_seqs[0, idx]), embedder.token_to_index['<MASK>'])

    def test_forward(self):
        embedder = nlp_embedding.NLPEmbedding(self.embedding_dim, self.dropout, self.max_len, self.mask_prob)
        x = embedder.forward(self.batch_seqs)
        self.assertEqual(x.size(), (len(self.batch_seqs), self.max_len, self.embedding_dim))

if __name__ == '__main__':
    unittest.main()
