"""
Test embedding
"""

import unittest
from pnlp.embedding.tokenizer import ProteinTokenizer
from pnlp.embedding.nlp_embedding import NLPEmbedding

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
        tokenizer = ProteinTokenizer(self.max_len, self.mask_prob)
        embedder = NLPEmbedding(self.embedding_dim, self.max_len, self.dropout)
        tokenized_seqs, _ = tokenizer.get_token(self.batch_seqs)
        embedded_seqs = embedder(tokenized_seqs)
        self.assertEqual(embedded_seqs.shape, (len(self.batch_seqs), self.max_len, self.embedding_dim))


    def test_mask(self):
        """Test mask the <PAD> tokens"""
        max_len = 1500

        tokenizer = ProteinTokenizer(max_len, self.mask_prob)
        embedder = NLPEmbedding(self.embedding_dim, max_len, self.dropout)
        tokenized_seqs, _ = tokenizer.get_token(self.batch_seqs)
        embedded_seqs = embedder(tokenized_seqs)

        padding_idx = tokenizer.token_to_index['<PAD>']
        mask_tensor = tokenized_seqs == padding_idx
        total_masks = (mask_tensor == True).sum().item()

        num_expected_pads = 0
        seq_lens = [len(seq) for seq in self.batch_seqs]
        longest = max(seq_lens)
        for l in seq_lens:
            num_expected_pads += longest - l

        print(f'Shape of sequence batch tensor after padding and masking: {tokenized_seqs.shape}')
        print(f'<PAD> token value: {padding_idx} and will be masked.')
        print(f'Mask tensor (<PAD>) shape: {mask_tensor.size()}')
        print(f'Totalk masks (tokens = <PAD>): {total_masks}, expecting {num_expected_pads}.')

        self.assertEqual(mask_tensor.size(), (len(self.batch_seqs), longest))
        self.assertEqual(total_masks, num_expected_pads)


if __name__ == '__main__':
    unittest.main()
