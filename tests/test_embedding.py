"""
Test embedding
"""
import unittest
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
        embedder = NLPEmbedding(self.embedding_dim, self.dropout, self.max_len, self.mask_prob)

        padded_seqs = embedder.batch_pad(self.batch_seqs)
        (padded_masked_seqs, masked_idx) = embedder.batch_mask(padded_seqs)

        self.assertEqual(padded_masked_seqs.shape, (len(self.batch_seqs), self.max_len))
        self.assertEqual(embedder.index_to_token[int(padded_masked_seqs[0,-1])], '<TRUNCATED>')

        for idx in masked_idx:
            self.assertEqual(int(padded_masked_seqs[0, idx]), embedder.token_to_index['<MASK>'])

    def test_forward(self):
        embedder = NLPEmbedding(self.embedding_dim, self.dropout, self.max_len, self.mask_prob)
        x = embedder.forward(self.batch_seqs)
        self.assertEqual(x.size(), (len(self.batch_seqs), self.max_len, self.embedding_dim))

    def test_mask(self):
        """Test mask the <PAD> tokens"""
        max_len = 1500
        embedder = NLPEmbedding(self.embedding_dim, self.dropout, max_len, self.mask_prob)
        x_padded = embedder.batch_pad(self.batch_seqs)
        x_padded_masked, _ = embedder.batch_mask(x_padded)
        padding_idx = embedder.token_to_index['<PAD>']
        mask_tensor = x_padded_masked ==padding_idx
        total_masks = (mask_tensor == True).sum().item()

        num_expected_pads = 0
        seq_lens = [len(seq) for seq in self.batch_seqs]
        longest = max(seq_lens)
        for l in seq_lens:
            num_expected_pads += longest - l        

        print(f'Shape of sequence batch tensor after padding and masking: {x_padded_masked.shape}')
        print(f'<PAD> token value: {padding_idx} and will be masked.')
        print(f'Mask tensor (<PAD>) shape: {mask_tensor.size()}')
        print(f'Totalk masks (tokens = <PAD>): {total_masks}, expecting {num_expected_pads}.')

        self.assertEqual(total_masks, num_expected_pads)


if __name__ == '__main__':
    unittest.main()
