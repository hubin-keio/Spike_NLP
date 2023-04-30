"""
Test embedding
"""

import torch
import torch.nn as nn
import unittest
from pnlp.embedding.tokenizer import ProteinTokenizer, PADDING_IDX, token_to_index
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
        tokenized_seqs, _ = tokenizer(self.batch_seqs)
        embedded_seqs, mask_tensor = embedder(tokenized_seqs)

        seq_lens = [len(seq) for seq in self.batch_seqs]
        longest = max(seq_lens)
        mask_dim = min(longest, self.max_len)

        print(f'Batch sequence embedded shape: {embedded_seqs.shape}')
        print(f'Mask tensor shape: {mask_tensor.shape}')

        self.assertEqual(embedded_seqs.shape, (len(self.batch_seqs), self.max_len, self.embedding_dim))
        self.assertEqual(mask_tensor.shape, (len(self.batch_seqs), mask_dim, mask_dim))

    def test_truncated_sequences_cuda(self):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print('No CUDA device found. Skip cuda test_mask_cuda')
            return
        print(f'Test embedding on {device}')
        tokenizer = ProteinTokenizer(self.max_len, self.mask_prob)
        embedder = NLPEmbedding(self.embedding_dim, self.max_len, self.dropout)
        embedder = embedder.cuda()

        
        tokenized_seqs, _ = tokenizer(self.batch_seqs)
        embedded_seqs, mask_tensor = embedder(tokenized_seqs.cuda())
        seq_lens = [len(seq) for seq in self.batch_seqs]
        longest = max(seq_lens)
        mask_dim = min(longest, self.max_len)

        print(f'Batch sequence embedded shape: {embedded_seqs.shape}')
        print(f'Mask tensor shape: {mask_tensor.shape}')

        self.assertEqual(embedded_seqs.shape, (len(self.batch_seqs), self.max_len, self.embedding_dim))
        self.assertEqual(mask_tensor.shape, (len(self.batch_seqs), mask_dim, mask_dim))
        

    def test_mask(self):
        """Test mask the <PAD> tokens"""
        max_len = 1500

        tokenizer = ProteinTokenizer(max_len, self.mask_prob)
        tokenized_seqs, _ = tokenizer(self.batch_seqs)
        mask_tensor = tokenized_seqs == PADDING_IDX
        total_pads = (mask_tensor == True).sum().item()

        num_expected_pads = 0
        seq_lens = [len(seq) for seq in self.batch_seqs]
        longest = max(seq_lens)
        for l in seq_lens:
            num_expected_pads += longest - l

        print(f'Shape of sequence batch tensor after padding and masking: {tokenized_seqs.shape}')
        print(f'<PAD> token value: {PADDING_IDX} and will be masked.')
        print(f'Total padded tokens : {total_pads}, expecting {num_expected_pads}.')

        self.assertEqual(mask_tensor.size(), (len(self.batch_seqs), longest))
        self.assertEqual(total_pads, num_expected_pads)


if __name__ == '__main__':
    unittest.main()
