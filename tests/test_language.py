"""Test Language Model"""

from os import path
import torch
import unittest
from pnlp.embedding.tokenizer import ProteinTokenizer, token_to_index, index_to_token, PADDING_IDX
from pnlp.embedding.nlp_embedding import NLPEmbedding
from pnlp.model.bert import BERT
from pnlp.model.language import ProteinLM, ProteinMaskedLanguageModel

class test_PLM(unittest.TestCase):
    def setUp(self):
        self.embedding_dim = 24
        self.dropout=0.1
        self.max_len = 1500
        self.mask_prob = 0.15

        self.tokenizer = ProteinTokenizer(self.max_len, self.mask_prob)
        self.embedder = NLPEmbedding(self.embedding_dim, self.max_len, self.dropout)

        self.vocab_size = len(token_to_index)
        self.padding_idx = PADDING_IDX
        self.hidden = self.embedding_dim
        self.n_transformer_layers = 12
        self.attn_heads = 12

        self.batch_seqs = []
        test_seq = path.abspath(path.dirname(__file__))
        test_seq = path.join(test_seq, 'test_spike_seq.txt')

        with open(test_seq, 'r') as fh:
            for seq in fh.readlines():
                self.batch_seqs.append(seq.rstrip())
        self.longest = max([len(seq) for seq in self.batch_seqs])

    def test_plm_forward(self):
        max_len = 50
        bert = BERT(self.embedding_dim,
                    self.dropout,
                    max_len,
                    self.mask_prob,
                    self.n_transformer_layers,
                    self.attn_heads)

        plm = ProteinLM(bert, self.vocab_size)
        tokenized_seqs, masked_idx = self.tokenizer(self.batch_seqs)
        tokenized_seqs = tokenized_seqs[:, :max_len]
        output = plm.forward(tokenized_seqs)
        output = torch.argmax(output, dim=-1)

        num_parameters = sum(p.numel() for p in plm.parameters() if p.requires_grad)
        self.assertEqual(output.size(), (len(self.batch_seqs), min(self.longest, max_len)))
        print(f'Number of parameters: {num_parameters}')
        print(f'Output shape: {output.shape}')

        for seq_idx in range(5):
            assert len(self.batch_seqs[seq_idx]) >= max_len
            input_seq = self.batch_seqs[seq_idx][:max_len]
            output_seq = ''.join([index_to_token[x.item()] for x in output[seq_idx]])
            print(f'{seq_idx}\ninput: {input_seq}\noutput: {output_seq}\n')


if __name__ == '__main__':
    unittest.main()
