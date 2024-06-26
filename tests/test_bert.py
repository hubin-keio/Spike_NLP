"""Test BERT Model"""

import unittest
from pnlp.embedding.tokenizer import ProteinTokenizer, token_to_index
from pnlp.embedding.nlp_embedding import NLPEmbedding
from pnlp.model.bert import BERT

class test_BERT(unittest.TestCase):
    def setUp(self):
        self.embedding_dim = 24
        self.dropout=0.1
        self.max_len = 500
        self.mask_prob = 0.15

        self.tokenizer = ProteinTokenizer(self.max_len, self.mask_prob)
        self.embedder = NLPEmbedding(self.embedding_dim, self.max_len, self.dropout)

        self.vocab_size = len(token_to_index)
        self.padding_idx = token_to_index['<PAD>']
        self.hidden = self.embedding_dim
        self.n_transformer_layers = 12
        self.attn_heads = 12

        self.batch_seqs = []
        with open('test_spike_seq.txt', 'r') as fh:
            for seq in fh.readlines():
                self.batch_seqs.append(seq.rstrip())

    def test_bert_foward(self):
        model = BERT(self.embedding_dim,
                     self.dropout,
                     self.max_len,
                     self.mask_prob,
                     self.n_transformer_layers,
                     self.attn_heads)
        tokenized_seqs = self.tokenizer(self.batch_seqs)
        output = model(tokenized_seqs)
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(output.size(), (len(self.batch_seqs), self.max_len, self.embedding_dim))
        print(f'Number of parameters: {num_parameters}')

if __name__ == '__main__':
    unittest.main()
