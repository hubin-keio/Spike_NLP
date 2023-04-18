"""Test Language Model"""

import unittest
from pnlp.embedding.nlp_embedding import NLPEmbedding
from pnlp.model.bert import BERT
from pnlp.model.language import ProteinLM, ProteinMaskedLanguageModel

class test_PLM(unittest.TestCase):
    def setUp(self):
        self.embedding_dim = 24
        self.dropout=0.1
        self.max_len = 500
        self.mask_prob = 0.15

        embedder = NLPEmbedding(self.embedding_dim, self.dropout, self.max_len, self.mask_prob)
        self.vocab_size = len(embedder.token_to_index)
        self.padding_idx = embedder.token_to_index['<PAD>']
        self.hidden = self.embedding_dim
        self.n_transformer_layers = 12
        self.attn_heads = 12

        self.batch_seqs = []
        with open('test_spike_seq.txt', 'r') as fh:
            for seq in fh.readlines():
                self.batch_seqs.append(seq.rstrip())

    def test_plm_forward(self):
        bert = BERT(self.embedding_dim,
                    self.dropout,
                    self.max_len,
                    self.mask_prob,
                    self.hidden,
                    self.n_transformer_layers,
                    self.attn_heads)

        plm = ProteinLM(bert, self.vocab_size)
        output = plm.forward(self.batch_seqs)
        num_parameters = sum(p.numel() for p in plm.parameters() if p.requires_grad)
        self.assertEqual(output.size(), (len(self.batch_seqs), self.max_len, self.vocab_size))
        print(f'Number of parameters: {num_parameters}')
        

if __name__ == '__main__':
    unittest.main()

