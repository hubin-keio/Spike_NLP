"""Test Language Model"""

import unittest
from pnlp.embedding.tokenizer import ProteinTokenizer
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

        self.vocab_size = len(self.tokenizer.token_to_index)
        self.padding_idx = self.tokenizer.token_to_index['<PAD>']
        self.hidden = self.embedding_dim
        self.n_transformer_layers = 12
        self.attn_heads = 12

        self.batch_seqs = []
        with open('test_spike_seq.txt', 'r') as fh:
            for seq in fh.readlines():
                self.batch_seqs.append(seq.rstrip())
        self.longest = max([len(seq) for seq in self.batch_seqs])

    def test_plm_forward(self):
        bert = BERT(self.embedding_dim,
                    self.dropout,
                    self.max_len,
                    self.mask_prob,
                    self.hidden,
                    self.n_transformer_layers,
                    self.attn_heads)

        plm = ProteinLM(bert, self.vocab_size)
        tokenized_seqs, masked_idx = self.tokenizer.get_token(self.batch_seqs)
        mask_tensor = tokenized_seqs == self.padding_idx
        output = plm.forward(tokenized_seqs, mask_tensor)
        num_parameters = sum(p.numel() for p in plm.parameters() if p.requires_grad)
        self.assertEqual(output.size(), (len(self.batch_seqs),
                                         min(self.longest, self.max_len),
                                         self.vocab_size))
        print(f'Number of parameters: {num_parameters}')
        print(f'Mask tensor shape: {mask_tensor.shape}')
        print(f'Total masks==True: {(mask_tensor == 1).sum().item()}')



if __name__ == '__main__':
    unittest.main()
