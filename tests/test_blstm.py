"""Test BLSTM Model"""

import os
import torch
import unittest
from pnlp.embedding.tokenizer import ProteinTokenizer, token_to_index, index_to_token, PADDING_IDX
from pnlp.embedding.nlp_embedding import NLPEmbedding
from pnlp.model.bert import BERT
from pnlp.model.language import ProteinLM, ProteinMaskedLanguageModel

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.pnlp.model.blstm import BLSTM

class test_BLSTM(unittest.TestCase):
    def setUp(self):
        self.lstm_input_size = 320
        self.lstm_hidden_size = 320
        self.lstm_num_layers = 1
        self.lstm_bidirectional = True
        self.fcn_hidden_size = 320

        self.embedding_dim = 320
        self.dropout= 0.1
        self.max_len = 280
        self.mask_prob = 0.15
        self.n_transformer_layers = 12
        self.attn_heads = 12

        self.tokenizer = ProteinTokenizer(self.max_len, self.mask_prob)
        self.embedder = NLPEmbedding(self.embedding_dim, self.max_len, self.dropout)

        self.batch_scores, self.batch_seqs = [], []
        test_seq = os.path.abspath(os.path.dirname(__file__))
        test_seq = os.path.join(test_seq, 'test_dms_binding_seq.txt')

        with open(test_seq, 'r') as fh:
            for line in fh.readlines():
                line = line.split(',')
                score, seq = line[0], line[1]
                self.batch_scores.append(float(score.rstrip()))
                self.batch_seqs.append(seq.rstrip())
        self.longest = max([len(seq) for seq in self.batch_seqs])

    def test_BLSTM_forward(self):
        """ Test the forwad funciton in the BLSTM model. """
        model = BLSTM(self.lstm_input_size,
                      self.lstm_hidden_size,
                      self.lstm_num_layers,
                      self.lstm_bidirectional,
                      self.fcn_hidden_size)

        tokenized_seqs = self.tokenizer(self.batch_seqs)
        embedded_seqs, _ = self.embedder(tokenized_seqs)
        output = model(embedded_seqs) # output is a prediction, which is a singular value

        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(output.size(), (len(self.batch_seqs), 1)) # singular value is why this is 1
        print(f'Number of parameters: {num_parameters}')
        print(f'Output shape: {output.shape}')
    
    def test_mse_error(self):
        """ Test mse error. """
        model = BLSTM(self.lstm_input_size,
                      self.lstm_hidden_size,
                      self.lstm_num_layers,
                      self.lstm_bidirectional,
                      self.fcn_hidden_size)

        loss_fn = torch.nn.MSELoss(reduction='sum')
        tokenized_seqs = self.tokenizer(self.batch_seqs)
        embedded_seqs, _ = self.embedder(tokenized_seqs)
        output = model(embedded_seqs).flatten() # output is a prediction, which is a singular value
    
        loss = loss_fn(output,torch.tensor(self.batch_scores))
        print(f'MSE loss: {loss}')

if __name__ == '__main__':
    unittest.main()
