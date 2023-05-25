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
        """Test the forward function in the language model"""
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

        num_parameters = sum(p.numel() for p in plm.parameters() if p.requires_grad)
        self.assertEqual(output.size(), (len(self.batch_seqs), min(self.longest, max_len), self.vocab_size))
        print(f'Number of parameters: {num_parameters}')
        print(f'Output shape: {output.shape}')


    def test_cross_entropy_error(self):
        """Test cross entropy error"""
        max_len = 50
        bert = BERT(self.embedding_dim,
                    self.dropout,
                    max_len,
                    self.mask_prob,
                    self.n_transformer_layers,
                    self.attn_heads)

        criterion = torch.nn.CrossEntropyLoss()
        softmax = torch.nn.Softmax(dim=-1)
        plm = ProteinLM(bert, self.vocab_size)
        tokenized_seqs, masked_idx = self.tokenizer(self.batch_seqs)
        tokenized_seqs = tokenized_seqs[:, :max_len]

        input_unmasked_token_index = self.tokenizer._batch_pad(self.batch_seqs)
        input_unmasked_token_index = input_unmasked_token_index[:, :max_len]
        print(f'Input sequence shape: {input_unmasked_token_index.shape}')
        
        logits = plm.forward(tokenized_seqs)
        logits = logits.reshape(logits.shape[0], self.vocab_size, -1)
        print(f'Model output shape: {logits.shape}')
        
        loss = criterion(logits, input_unmasked_token_index)

        print(f'Cross entropy loss: {loss}')

        # for seq_idx in range(5):
        #     assert len(self.batch_seqs[seq_idx]) >= max_len
        #     input_seq = self.batch_seqs[seq_idx][:max_len]
        #     input_unmasked_token_index  = self.tokenizer._batch_pad(input_seq)


        #     # print(f'output token index shape: {output_token_index.shape}')
        #     # output_token_index = [x.item() for x in output_token_index]
        #     # output_seq = ''.join([index_to_token[x.item()] for x in output_token_index.view(-1)])
        #     print(f'Shape of logits: {logits[seq_idx].shape}')
        #     # print(f'Shape of input: {input_unmasked_token_index[seq_idx].reshape(,1).shape}')
        #     print(input_unmasked_token_index[seq_idx])
        #     loss = criterion(logits[seq_idx], input_unmasked_token_index[seq_idx])

        #     # print(f'\nTest {seq_idx}')
        #     # print(f'Input seq: {input_seq}')
        #     # print(f'Output seq: {output_seq}')
        #     # print(f'Unmasked input tokens: {input_unmasked_token_index}')
        #     # print(f'Output tokens: {output_token_index}')
        #     print(f'CrossEntropyLoss: {loss}')


if __name__ == '__main__':
    unittest.main()
