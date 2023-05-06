"""Test PLM Trainer"""

import unittest
import os.path as path

import torch
from torch.utils.data import DataLoader

from pnlp.db.dataset import SeqDataset
from pnlp.embedding.tokenizer import token_to_index
from pnlp.embedding.tokenizer import ProteinTokenizer
from pnlp.embedding.nlp_embedding import NLPEmbedding
from pnlp.trainer.plm_trainer import PLM_Trainer, ScheduledOptim


class TestTrainer(unittest.TestCase):
    def setUp(self):
        db_file = path.abspath(path.dirname(__file__))
        db_file = path.join(db_file, '../data/SARS_CoV_2_spike.db')
        self.train_dataset = SeqDataset(db_file, "train")
        print(f'Test training process using sequence db file: {db_file}')

    def test_plm_trainer(self):
        embedding_dim = 36
        dropout=0.1
        max_len = 1500
        mask_prob = 0.15
        n_transformer_layers = 12
        attn_heads = 12

        batch_size = 10
        lr = 0.0001
        weight_decay = 0.01
        warmup_steps = 1000

        betas=(0.9, 0.999)
        tokenizer = ProteinTokenizer(max_len, mask_prob)
        embedder = NLPEmbedding(embedding_dim, max_len,dropout)

        vocab_size = len(token_to_index)
        hidden = embedding_dim

        num_epochs = 2
        num_workers = 1
        n_test_baches = 5

        USE_GPU = True
        SAVE_MODEL = True
        device = torch.device("cuda:0" if torch.cuda.is_available() and USE_GPU else "cpu")
        print(f'\nUsing device: {device}')


        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


        trainer = PLM_Trainer(SAVE_MODEL, vocab_size, embedding_dim=embedding_dim,
                              dropout=dropout, max_len=max_len,
                              mask_prob=mask_prob, n_transformer_layers=n_transformer_layers,
                              n_attn_heads=attn_heads, batch_size=batch_size, lr=lr, betas=betas,
                              weight_decay=weight_decay, warmup_steps=warmup_steps, device=device)
        # trainer.print_model_params()
 

if __name__ == '__main__':
    unittest.main()
