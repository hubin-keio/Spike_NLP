"""Test PLM Trainer"""

import unittest
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
            embedding_dim = 24
            dropout=0.1
            max_len = 1500
            mask_prob = 0.15
            lr=0.0001

            tokenizer = ProteinTokenizer(max_len, mask_prob)
            embedder = NLPEmbedding(embedding_dim, max_len, dropout)

            vocab_size = len(tokenizer.token_to_index)
            padding_idx = tokenizer.token_to_index['<PAD>']
            hidden = embedding_dim
            n_transformer_layers = 12
            attn_heads = 12

            batch_size = 32
            num_workers = 1

            train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
            n_test_baches = 5

            trainer = PLM_Trainer(BERT, vocab_dict=vocab_dict, max_len=max_len,
                          embedding_dim=embedding_dim, n_transformer_layers=n_transformer_layers,
                          n_attn_heads=attn_heads, dropout=dropout,
                          mask_prob=mask_prob, batch_size=batch_size,
                          lr=lr)
    
    trainer.train(train_data = train_loader)
        s

if __name__ == '__main__':
    unittest.main()
