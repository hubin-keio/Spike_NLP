"""Test database and SeqDataset"""

import unittest
import sqlite3
from os import path
from torch.utils.data import DataLoader
from pnlp.db.dataset import SeqDataset


class TestSeqDataset(unittest.TestCase):
    def setUp(self):
        db_file = path.abspath(path.dirname(__file__))
        db_file = path.join(db_file, '../data/SARS_CoV_2_spike_noX.db')
        self.train_dataset = SeqDataset(db_file, "train")
        print(f'Sequence db file: {db_file}')
        print(f'Total seqs in training set: {len(self.train_dataset)}')

    def test_SeqDataset(self):
        batch_size = 5
        num_workers = 1

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        n_test_baches = 5
        for i, batch in enumerate(train_loader):
            if i >= n_test_baches:
                break
            print(f'\nbatch {i}')
            [seq_ids, seqs] = batch
            print(seq_ids)

if __name__ == '__main__':
    unittest.main()
