"""Test database and SeqDataset"""

import random
import numpy
import unittest
import sqlite3
from os import path
import torch
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

    def test_resume_dataloader(self):


        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            numpy.random.seed(worker_seed)
            random.seed(worker_seed)

        num_workers = 2
        batch_size = 2
        n_test_baches = 5
        resume_after = 3
        
        gen = torch.Generator()
        gen.manual_seed(0)
        
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  worker_init_fn = seed_worker,
                                  generator=gen,
                                  drop_last=True
                                  )
        saved_state = gen.get_state()

        seq_id_a = seq_id_b = []
        for i, batch in enumerate(train_loader):
            if i >= n_test_baches:
                break
            [seq_ids, seqs] = batch
            seq_id_a.append(seq_ids)
        

        gen.set_state(saved_state)  # Expect the same sequence.
        for i, batch in enumerate(train_loader):
            if i >= n_test_baches:
                break
            [seq_ids, seqs] = batch
            seq_id_b.append(seq_ids)
            
        print(f'SeqIDs fresh: {seq_id_a}')
        print(f'SeqIDs from saved state: {seq_id_b}')
        self.assertEqual(seq_id_a, seq_id_b)


if __name__ == '__main__':
    unittest.main()
