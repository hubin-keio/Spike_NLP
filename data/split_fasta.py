"""Randomly split a fasta file into two files"""

import os
import random
from Bio import SeqIO

def split_fasta(rnd_seed: int, input_fa: str, train_fa: str, test_fa: str):

    train_count = 0
    test_count = 0
    total_count = 0

    input_records = list(SeqIO.parse(input_fa, 'fasta'))

    random.seed(rnd_seed)
    random.shuffle(input_records)

    with open(train_fa, 'w') as ft, open(test_fa, 'w') as fv:

        for record in input_records:
            rnd = random.random()
            if rnd > 0.2:
                SeqIO.write(record, ft, 'fasta')
                train_count += 1
            else:
                SeqIO.write(record, fv, 'fasta')
                test_count += 1
            total_count += 1
    print(f'Total: {total_count}, Train: {train_count}, Test: {test_count}')


if __name__ == '__main__':
    root_dir = os.path.dirname(__file__)
    input_fa = os.path.join(root_dir, 'spikeprot0203.clean.uniq.noX.RBD.fasta')
    train_fa = input_fa.replace('.fasta', '_train.fasta')
    test_fa  = input_fa.replace('.fasta', '_test.fasta')

    rnd_seed = 0
    split_fasta(rnd_seed, input_fa, train_fa, test_fa)
