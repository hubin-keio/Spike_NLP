#!/usr/bin/env python
""" Randomly split data files into multiple files. """

import os
import csv
import random
import pandas as pd
from Bio import SeqIO

def split_fasta(rnd_seed: int, input_fa: str):
    """
    Split fasta file into train, test data.
    """
    train_fa = input_fa.replace('.fasta', '_train.fasta')
    test_fa  = input_fa.replace('.fasta', '_test.fasta')

    input_records = list(SeqIO.parse(input_fa, 'fasta'))

    random.seed(rnd_seed)
    random.shuffle(input_records)

    split_idx = int(0.8 * len(input_records))
    train_records = input_records[:split_idx]
    test_records = input_records[split_idx:]

    with open(train_fa, 'w') as ft, open(test_fa, 'w') as fv:
        SeqIO.write(train_records, ft, 'fasta')
        SeqIO.write(test_records, fv, 'fasta')

    print(f'Total: {len(input_records)}, Train: {len(train_records)}, Test: {len(test_records)}')

def split_csv(rnd_seed: int, input_csv: str):
    """
    Split csv file into train, test data.
    """
    train_csv = input_csv.replace('.csv', '_train.csv')
    test_csv  = input_csv.replace('.csv', '_test.csv')

    with open(input_csv, "r") as input_file:
        reader = csv.reader(input_file)
        header = next(reader)
        input_records = list(reader)

    random.seed(rnd_seed)
    random.shuffle(input_records)

    split_idx = int(0.8 * len(input_records))
    train_records = input_records[:split_idx]
    test_records = input_records[split_idx:]

    with open(train_csv, 'w') as ft, open(test_csv, 'w') as fv:
        train_writer = csv.writer(ft)
        test_writer = csv.writer(fv)

        train_writer.writerow(header)
        test_writer.writerow(header)

        for record in train_records:
            train_writer.writerow(record)

        for record in test_records:
            test_writer.writerow(record)

    print(f'Total: {len(input_records)}, Train: {len(train_records)}, Test: {len(test_records)}')

def count_sequences_per_variant(input_df):
    """Count the number of sequences per variant and print the result."""
    
    # Use the 'variant' column to group and count the sequences per variant
    variant_counts = input_df['variant'].value_counts().reset_index()
    variant_counts.columns = ['Variant', 'Count']
    print(variant_counts)

if __name__ == '__main__':
    data_dir = os.path.dirname(__file__)
    rnd_seed = 0
    
    # RBD w/ metadata variants
    # (new split, old data may no longer retain 80/20 split after removal of entries w/ no variant label)
    train_csv_file = os.path.join(data_dir, "spike_variants/rbd_train_variant_seq.csv")
    test_csv_file = os.path.join(data_dir, "spike_variants/rbd_test_variant_seq.csv")
    train_df = pd.read_csv(train_csv_file, sep=',', header=0)
    test_df = pd.read_csv(test_csv_file, sep=',', header=0)
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    input_csv = os.path.join(data_dir, "spike_variants/spikeprot0528.clean.uniq.noX.RBD_variants.csv")
    combined_df.to_csv(input_csv, index=False)
    split_csv(rnd_seed, input_csv)

    # AlphaSeq
    input_csv = os.path.join(data_dir, 'alphaseq/clean_avg_alpha_seq.csv')
    full_df = pd.read_csv(input_csv, sep=',', header=0)
    selected_df = full_df[['POI', 'Sequence', 'Mean_Affinity']]
    selected_csv = input_csv.replace(".csv", "_selected.csv")
    selected_df.to_csv(selected_csv, index=False)
    split_csv(rnd_seed, selected_csv)

    # RBD
    input_fa = os.path.join(data_dir, 'spike/spikeprot0528.clean.uniq.noX.RBD.fasta')
    split_fasta(rnd_seed, input_fa)

