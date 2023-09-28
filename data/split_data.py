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

def split_csv(rnd_seed: int, input_csv: str):
    """
    Split csv file into train, test data.
    """
    train_csv = input_csv.replace('.csv', '_train.csv')
    test_csv  = input_csv.replace('.csv', '_test.csv')

    with open(input_csv, "r") as input:
        reader = csv.reader(input)
        header = next(reader)
        input_records = list(reader)

    train_count = 0
    test_count = 0
    total_count = 0

    random.seed(rnd_seed)
    random.shuffle(input_records)

    with open(train_csv, 'w') as ft, open(test_csv, 'w') as fv:
        train_writer = csv.writer(ft)
        test_writer = csv.writer(fv)

        train_writer.writerow(header)
        test_writer.writerow(header)

        for record in input_records:
            rnd = random.random()
            if rnd > 0.2:
                train_writer.writerow(record)
                train_count += 1
            else:
                test_writer.writerow(record)                
                test_count += 1
            total_count += 1
    print(f'Total: {total_count}, Train: {train_count}, Test: {test_count}')

def sample_meta_csv(rnd_seed:int, train_csv_file: str, test_csv_file: str):
    """
    Randomly sample even parts of a fraction of the original data csv.
    Using this for the metadata to sample among variants.
    """

    train_df = pd.read_csv(train_csv_file, sep=',', header=0)
    test_df = pd.read_csv(test_csv_file, sep=',', header=0)
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    count_sequences_per_variant(combined_df)

    filtered_df = combined_df[combined_df['variant'].isin(["Alpha", "Delta", "Omicron"])]
    sampled_df = pd.concat([filtered_df[filtered_df['variant'] == variant].sample(n=400, random_state=rnd_seed)
                            for variant in ["Alpha", "Delta", "Omicron"]])

    save_as = train_csv_file.replace("_train_variant_seq.csv", f"_variant_seq_sampled_ADO_1200.csv")
    sampled_df.to_csv(save_as, index=False)
    count_sequences_per_variant(sampled_df)

def count_sequences_per_variant(input_df):
    """Count the number of sequences per variant and print the result."""
    
    # Use the 'variant' column to group and count the sequences per variant
    variant_counts = input_df['variant'].value_counts().reset_index()
    variant_counts.columns = ['Variant', 'Count']
    print(variant_counts)

if __name__ == '__main__':
    data_dir = os.path.dirname(__file__)
    rnd_seed = 0
    
    # RBD w/ metadata variants (sampling)
    # train_csv_file = os.path.join(data_dir, "spike_variants/rbd_train_variant_seq.csv")
    # test_csv_file = os.path.join(data_dir, "spike_variants/rbd_test_variant_seq.csv")
    # sample_meta_csv(rnd_seed, train_csv_file, test_csv_file)

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

    # DMS
    # input_csv = os.path.join(data_dir, 'dms/mutation_binding_Kds.csv')
    # split_csv(rnd_seed, input_csv)

    # AlphaSeq
    # input_csv = os.path.join(data_dir, 'alphaseq/clean_avg_alpha_seq.csv')
    # split_csv(rnd_seed, input_csv)

    # RBD
    # input_fa = os.path.join(data_dir, 'spike/spikeprot0528.clean.uniq.noX.RBD.fasta')
    # split_fasta(rnd_seed, input_fa)
