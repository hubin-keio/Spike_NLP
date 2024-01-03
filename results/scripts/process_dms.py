#!/usr/bin/env python
""" DMS Datasets Data Processing """

import os
import re
import pandas as pd
from split_data import split_csv

def label_to_seq(label: str) -> str:
    """ Generate sequence based on reference sequence and mutation label. """

    refseq = list("NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST")
    seq = refseq.copy()
    p = '([0-9]+)'
    if '_' in label:
        for mutcode in label.split('_'):
            [ori, pos, mut] = re.split(p, mutcode)
            pos = int(pos)-1    # use 0-based counting
            assert refseq[pos].upper() == ori
            seq[pos] = mut.upper()
        seq = ''.join(seq)
        return seq

    if label=='wildtype': return ''.join(seq)

    [ori, pos, mut] = re.split(p, label)
    pos = int(pos)-1    # use 0-based counting
    assert refseq[pos] == ori
    seq[pos] = mut.upper()
    seq = ''.join(seq)
    return seq

def process_data(input_csv, output_csv):
    """ 
    Process the DMS binding and expression datasets.
    - Drop NAs
    - Select only nonsynonymous variant classes
    - Select unique nonsynonymous mutations
        - For duplicate aa_substitutions, take the mean value of log10Ka or ML_meanF
    - Apply mutations to reference sequence
    """

    if  "binding" in input_csv:
        value_column = "log10Ka"
        print("Looking at DMS binding dataset.")
    elif "expression" in input_csv:
        value_column = "ML_meanF"
        print("Looking at DMS expression dataset.")

    full_df = pd.read_csv(input_csv, sep=',', header=0)
    row_count = len(full_df)
    print(f"Number of data points: {row_count}")

    # Remove rows where specified column is NA
    full_df = full_df.dropna(subset=[value_column]).reset_index(drop=True)
    print(f"Number of data points with na: {row_count-len(full_df)}")
    print(f"Number of data points left {len(full_df)}")
    
    # Count number of entries per variant class
    value_counts = full_df["variant_class"].value_counts()
    print(f"{value_counts}")
    # Filter out variant classes that are not nonsynonymous
    nonsynonymous_df = full_df[full_df['variant_class'].str.contains('nonsynonymous', case=False, na=False)]
    
    # Group by 'aa_substitutions' and calculate the mean of the specified value column for each group
    unique_nonsynonymous_df = nonsynonymous_df.groupby('aa_substitutions', as_index=False)[value_column].mean()
    # Merge dfs
    merged_df = pd.merge(unique_nonsynonymous_df, nonsynonymous_df.drop(columns=value_column), on='aa_substitutions', how='left')
    # Drop duplicate rows based on 'aa_substitutions'
    merged_df = merged_df.drop_duplicates(subset='aa_substitutions').reset_index(drop=True)
    # Count number of unique nonsynonymous mutations
    unique_nonsynonymous_mutations_counts = merged_df['aa_substitutions'].nunique()
    print(f"Number of unique nonsynonymous mutations: {unique_nonsynonymous_mutations_counts}")
    
    # Filter to only the columns we want and copy the DataFrame to avoid SettingWithCopyWarning
    unique_filtered_df = merged_df[["aa_substitutions", value_column]].copy()
    duplicate_filtered_df = nonsynonymous_df.copy()
    # Add '_' to substitutions
    unique_filtered_df.loc[:, "aa_substitutions"] = unique_filtered_df["aa_substitutions"].replace(' ', '_', regex=True)
    duplicate_filtered_df.loc[:, "aa_substitutions"] = duplicate_filtered_df["aa_substitutions"].replace(' ', '_', regex=True)
    # Utilize "aa_substitutions" to generate the mutated sequence
    unique_filtered_df['sequence'] = unique_filtered_df['aa_substitutions'].apply(label_to_seq)
    duplicate_filtered_df['sequence'] = duplicate_filtered_df['aa_substitutions'].apply(label_to_seq)
    # Change "aa_substitutions" to "labels"
    unique_filtered_df = unique_filtered_df.rename(columns={"aa_substitutions": "labels"})
    duplicate_filtered_df = duplicate_filtered_df.rename(columns={"aa_substitutions": "labels"})
    
    # Save to csv
    unique_output_csv = output_csv.replace("mutation", "unique_mutation")
    unique_filtered_df.to_csv(unique_output_csv, index=False)
    duplicate_output_csv = output_csv.replace("mutation", "duplicate_mutation")
    duplicate_filtered_df.to_csv(duplicate_output_csv, index=False)

    # Split to train, test data (80/20)
    rnd_seed = 0
    split_csv(rnd_seed, unique_output_csv)
    split_csv(rnd_seed, duplicate_output_csv)
    print("")

if __name__=="__main__":

    data_dir = os.path.join(os.path.dirname(__file__), 'dms')

    expression_input_csv = os.path.join(data_dir, "expression/expression_meanFs.csv")
    expression_output_csv = os.path.join(data_dir, "expression/mutation_expression_meanFs.csv")
    process_data(expression_input_csv, expression_output_csv)

    binding_input_csv = os.path.join(data_dir, "binding/binding_Kds.csv")
    binding_output_csv = os.path.join(data_dir, "binding/mutation_binding_Kds.csv")
    process_data(binding_input_csv, binding_output_csv)