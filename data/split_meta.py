#!/usr/bin/env python
import os
import random
import pandas as pd

def sample_csv(training_csv_file: str, testing_csv_file: str):
    """Randomly sample even parts of a fraction of the original data csv"""

    training_df = pd.read_csv(training_csv_file, sep=',', header=0)
    testing_df = pd.read_csv(testing_csv_file, sep=',', header=0)
    combined_df = pd.concat([training_df, testing_df], ignore_index=True)
    count_sequences_per_variant(combined_df)

    filtered_df = combined_df[combined_df['variant'].isin(["Alpha", "Delta", "Omicron"])]
    sampled_df = pd.concat([filtered_df[filtered_df['variant'] == variant].sample(n=400, random_state=0)
                            for variant in ["Alpha", "Delta", "Omicron"]])

    save_as = training_csv_file.replace("_train_variant_seq.csv", f"_variant_seq_sampled_ADO_1200.csv")
    sampled_df.to_csv(save_as, index=False)
    count_sequences_per_variant(sampled_df)

def count_sequences_per_variant(input_df):
    """Count the number of sequences per variant and print the result."""
    
    # Use the 'variant' column to group and count the sequences per variant
    variant_counts = input_df['variant'].value_counts().reset_index()
    variant_counts.columns = ['Variant', 'Count']
    
    # Print the table of variant counts
    print(variant_counts)

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), '../results/plot_results')
    training_csv_file = os.path.join(data_dir, "rbd_train_variant_seq.csv")
    testing_csv_file = os.path.join(data_dir, "rbd_test_variant_seq.csv")

    sample_csv(training_csv_file, testing_csv_file)
