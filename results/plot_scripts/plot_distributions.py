#!/usr/bin/env python
"""
Variant Distribution

Bash code used to extract the Accession IDs and Variants from 
larger meta data file:
    cat spikeprot0528.clean.uniq.noX.RBD.metadata.tsv | cut -f 5,16 | awk '{print $1, $3}' > spikeprot0528.clean.uniq.noX.RBD.metadata_variants.txt

> Extracts the Accession IDs that have Variants from the resulting
  txt file of the above command. 

> Then goes through the training and testing database to see which 
  Accession IDs with variant labels are within each database. 

> Resulting Accession IDs with no variant labels are kept track of 
  in a csv. 

> Those with variant labels are also kept track of in a csv, but 
  plotted to show distribution of the variants among Accession IDs 
  within the training and testing databases. 

Other distribution plots included, such as 
    - amino acid distribution (related to the above desc), 
    - sequence length distribution,
    - variant distribution of the highest frequency sequence length.
"""
import os
import tqdm
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader
from pnlp.db.dataset import SeqDataset, initialize_db

def extract_variant(file_name:str) -> tuple:
    """
    Create and return a DataFrame from given meta data file that contains:
        - seq_id
        - variant
    If the seq_id had no variant, it was dropped from the dataframe.
    """
    meta_data_df = pd.read_csv(file_name, sep=' ', header=0)
    meta_data_df.columns = ["seq_id", "variant"]

    # Filter out rows that have NaN values in the 'variant' column vs not
    clean_meta_data_df = meta_data_df.dropna(subset=['variant'])
    clean_meta_data_df.reset_index(drop=True, inplace=True)

    dropped_meta_data_df = meta_data_df[meta_data_df['variant'].isna()]
    dropped_meta_data_df.reset_index(drop=True, inplace=True)

    return clean_meta_data_df, dropped_meta_data_df

def add_variants(seq_data:SeqDataset, meta_data_df, save_as:str, mode:str) -> tuple:
    """
    Mode: train or test
    """
    variant_file_name = os.path.join(save_as, f'spike_variants/rbd_{mode}_variant_seq.csv') 
    no_variant_file_name = os.path.join(save_as, f'spike_variants/rbd_{mode}_no_variant_seq.csv') 

    # Set the tqdm progress bar
    data_iter = tqdm.tqdm(enumerate(seq_data),
                            desc=f'{mode}',
                            total = len(seq_data),
                            bar_format='{l_bar}{r_bar}')

    with open(variant_file_name, "w") as fv, open(no_variant_file_name, "w") as fnv:
        fv.write(f"seq_id,variant,sequence\n")
        fnv.write(f"seq_id,sequence\n")

        for i, batch_data in data_iter:
            seq_ids, seqs = batch_data

            for seq_id, seq in zip(seq_ids, seqs):
                if seq_id in meta_data_df['seq_id'].values:
                    variant = meta_data_df.loc[meta_data_df['seq_id'] == seq_id, 'variant'].values[0]
                    fv.write(f"{seq_id},{variant},{seq}\n")
                    fv.flush()
                else: 
                    fnv.write(f"{seq_id},{seq}\n")
                    fv.flush()
    
    return variant_file_name, no_variant_file_name

def plot_variant_distribution(training_csv_file:str, testing_csv_file:str, save_as:str):
    distribution_file_name = os.path.join(save_as, f'rbd_variant_percentage_distribution.png')

    training_df = pd.read_csv(training_csv_file, sep=',', header=0)
    testing_df = pd.read_csv(testing_csv_file, sep=',', header=0)

    # Get unique variants from both training and testing datasets
    all_variants = sorted(np.unique(pd.concat([training_df['variant'], testing_df['variant']])))

    # Calculate variant percentages for training and testing data
    training_variant_percentages = [(training_df['variant'] == variant).mean() * 100 for variant in all_variants]
    testing_variant_percentages = [(testing_df['variant'] == variant).mean() * 100 for variant in all_variants]

    # Create a grouped bar plot to visualize the distribution
    plt.figure(figsize=(16, 8))

    bar_width= 0.45
    x = np.arange(len(all_variants))
    train_bars = plt.bar(x - bar_width/2, training_variant_percentages, width=bar_width, color="lightgrey", label='Training Data')
    test_bars = plt.bar(x + bar_width/2, testing_variant_percentages, width=bar_width, color="dimgrey", label='Testing Data')

    plt.title('Training and Testing Variant Distribution')
    plt.xlabel('Variant')
    plt.ylabel('Quantity (%)')
    plt.xticks(x, all_variants)
    plt.legend()
    
    # Add bar labels
    for i, (train_percentage, test_percentage) in enumerate(zip(training_variant_percentages, testing_variant_percentages)):
        plt.text(x[i] - bar_width/2, train_percentage, f'{train_percentage:.2f}%', ha='center', va='bottom', fontsize=8)
        plt.text(x[i] + bar_width/2, test_percentage, f'{test_percentage:.2f}%', ha='center', va='bottom', fontsize=8)

    plt.style.use('ggplot')
    plt.tight_layout()
    plt.savefig(distribution_file_name)

def plot_aa_distribution(training_csv_file:str, testing_csv_file:str, save_as:str):
    distribution_file_name = os.path.join(save_as, f'rbd_variant_aa_seq_percentage_distribution.png')
    aa_dict = defaultdict(int)
    m_seqs = defaultdict(list)

    # Combine the training and testing DataFrames
    training_df = pd.read_csv(training_csv_file, sep=',', header=0)
    testing_df = pd.read_csv(testing_csv_file, sep=',', header=0)
    combined_df = pd.concat([training_df, testing_df], ignore_index=True)

    for seq_id, seq, variant in zip(combined_df['seq_id'], combined_df['sequence'], combined_df['variant']):
        m_count = 0
        for aa in seq:
            aa_dict[aa] += 1
            if aa == "M":
                m_count += 1

        if "M" in seq:
            if variant not in m_seqs:
                m_seqs[variant] = [0, 0] 

            if m_count == 1:
                m_seqs[variant][0] += 1
            elif m_count > 1:
                m_seqs[variant][1] += 1

    n_m_seqs = sum([sum(counts) for counts in m_seqs.values()])
    print(f"M count: {aa_dict['M']} AAs, n seqs with M: {n_m_seqs} (out of {len(combined_df)}), % seqs with M: {n_m_seqs/len(combined_df)*100:.2f}%")
    print(m_seqs)

    amino_acids = sorted(aa_dict.keys())
    counts = [aa_dict[aa] for aa in amino_acids]
    total_counts = sum(counts)
    percentages = [(count/total_counts) * 100 for count in counts]

    plt.figure(figsize=(14, 8))

    x = np.arange(len(aa_dict))
    bar_width = 0.45
    plt.bar(x, percentages, width=bar_width, color="dimgrey", label='Amino Acid Distribution')

    plt.title('Amino Acid Distribution')
    plt.xlabel('Amino Acid')
    plt.ylabel('Quantity (%)')
    plt.xticks(x, amino_acids)

    # Add bar labels
    for i, percentage in enumerate(percentages):
        plt.text(i, percentage, f'{percentage:.3f}%', ha='center', va='bottom', fontsize=10)

    plt.grid(False)
    plt.gca().set_facecolor('none')
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_color('black')  

    plt.style.use('ggplot')
    plt.tight_layout()
    plt.savefig(distribution_file_name)

def plot_seq_length_distibution(csv_file:str):
    """
    Plots the sequence length distribution %s of a given csv file that has
    at least these 2 columns:
        - sequence: full sequence
        - variant: variant name
    """
    full_df = pd.read_csv(csv_file, sep=',', header=0)
    full_df['seq_length'] = full_df['sequence'].apply(len)

    # Calculate % of total 
    counts = full_df.groupby(['seq_length', 'variant']).size().reset_index(name='counts')
    total_counts = counts.groupby('seq_length')['counts'].sum().reset_index(name='total_counts')
    counts = pd.merge(counts, total_counts, on='seq_length')
    counts['percentage'] = round((counts['counts'] / counts['total_counts']) * 100, 3)
    
    # Calculate the frequency of each sequence length
    freq = full_df['seq_length'].value_counts()
    freq_perc = ((freq / freq.sum()) * 100).sort_index()

    # Plotting with broken axis
    # Create two subplots to simulate a broken y-axis effect
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    # Plot the first subplot for smaller values
    ax1.bar(freq_perc.index, freq_perc.values)
    ax1.set_ylim(1, 100)  # Adjust these values as per your specific dataset

    # Plot the second subplot for larger values
    ax2.bar(freq_perc.index, freq_perc.values)
    ax2.set_ylim(0, 0.08)  # Adjust these values as per your specific dataset

    # Hide the spines between the subplots
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)
    ax2.xaxis.tick_bottom()

    # Highlighting the bar with the highest frequency
    max_value = freq_perc.max()
    max_index = freq_perc.idxmax()
    ax1.annotate(f'{max_value:.2f}%', 
                 xy=(max_index, max_value), 
                 xytext=(max_index - 10, max_value - 10), 
                 arrowprops=dict(facecolor='black', arrowstyle='-'),
                 fontsize=8, color='black')

    plt.suptitle('Sequence Length Distribution %', fontsize=16)
    plt.xlabel('Sequence Length')
    f.text(0.04, 0.5, 'Percentage', va='center', rotation='vertical')

    save_as = csv_file.replace('.csv', '_seq_length_dist.png')
    plt.savefig(save_as)

    # Calling the plot_most_seq_variant_distribution function with the highest frequency sequence length
    plot_most_seq_variant_distribution(csv_file, False, full_df, max_index)
    plot_most_seq_variant_distribution(csv_file, True, full_df, max_index)

def plot_most_seq_variant_distribution(csv_file: str, ado: bool, full_df, seq_length: int):    
    """
    Called by 'plot_seq_length_distibution'. Plots the variant distribution
    of the sequence with the highest frequency sequence length occurence.
    If 'ado' is called, the data is filtered to 'Alpha', 'Delta', 'Omicron'
    variants.
    """
    if ado:
        # Filter the DataFrame to keep only the selected variants
        selected_variants = ['Alpha', 'Omicron', 'Delta']
        full_df = full_df[full_df['variant'].isin(selected_variants)]

    counts = full_df[full_df['seq_length'] == seq_length].groupby(['variant']).size().reset_index(name='counts')
    total_count = counts['counts'].sum()
    counts['percentage'] = (counts['counts'] / total_count) * 100

    # Plotting
    plt.figure(figsize=(10, 6))
    bars = sns.barplot(data=counts, x='variant', y='percentage', palette="viridis")
    
    # Adding labels on top of the bars
    for p in bars.patches:
        bars.annotate(f'{p.get_height():.2f}%', 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha = 'center', va = 'baseline', 
                     xytext = (0, 4), 
                     textcoords = 'offset points')

    plt.title(f'Variant Distribution Percentage for Sequence Length {seq_length}')
    plt.xlabel('Variant')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_as = csv_file.replace('.csv', f'_variant_{"ado_" if ado else ""}distribution_{seq_length}.png')
    plt.savefig(save_as)

def calc_num_seq_ids(file_name:str):
    line_count = 0

    with open(file_name, "r") as fh:
        header = fh.readline()

        for line in fh:
            line_count += 1
    
    return line_count

if __name__ == '__main__':

    data_dir = os.path.join(os.path.dirname(__file__), '../../../data')
    results_dir = os.path.join(os.path.dirname(__file__), '../../../results/plot_results')

    # Sequence length dist plotting on full variant csv, no test/train split
    full_csv =  os.path.join(data_dir, "spike_variants/spikeprot0528.clean.uniq.noX.RBD_variants.csv")
    plot_variant_distribution(results_dir)
    plot_aa_distribution(results_dir)
    # plot_seq_length_distibution(full_csv)

    # BELOW: code that runs the description above, sequence length dist is separate
    # # Data loader
    # db_file = os.path.join(data_dir, 'SARS_CoV_2_spike_noX_RBD.db')
    # train_dataset = SeqDataset(db_file, "train")
    # test_dataset = SeqDataset(db_file, "test")

    # # Meta data
    # meta_data_file = os.path.join(data_dir, 'spike/spikeprot0528.clean.uniq.noX.RBD.metadata_variants.txt')  

    # # Extract data that includes variants
    # meta_data_df, dropped_meta_data_df = extract_variant(meta_data_file)
    # print(f"Number of seq_ids dropped due to NaN in 'variant' column: {len(dropped_meta_data_df)}")
    # print(f"Number of seq_ids left post NaN removal in 'variant' column: {len(meta_data_df)}")

    # # Combine variants with training and testing data
    # batch_size = 64
    # torch.manual_seed(0)
    # train_seq_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # test_seq_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
   
    # train_variant_csv, train_no_variant_csv = add_variants(train_seq_loader, meta_data_df, data_dir, "train")
    # test_variant_csv, test_no_variant_csv = add_variants(test_seq_loader, meta_data_df, data_dir, "test")
    # print(f"Number of seq_ids in training database: (w/ variant label):{calc_num_seq_ids(train_variant_csv)}, (w/o variant label):{calc_num_seq_ids(train_no_variant_csv)} ")
    # print(f"Number of seq_ids in testing database: (w/ variant label):{calc_num_seq_ids(test_variant_csv)}, (w/o variant label):{calc_num_seq_ids(test_no_variant_csv)} ")

    # Make variant distribution plot
    # train_variant_csv = os.path.join(results_dir, "rbd_train_variant_seq.csv")
    # test_variant_csv = os.path.join(results_dir, "rbd_test_variant_seq.csv")
    # plot_variant_distribution(train_variant_csv, test_variant_csv, results_dir)
    # plot_aa_distribution(train_variant_csv, test_variant_csv, results_dir)


