#!/usr/bin/env python

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_variant_distribution(csv_file:str, save_as:str):
    distribution_file_name = os.path.join(save_as, f'rbd_variant_percentage_distribution')
    df = pd.read_csv(csv_file, sep=',', header=0)
    all_variants = sorted(np.unique(df['variant']))
    variant_percentages = [(df['variant'] == variant).mean() * 100 for variant in all_variants]

    # Create a bar plot to visualize the distribution
    sns.set_theme()
    sns.set_context('talk')
    sns.set(style='white')
    plt.figure(figsize=(16, 9))
    plt.rcParams['font.family'] = 'sans-serif'

    bar_width= 0.45
    x = np.arange(len(all_variants))
    bars = plt.bar(x, variant_percentages, width=bar_width, color="dimgrey")

    # Add bar labels
    for i, percentage in enumerate(variant_percentages):
        plt.text(x[i], percentage, f'{percentage:.2f}%', ha='center', va='bottom', fontsize=10)

    # Remove grey grid background
    plt.grid(False)
    plt.gca().set_facecolor('none')
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_color('black')
    
    #plt.title('Variant Distribution')
    plt.xlabel('Variant')
    plt.ylabel('Quantity (%)')
    plt.xticks(x, all_variants)
    plt.style.use('ggplot')
    plt.tight_layout()
    plt.savefig(distribution_file_name+'.png', format='png')
    plt.savefig(distribution_file_name+'.pdf', format='pdf')    
    return distribution_file_name

def plot_split_variant_distribution(training_csv_file:str, testing_csv_file:str, save_as:str):
    distribution_file_name = os.path.join(save_as, f'rbd_variant_split_percentage_distribution')
    training_df = pd.read_csv(training_csv_file, sep=',', header=0)
    testing_df = pd.read_csv(testing_csv_file, sep=',', header=0)
    all_variants = sorted(np.unique(pd.concat([training_df['variant'], testing_df['variant']])))
    training_variant_percentages = [(training_df['variant'] == variant).mean() * 100 for variant in all_variants]
    testing_variant_percentages = [(testing_df['variant'] == variant).mean() * 100 for variant in all_variants]

    # Create a grouped bar plot to visualize the distribution
    sns.set_theme()
    sns.set_context('talk')
    sns.set(style='white')
    plt.figure(figsize=(16, 9))
    plt.rcParams['font.family'] = 'sans-serif'

    bar_width= 0.45
    x = np.arange(len(all_variants))
    train_bars = plt.bar(x - bar_width/2, training_variant_percentages, width=bar_width, color="lightgrey", label='Training Data')
    test_bars = plt.bar(x + bar_width/2, testing_variant_percentages, width=bar_width, color="dimgrey", label='Testing Data')

    # Add bar labels
    for i, (train_percentage, test_percentage) in enumerate(zip(training_variant_percentages, testing_variant_percentages)):
        plt.text(x[i] - bar_width/2, train_percentage, f'{train_percentage:.2f}%', ha='center', va='bottom', rotation=30, fontsize=10)
        plt.text(x[i] + bar_width/2, test_percentage, f'{test_percentage:.2f}%', ha='center', va='bottom', rotation=30, fontsize=10)

    # Remove grey grid background
    plt.grid(False)
    plt.gca().set_facecolor('none')
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_color('black')
    
    #plt.title('Training and Testing Variant Distribution')
    plt.xlabel('Variant')
    plt.ylabel('Quantity (%)')
    plt.xticks(x, all_variants)
    plt.legend()
    plt.style.use('ggplot')
    plt.tight_layout()
    plt.savefig(distribution_file_name+'.png', format='png')
    plt.savefig(distribution_file_name+'.pdf', format='pdf')
    return distribution_file_name

def plot_aa_distribution(csv_file:str, save_as:str):
    distribution_file_name = os.path.join(save_as, f'rbd_variant_aa_percentage_distribution')
    aa_dict = defaultdict(int)
    m_seqs = defaultdict(list)

    df = pd.read_csv(csv_file, sep=',', header=0)

    for seq_id, variant, seq in zip(df['seq_id'], df['variant'], df['sequence']):
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
    print(f"M count: {aa_dict['M']} AAs, n seqs with M: {n_m_seqs} (out of {len(df)}), % seqs with M: {n_m_seqs/len(df)*100:.2f}%")
    print(m_seqs)

    amino_acids = sorted(aa_dict.keys())
    counts = [aa_dict[aa] for aa in amino_acids]
    total_counts = sum(counts)
    percentages = [(count/total_counts) * 100 for count in counts]

    # Create a bar plot to visualize the distribution
    sns.set_theme()
    sns.set_context('talk')
    sns.set(style='white')
    plt.figure(figsize=(16, 9))
    plt.rcParams['font.family'] = 'sans-serif'

    x = np.arange(len(aa_dict))
    bar_width = 0.45
    plt.bar(x, percentages, width=bar_width, color="dimgrey")

    # Add bar labels
    for i, percentage in enumerate(percentages):
        plt.text(i, percentage, f'{percentage:.3f}%', ha='center', va='bottom', fontsize=10)

    # Remove grey grid background
    plt.grid(False)
    plt.gca().set_facecolor('none')
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_color('black')
    
    #plt.title('Amino Acid Distribution')
    plt.xlabel('Amino Acid')
    plt.ylabel('Quantity (%)')
    plt.xticks(x, amino_acids)
    plt.style.use('ggplot')
    plt.tight_layout()
    plt.savefig(distribution_file_name+'.png', format='png')
    plt.savefig(distribution_file_name+'.pdf', format='pdf')
    return distribution_file_name

if __name__ == '__main__':

    data_dir = os.path.join(os.path.dirname(__file__), '../../../data')  
    results_dir = os.path.join(os.path.dirname(__file__), '../../../results/plots') 
    full_csv = os.path.join(data_dir, "spikeprot0528.clean.uniq.noX.RBD_variants.csv")
    train_csv = os.path.join(data_dir, "spikeprot0528.clean.uniq.noX.RBD_variants_train.csv")
    test_csv = os.path.join(data_dir, "spikeprot0528.clean.uniq.noX.RBD_variants_test.csv")
    plot_variant_distribution(full_csv, data_dir)
    plot_split_variant_distribution(train_csv, test_csv, data_dir)
    plot_aa_distribution(full_csv, data_dir)