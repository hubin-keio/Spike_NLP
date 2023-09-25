"""Plot accuracy stats for model prediction csv file"""

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def csv_reader(csv_file: str):
    """Read in prediction data from csv_file (if using from outside csv_file instead of dict)."""
    
    data = {}

    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader) 

        for row in csv_reader:
            key = row[0]
            value = [int(i) for i in row[1:]]
            data[key] = value

    return data

def plot_aa_perc_pred_stats_heatmap(incorrect_preds:dict, csv_name:str, save:bool=True):
    """Plots heatmap of expected vs predicted amino acid incorrect prediction counts."""
    ALL_AAS = 'ACDEFGHIKLMNPQRSTVWY'

    # Create a DataFrame with all possible amino acid combinations
    all_combinations = [(e_aa, p_aa) for e_aa in ALL_AAS for p_aa in ALL_AAS]
    all_df = pd.DataFrame(all_combinations, columns=["Expected", "Predicted"])

    # Go through incorrect preds data and create a dataframe
    data = []
    for error_type, count in incorrect_preds.items():
        e_aa, p_aa = error_type.split('->')
        data.append((e_aa, p_aa, count[0]))

    df = pd.DataFrame(data, columns=["Expected", "Predicted", "Count"])
    
    # Merge dataframes so any aa that did not show up are accounted for
    df = pd.merge(all_df, df, how="left", on=["Expected", "Predicted"])
    df["Count"].fillna(0, inplace=True)

    # Calculate the total counts for each expected aa
    total_counts = df.groupby("Expected")["Count"].sum()
    df["Expected Total"] = df["Expected"].map(total_counts)

    # Calculate error percentage
    df["Error Percentage"] = (df["Count"] / df["Expected Total"]) * 100
    df["Error Percentage"].fillna(0, inplace=True)

    # Plotting
    # Pivot the DataFrame to create a heatmap data structure
    heatmap_data = df.pivot_table(index="Expected", columns="Predicted", values="Error Percentage")

    # Create the heatmap using seaborn
    plt.figure(figsize=(16, 8))

    cmap = sns.color_palette("rocket_r", as_cmap=True)
    sns.heatmap(heatmap_data, 
                annot=True, fmt=".3f", 
                linewidth=.5,
                cmap=cmap, vmin=0, vmax=100,
                cbar_kws={'drawedges':False, 'label': 'Prediction Rate (%)'})

    plt.title('Amino Acid Prediction Distribution')
    plt.xlabel('Predicted Amino Acid')
    plt.ylabel('Expected Amino Acid')
    plt.yticks(rotation=0)
    plt.style.use('ggplot')
    plt.tight_layout()
    
    if save:
        fname = csv_name.replace('.csv', '_aa_perc_pred_stats_heatmap.png')
        plt.savefig(fname)

def plot_aa_error_history(incorrect_preds:dict, csv_name:str, normalize:bool, save:bool=True):
    """
    Calculate the incorrect prediction count history across all epochs per amino acid.
    Returns a dictionary of lists where 'expected aa: [epoch1_ct, epoch2_ct, epoch3_ct, ...]'.
    """
    ALL_AAS = 'ACDEFGHIKLMNPQRSTUVWXY'

    # Initialize dictionary, get error count length (epochs) from first key
    e_aa_error_history = {aa: [0] * len(incorrect_preds[next(iter(incorrect_preds))]) for aa in ALL_AAS}

    for error_type, counts in incorrect_preds.items():
        e_aa, _ = error_type.split('->')
        for epoch, count in enumerate(counts):
            e_aa_error_history[e_aa][epoch] += count
  
    df = pd.DataFrame(e_aa_error_history)

    if normalize:
        #df = df.div(df.sum(axis=1), axis=0)
        df = ((df - df.min()) / (df.max() - df.min())) * 100
         
    # Plot
    plt.figure(figsize=(10, 8))  # Adjust the figsize for wider plot
    df.plot(ax=plt.gca())

    plt.title('Error Counts per Epoch for Expected Amino Acids')
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Error Counts' if normalize else 'Error Counts')
    plt.yticks(rotation=0)
    plt.style.use('ggplot')
    plt.tight_layout()
    plt.legend(title="Expected Amino Acid", loc='upper left', bbox_to_anchor=(1, 1))

    if save:
        fname = csv_name.replace('.csv', '_aa_nmm_error_history.png') if normalize else csv_name.replace('.csv', '_aa_error_history.png')
        plt.savefig(fname, bbox_inches='tight') # to avoid cutting off legend

