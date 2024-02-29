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

def plot_aa_perc_pred_stats_heatmap(incorrect_preds: dict, csv_name: str, save: bool = True):
    """Plots heatmap of expected vs predicted amino acid incorrect prediction counts. Expected on x axis."""
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

    # Pivot the DataFrame to create a heatmap data structure
    heatmap_data = df.pivot_table(index="Predicted", columns="Expected", values="Error Percentage")

    # Plotting heatmap...
    sns.set_theme()
    sns.set_context('talk')

    # Converting mm to inches for figsize
    plt.figure(figsize=(16, 9))
    plt.rcParams['font.family'] = 'sans-serif'

    # Plot
    cmap = sns.color_palette("rocket_r", as_cmap=True)
    sns.heatmap(heatmap_data,
                annot=True, fmt=".2f",
                linewidth=.5,
                cmap=cmap, vmin=0, vmax=100,
                annot_kws={"size": 13},
                cbar_kws={'drawedges': False, 'label': 'Prediction Rate (%)'})

    plt.ylabel('Predicted Amino Acid')
    plt.xlabel('Expected (Masked) Amino Acid')
    plt.xticks(rotation=0)
    plt.style.use('ggplot')
    plt.tight_layout()

    if save:
        fname = csv_name.replace('.csv', '_aa_perc_pred_stats_heatmap')
        plt.savefig(fname + '.png', format='png')
        plt.savefig(fname + '.pdf', format='pdf')

if __name__=="__main__":

    results_dir = os.path.join(os.path.dirname(__file__), f'../run_results/runner')
    # 24
    csv_file = os.path.join(results_dir, "original_runner-2024-01-16_13-52/original_runner-2024-01-16_13-52_train_278299_test_69325_predictions.csv")
    incorrect_preds = csv_reader(csv_file)
    plot_aa_perc_pred_stats_heatmap(incorrect_preds, csv_file, True)
    # 320
    csv_file = os.path.join(results_dir, "original_runner-2023-12-19_01-51/original_runner-2023-12-19_01-51_train_278299_test_69325_predictions.csv")
    incorrect_preds = csv_reader(csv_file)
    plot_aa_perc_pred_stats_heatmap(incorrect_preds, csv_file, True)
    # 768
    csv_file = os.path.join(results_dir, "original_runner-2023-12-19_01-52/original_runner-2023-12-19_01-52_train_278299_test_69325_predictions.csv")
    incorrect_preds = csv_reader(csv_file)
    plot_aa_perc_pred_stats_heatmap(incorrect_preds, csv_file, True)