#!/usr/bin/env python
"""
This is for plotting the rmse curve plots together with 4 different embedding methods:
- esm embedding curves
- 320 dimension rbd learned embedding curves
- aa index embedding curves
- one-hot embedding curves
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_multi_history(embedding_methods: list, blstm_embedding_csvs: list, save_as: str):
    """Plot training and testing rmse history per epoch of 4 different embedding methods."""
    
    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))

    # Flatten the grid of subplots
    axes = axes.flatten()

    # Iterate over embedding methods, their corresponding CSV files, and subplots
    for method, csv_file, ax in zip(embedding_methods, blstm_embedding_csvs, axes):
        df = pd.read_csv(csv_file)

        ax.plot(df.index, df['train_rmse'], label='Train RMSE')
        ax.plot(df.index, df['test_rmse'], label='Test RMSE')

        ax.set_title(method)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average RMSE per Sample')
        ax.legend()

    fig.text(0.5, 0, 'Learning Rate = 1e-5', ha='center', va='center')
    plt.suptitle('BLSTM Model Performance on DMS Dataset Across Embedding Methods')    
    plt.style.use('ggplot')
    plt.tight_layout()
    plt.savefig(save_as, bbox_inches='tight')

if __name__=='__main__':

    blstm_results_dir = os.path.join(os.path.dirname(__file__), '../../../results/blstm')
    save_as = os.path.join(blstm_results_dir, "blstm_dms_embedding_rmse_plot.png")

    esm_blstm_csv = os.path.join(blstm_results_dir, "blstm_esm-2023-10-11_01-42/blstm-2023-10-11_01-42_train_84242_test_21285.csv")
    rbd_learned_320_blstm_csv = os.path.join(blstm_results_dir, "blstm_rbd_learned-2023-10-11_01-50/blstm-2023-10-11_01-50_train_84242_test_21285.csv")

    blstm_embedding_csvs = [esm_blstm_csv, rbd_learned_320_blstm_csv]
    embedding_methods = ["ESM", "RBD Learned"]

    plot_multi_history(embedding_methods, blstm_embedding_csvs, save_as)
