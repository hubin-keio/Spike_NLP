#!/usr/bin/env python

import os
import sys
import tqdm
import torch
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from typing import Union
from prettytable import PrettyTable
from collections import defaultdict
from torch.utils.data import DataLoader, random_split

def save_model(model, optimizer: torch.optim.SGD, epoch: int, save_as: str):
    """
    Save model parameters.

    model: a model object
    optimizer: model optimizer
    epoch: number of epochs in the end of the model running
    save_as: file name for saveing the model.
    """
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                save_as)

def count_parameters(model):
    """
    Count model parameters and print a summary

    A nice hack from:
    https://stackoverflow.com/a/62508086/1992369
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}\n")
    return total_params

def calc_train_test_history(metrics_csv: str, n_train: int, n_test: int, save_as: str):
    """ Calculate the average mse per item and rmse """

    history_df = pd.read_csv(metrics_csv, sep=',', header=0)

    history_df['train_loss_per'] = history_df['train_loss']/n_train  # average mse per item
    history_df['test_loss_per'] = history_df['test_loss']/n_test

    history_df['train_rmse_per'] = np.sqrt(history_df['train_loss_per'].values)  # rmse
    history_df['test_rmse_per'] = np.sqrt(history_df['test_loss_per'].values)

    history_df.to_csv(metrics_csv.replace('.csv', '_per.csv'), index=False)
    plot_rmse_history(history_df, save_as)

def plot_rmse_history(history_df, save_as: str):
    """ Plot RMSE training and testing history per epoch. """
    
    sns.set_theme()
    sns.set_context('talk')
    sns.set(style="darkgrid")
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.lineplot(data=history_df, x=history_df.index, y='train_rmse_per', label='Train RMSE', color='tab:orange', ax=ax)
    sns.lineplot(data=history_df, x=history_df.index, y='test_rmse_per', label='Test RMSE', color='tab:blue', alpha=0.9, ax=ax)
    
    # # Skipping every other y-axis tick mark
    # ax_yticks = ax.get_yticks()
    # ax.set_yticks(ax_yticks[::2])  # Keep every other tick

    ax.set(xlabel='Epoch', ylabel='Average RMSE Per Sample')
    plt.style.use('ggplot')
    plt.tight_layout()
    plt.savefig(save_as + '_rmse.png', format='png')
    plt.savefig(save_as + '_rmse.pdf', format='pdf')

#def plot_subplot(, save_as:str):


if __name__=='__main__':

    results_dir = os.path.join(os.path.dirname(__file__), f'../../../results/run_results/fcn/fcn_esm-2023-10-17_15-38')
    csv_file = os.path.join(results_dir, "fcn2023-10-17_15-38_train_84242_test_21285.csv")
    history_df = pd.read_csv(csv_file, sep=',', header=0)
    save_as = os.path.join(results_dir, 'fcn-dms_binding-2023-10-17_15-38_train_84242_test_21285.csv')
    plot_rmse_history(history_df, save_as)
