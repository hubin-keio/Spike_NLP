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

def calc_train_test_history(metrics: dict, n_train: int, n_test: int, embedding_method:str, dataset:str, model:str, lr:str, save_as: str):
    """ Calculate the average mse per item and rmse """

    history_df = pd.DataFrame(metrics)
    history_df['train_loss'] = history_df['train_loss']/n_train  # average mse per item
    history_df['test_loss'] = history_df['test_loss']/n_test

    history_df['train_rmse'] = np.sqrt(history_df['train_loss'])  # rmse
    history_df['test_rmse'] = np.sqrt(history_df['test_loss'])

    plot_rmse_history(embedding_method, dataset, model, lr, history_df, save_as)
    plot_history(embedding_method, dataset, model, lr, history_df, save_as)

def plot_from_csv(csv_file:str, embedding_method:str, dataset:str, model:str, lr:str, save_as: str):
    history_df = pd.read_csv(csv_file, sep=',', header=0)
    plot_rmse_history(embedding_method, dataset, model, lr, history_df, save_as)
    plot_history(embedding_method, dataset, model, lr, history_df, save_as)

# def plot_duplicate_rmse_stats():
#     # TODO

def plot_rmse_history(embedding_method: str, dataset:str, model:str, lr:str, history_df, save_as: str):
    """ Plot RMSE training and testing history per epoch. """
    
    sns.set_theme()
    sns.set_context('talk')
    palette = sns.color_palette()
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.lineplot(data=history_df, x=history_df.index, y='test_rmse', label='testing', color=palette[0], ax=ax)
    sns.lineplot(data=history_df, x=history_df.index, y='train_rmse', label='training', color=palette[1], ax=ax)
    ax.set(xlabel='Epochs', ylabel='Average RMSE per sample')

    title = f'RMSE over epochs using {embedding_method} embedding on {dataset} dataset ({model} Model)'
    subtitle = f'Learning Rate = {lr}'
    fig.suptitle(title, fontsize=16)
    ax.set_title(subtitle, fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.895)  # Adjust this value to change the spacing
    plt.savefig(save_as + '-rmse.png')

def plot_history(embedding_method: str, dataset:str, model:str, lr:str, history_df, save_as: str):
    """ Plot training and testing history per epoch. """
    
    sns.set_theme()
    sns.set_context('talk')
    palette = sns.color_palette()
    plt.ion()
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))

    sns.lineplot(data=history_df, x=history_df.index, y='test_loss', label='testing', color=palette[0], ax=axes[0])
    sns.lineplot(data=history_df, x=history_df.index, y='train_loss', label='training', color=palette[1], ax=axes[0])
    axes[0].set(xlabel='Epochs', ylabel='Average MSE per sample')

    sns.lineplot(data=history_df, x=history_df.index, y='test_rmse', label='testing', color=palette[0], ax=axes[1])
    sns.lineplot(data=history_df, x=history_df.index, y='train_rmse', label='training', color=palette[1], ax=axes[1])
    axes[1].set(xlabel='Epochs', ylabel='Average RMSE per sample')

    title = f'MSE & RMSE over epochs using {embedding_method} embedding on {dataset} dataset ({model} Model)'
    subtitle = f'Learning Rate = {lr}'
    fig.suptitle(title, fontsize=16, y=0.985)
    fig.text(0.5425, 0.945, subtitle, fontsize=14, ha='center')
    
    plt.tight_layout()
    plt.savefig(save_as + '-mse_rmse.png')

def plot_multi_history(model:str, embedding_methods: list, embedding_csvs: list, save_as:str):
    """Plot training and testing rmse history per epoch of different embedding methods."""
    
    sns.set_theme()
    palette = sns.color_palette()
    
    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    # Flatten the grid of subplots
    axes = axes.flatten()

    # Iterate over embedding methods, their corresponding CSV files, and subplots
    for method, csv_file, ax in zip(embedding_methods, embedding_csvs, axes):
        df = pd.read_csv(csv_file)

        ax.plot(df.index, df['test_rmse'], label='Test RMSE', color=palette[0])
        ax.plot(df.index, df['train_rmse'], label='Train RMSE', color=palette[1])

        ax.set_title(method)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average RMSE per Sample')
        ax.legend()
    
    # Hide the unused subplot
    for i in range(len(embedding_methods), len(axes)):
        axes[i].axis('off')

    fig.text(0.5, 0, 'Learning Rate = 1e-5', ha='center', va='center')
    plt.suptitle(f'{model.upper()} Model Performance on DMS Binding Dataset Across Embedding Methods')    
    plt.style.use('ggplot')
    plt.tight_layout()
    plt.savefig(save_as, bbox_inches='tight')

if __name__=='__main__':
    csv_file = "../../../results/graphsage/dms/expression/graphsage_rbd_learned-2023-10-25_16-52/graphsage-2023-10-25_16-52_train_93005_test_23252.csv"
    embedding_method = "RBD Learned"
    dataset = "DMS expression"
    model = "GraphSAGE"
    lr = "1e-5"
    save_as = "../../../results/graphsage/dms/expression/graphsage_rbd_learned-2023-10-25_16-52/graphsage-2023-10-25_16-52_train_93005_test_23252"
    plot_from_csv(csv_file, embedding_method, dataset, model, lr, save_as)

    # model = "fcn"
    # dataset = "dms_binding"
    # results_dir = os.path.join(os.path.dirname(__file__), f'../../../results/{model}')
    
    # esm_csv = os.path.join(results_dir, "fcn_esm-2023-10-17_15-38/fcn2023-10-17_15-38_train_84242_test_21285.csv")
    # rbd_learned_csv = os.path.join(results_dir, "fcn_rbd_learned-2023-10-17_11-03/fcn2023-10-17_11-03_train_84242_test_21285.csv")
    # rbd_bert_csv = os.path.join(results_dir, "fcn_rbd_bert-2023-10-17_11-12/fcn2023-10-17_11-12_train_84242_test_21285.csv")

    # embedding_csvs = [esm_csv, rbd_learned_csv, rbd_bert_csv]
    # embedding_methods = ["ESM", "RBD Learned", "RBD Learned with BERT"]

    # save_as = os.path.join(results_dir, f"{model}_{dataset}_rmse_multiplot.png")
    # plot_multi_history(model, embedding_methods, embedding_csvs, save_as)