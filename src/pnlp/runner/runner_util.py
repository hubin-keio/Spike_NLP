#!/usr/bin/env python

import os
import sys
import tqdm
import torch
import textwrap
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

    history_df['train_rmse'] = np.sqrt(history_df['train_loss_per'].values)  # rmse
    history_df['test_rmse'] = np.sqrt(history_df['test_loss_per'].values)

    history_df.to_csv(metrics_csv.replace('.csv', '_per.csv'), index=False)
    plot_rmse_history(history_df, save_as)

def plot_rmse_history(history_df, save_as: str):
    """ Plot RMSE training and testing history per epoch. """

    sns.set_theme()
    sns.set_context('talk')
    sns.set(style="darkgrid")

    # Converting mm to inches for figsize
    width_in = 88/25.4 # mm to inches
    ratio = 16/9
    height_in = width_in/ratio 
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    plt.rcParams['font.family'] = 'sans-serif'

    # Plot
    sns.lineplot(data=history_df, x=history_df.index, y='test_rmse', color='tab:blue', linewidth=0.5, ax=ax) # add label='Test RMSE' for legend
    sns.lineplot(data=history_df, x=history_df.index, y='train_rmse', color='tab:orange', linewidth=0.5,ax=ax) # add label='Train RMSE' for legend
    
    # Set the font size
    font_size = 8
    ax.set_xlabel('Epoch', fontsize=font_size)
    ax.set_ylabel(f'RMSE', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    # ax.legend(fontsize=font_size)

    # Skipping every other x-axis tick mark
    ax_yticks = ax.get_yticks()
    ax.set_ylim(-0.1, 1.8)

    ax_xticks = ax.get_xticks()
    new_xlabels = ['' if i % 2 else label for i, label in enumerate(ax.get_xticklabels())]
    ax.set_xticks(ax_xticks)
    ax.set_xticklabels(new_xlabels)
    ax.set_xlim(-100, 5000)

    plt.tight_layout()
    plt.savefig(save_as + '_rmse.pdf', format='pdf')

def plot_run(csv_name: str, save: bool = True):
    '''
    Generate a single figure with subplots for training loss and training accuracy
    from the model run csv file.

    For runner.py and gpu_ddp_runner.py
    '''
    df = pd.read_csv(csv_name)
    df.columns = df.columns.str.strip()

    sns.set_theme()
    sns.set_context('talk')
    sns.set(style="darkgrid")

    # Converting mm to inches for figsize
    width_in = 88/25.4 # mm to inches
    ratio = 16/9
    height_in = width_in/ratio 
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(width_in, 2*height_in))
    plt.rcParams['font.family'] = 'sans-serif'

    # Set the font size
    font_size = 8

    # Plot Training Loss
    train_loss_line = ax1.plot(df['epoch'], df['train_loss'], color='red', linewidth=0.5, label='Train Loss')
    test_loss_line = ax1.plot(df['epoch'], df['test_loss'], color='orange', linewidth=0.5, label='Test Loss')
    ax1.set_ylabel('Loss', fontsize=font_size)
    ax1.legend(loc='upper right', fontsize=font_size)
    ax1.tick_params(axis='both', which='major', labelsize=font_size)
    ax1.yaxis.get_offset_text().set_fontsize(font_size) 
    ax1.set_ylim(-0.5e6, 8e6) 

    # Plot Training Accuracy
    train_accuracy_line = ax2.plot(df['epoch'], df['train_accuracy']*100, color='blue', linewidth=0.5, label='Train Accuracy')
    test_accuracy_line = ax2.plot(df['epoch'], df['test_accuracy']*100, color='green', linewidth=0.5, label='Test Accuracy')
    ax2.set_xlabel('Epoch', fontsize=font_size)
    ax2.set_ylabel('Accuracy', fontsize=font_size)
    ax2.set_ylim(0, 100) 
    ax2.legend(loc='lower right', fontsize=font_size)
    ax2.tick_params(axis='both', which='major', labelsize=font_size)

    plt.tight_layout()

    if save:
        combined_fname = csv_name.replace('.csv', '_loss_acc.pdf')
        plt.savefig(combined_fname, format='pdf')

if __name__=='__main__':

    results_dir = os.path.join(os.path.dirname(__file__), f'../../../results/run_results')
    plot_run(os.path.join(results_dir,"ddp_runner/original_ddp_runner-2024-01-08_01-14/original_ddp_runner-2024-01-08_01-14_train_278299_test_69325_results.csv"), True)
    plot_run(os.path.join(results_dir,"ddp_runner/ddp-2023-08-16_08-41/ddp-2023-08-16_08-41_results.csv"), True)
    plot_run(os.path.join(results_dir,"ddp_runner/ddp-2023-10-06_20-16/ddp-2023-10-06_20-16_results.csv"), True)

    # # = BINDING = -> y axis (-0.2, 3.2)
    # # Figure - RBD Binding w/ ESM 
    # # - FCN
    # csv_file = os.path.join(results_dir, "fcn/fcn-esm_dms_binding-2023-10-17_15-38/fcn-esm_dms_binding-2023-10-17_15-38_train_84420_test_21105_metrics_per.csv")
    # history_df = pd.read_csv(csv_file, sep=',', header=0)
    # save_as = csv_file.replace("_metrics_per.csv", "")
    # plot_rmse_history(history_df, save_as)
    # # - GCN
    # csv_file = os.path.join(results_dir, "graphsage/graphsage-esm_dms_binding-2023-12-13_18-14/graphsage-esm_dms_binding-2023-12-13_18-14_train_84420_test_21105_metrics_per.csv")
    # history_df = pd.read_csv(csv_file, sep=',', header=0)
    # save_as = csv_file.replace("_metrics_per.csv", "")
    # plot_rmse_history(history_df, save_as)
    # # - BLSTM
    # csv_file = os.path.join(results_dir, "blstm/blstm-esm_dms_binding-2023-12-07_13-53/blstm-esm_dms_binding-2023-12-07_13-53_train_84420_test_21105_metrics_per.csv")
    # history_df = pd.read_csv(csv_file, sep=',', header=0)
    # save_as = csv_file.replace("_metrics_per.csv", "")
    # plot_rmse_history(history_df, save_as)
    # # - ESM-BLSTM 
    # csv_file = os.path.join(results_dir, "esm-blstm/esm-blstm-esm_dms_binding-2023-12-12_17-02/esm-blstm-esm_dms_binding-2023-12-12_17-02_train_84420_test_21105_metrics_per.csv")
    # history_df = pd.read_csv(csv_file, sep=',', header=0)
    # save_as = csv_file.replace("_metrics_per.csv", "")
    # plot_rmse_history(history_df, save_as)    
    # # - BERT-BLSTM_ESM -> plot_bert_blstm.py

    # # Figure - RBD Binding w/ RBD Learned
    # # - FCN
    # csv_file = os.path.join(results_dir, "fcn/fcn-rbd_learned_320_dms_binding-2023-12-21_09-44/fcn-rbd_learned_320_dms_binding-2023-12-21_09-44_train_84420_test_21105_metrics_per.csv")
    # history_df = pd.read_csv(csv_file, sep=',', header=0)
    # save_as = csv_file.replace("_metrics_per.csv", "")
    # plot_rmse_history(history_df, save_as)
    # # - GCN
    # csv_file = os.path.join(results_dir, "graphsage/graphsage-rbd_learned_320_dms_binding-2024-01-02_14-50/graphsage-rbd_learned_320_dms_binding-2024-01-02_14-50_train_84420_test_21105_metrics_per.csv")
    # history_df = pd.read_csv(csv_file, sep=',', header=0)
    # save_as = csv_file.replace("_metrics_per.csv", "")
    # plot_rmse_history(history_df, save_as)
    # # - BLSTM 
    # csv_file = os.path.join(results_dir, "blstm/blstm-rbd_learned_320_dms_binding-2024-01-04_14-12/blstm-rbd_learned_320_dms_binding-2024-01-04_14-12_train_84420_test_21105_metrics_per.csv")
    # history_df = pd.read_csv(csv_file, sep=',', header=0)
    # save_as = csv_file.replace("_metrics_per.csv", "")
    # plot_rmse_history(history_df, save_as)
    # # - BERT-BLSTM -> plot_bert_blstm.py

    # # = EXPRESSION = -> y axis (-0.2, 2.2)
    # # Figure - RBD Expression w/ ESM
    # # - FCN
    # csv_file = os.path.join(results_dir, "fcn/fcn-esm_dms_expression-2023-12-19_15-53/fcn-esm_dms_expression-2023-12-19_15-53_train_93005_test_23252_metrics_per.csv")
    # history_df = pd.read_csv(csv_file, sep=',', header=0)
    # save_as = csv_file.replace("_metrics_per.csv", "")
    # plot_rmse_history(history_df, save_as)
    # # - GCN
    # csv_file = os.path.join(results_dir, "graphsage/graphsage-esm_dms_expression-2023-12-20_08-48/graphsage-esm_dms_expression-2023-12-20_08-48_train_93005_test_23252_metrics_per.csv")
    # history_df = pd.read_csv(csv_file, sep=',', header=0)
    # save_as = csv_file.replace("_metrics_per.csv", "")
    # plot_rmse_history(history_df, save_as)
    # # - BLSTM
    # csv_file = os.path.join(results_dir, "blstm/blstm-esm_dms_expression-2023-12-07_13-58/blstm-esm_dms_expression-2023-12-07_13-58_train_93005_test_23252_metrics_per.csv")
    # history_df = pd.read_csv(csv_file, sep=',', header=0)
    # save_as = csv_file.replace("_metrics_per.csv", "")
    # plot_rmse_history(history_df, save_as)
    # # - ESM-BLSTM 
    # csv_file = os.path.join(results_dir, "esm-blstm/esm-blstm-esm_dms_expression-2023-12-12_16-58/esm-blstm-esm_dms_expression-2023-12-12_16-58_train_93005_test_23252_metrics_per.csv")
    # history_df = pd.read_csv(csv_file, sep=',', header=0)
    # save_as = csv_file.replace("_metrics_per.csv", "")
    # plot_rmse_history(history_df, save_as)    
    # # - BERT-BLSTM_ESM -> plot_bert_blstm.py

    # # Figure - RBD Expression w/ RBD Learned
    # # - FCN
    # csv_file = os.path.join(results_dir, "fcn/fcn-rbd_learned_320_dms_expression-2023-12-20_12-02/fcn-rbd_learned_320_dms_expression-2023-12-20_12-02_train_93005_test_23252_metrics_per.csv")
    # history_df = pd.read_csv(csv_file, sep=',', header=0)
    # save_as = csv_file.replace("_metrics_per.csv", "")
    # plot_rmse_history(history_df, save_as)
    # # - GCN
    # csv_file = os.path.join(results_dir, "graphsage/graphsage-rbd_learned_320_dms_expression-2023-12-25_00-54/graphsage-rbd_learned_320_dms_expression-2023-12-25_00-54_train_93005_test_23252_metrics_per.csv")
    # history_df = pd.read_csv(csv_file, sep=',', header=0)
    # save_as = csv_file.replace("_metrics_per.csv", "")
    # plot_rmse_history(history_df, save_as)
    # # - BLSTM 
    # csv_file = os.path.join(results_dir, "blstm/blstm-rbd_learned_320_dms_expression-2024-01-04_14-15/blstm-rbd_learned_320_dms_expression-2024-01-04_14-15_train_93005_test_23252_metrics_per.csv")
    # history_df = pd.read_csv(csv_file, sep=',', header=0)
    # save_as = csv_file.replace("_metrics_per.csv", "")
    # plot_rmse_history(history_df, save_as)
    # # - BERT-BLSTM -> plot_bert_blstm.py
