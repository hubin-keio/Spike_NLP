#!/usr/bin/env python

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_combined_history(history_df: str, save_as):
    '''
    Generate a single figure with subplots for combined training loss
    from the model run csv file.
    '''
    sns.set_theme()
    sns.set_context('talk')
    sns.set(style="darkgrid")
    plt.ion()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

    # Plot Training Loss
    train_loss_line = ax.plot(history_df['epoch'], history_df['train_combined_loss'], color='tab:orange', label='Train Loss')
    test_loss_line = ax.plot(history_df['epoch'], history_df['test_combined_loss'],color='tab:blue', label='Test Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')

    # Skipping every other y-axis tick mark
    yticks = ax.get_yticks()
    ax.set_yticks(yticks[::2])  # Keep every other tick

    plt.style.use('ggplot')
    plt.tight_layout()
    plt.savefig(save_as+'_combined_loss.png', format='png')
    plt.savefig(save_as+'_combined_loss.pdf', format='pdf')

def plot_mlm_history(history_df: str, save_as):
    '''
    Generate a single figure with subplots for training loss and training accuracy
    from the model run csv file.
    '''
    sns.set_theme()
    sns.set_context('talk')
    sns.set(style="darkgrid")
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

    # Plot Training Loss
    train_loss_line = ax1.plot(history_df['epoch'], history_df['train_mlm_loss'], color='tab:red', label='Train Loss')
    test_loss_line = ax1.plot(history_df['epoch'], history_df['test_mlm_loss'],color='tab:orange', label='Test Loss')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')

    # Plot Training Accuracy
    train_accuracy_line = ax2.plot(history_df['epoch'], history_df['train_mlm_accuracy'], color='tab:blue', label='Train Accuracy')
    test_accuracy_line = ax2.plot(history_df['epoch'], history_df['test_mlm_accuracy'], color='tab:green', label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1) 
    ax2.legend(loc='upper right')

    # Skipping every other y-axis tick mark
    a1_yticks = ax1.get_yticks()
    ax1.set_yticks(a1_yticks[::2])  # Keep every other tick

    plt.style.use('ggplot')
    plt.tight_layout()
    plt.savefig(save_as+'_loss_acc.png', format='png')
    plt.savefig(save_as+'_loss_acc.pdf', format='pdf')

def plot_rmse_history(history_df, save_as: str):
    """ Plot RMSE training and testing history per epoch. """
    
    sns.set_theme()
    sns.set_context('talk')
    sns.set(style="darkgrid")
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.lineplot(data=history_df, x=history_df.index, y='train_blstm_rmse_per', label='Train RMSE', color='tab:orange', ax=ax)
    sns.lineplot(data=history_df, x=history_df.index, y='test_blstm_rmse_per', label='Test RMSE', color='tab:blue', ax=ax)
    
    # Skipping every other y-axis tick mark
    ax_yticks = ax.get_yticks()
    ax.set_yticks(ax_yticks[::2])  # Keep every other tick

    ax.set(xlabel='Epoch', ylabel='Average RMSE Per Sample')
    plt.style.use('ggplot')
    plt.tight_layout()
    plt.savefig(save_as + '_rmse.png', format='png')
    plt.savefig(save_as + '_rmse.pdf', format='pdf')
 
if __name__=='__main__':
    # Data/results directories
    data_dir = os.path.join(os.path.dirname(__file__), '../../../results/run_results/bert_blstm')
    binding_csv = os.path.join(data_dir, 'bert_blstm-dms_binding-2023-11-22_23-03/bert_blstm-dms_binding-2023-11-22_23-03_train_84420_test_21105_metrics_per.csv')
    binding_df = pd.read_csv(binding_csv, sep=',', header=0)
    plot_mlm_history(binding_df, binding_csv[:-4]+'_binding')
    plot_rmse_history(binding_df, binding_csv[:-4]+'_binding')
    plot_combined_history(binding_df, binding_csv[:-4]+'_binding')

    expression_csv = os.path.join(data_dir, 'bert_blstm-dms_expression-2023-11-22_23-05/bert_blstm-dms_expression-2023-11-22_23-05_train_93005_test_23252_metrics_per.csv')
    expression_df = pd.read_csv(expression_csv, sep=',', header=0)
    plot_mlm_history(expression_df, expression_csv[:-4]+'_expression')
    plot_rmse_history(expression_df, expression_csv[:-4]+'_expression')
    plot_combined_history(expression_df, expression_csv[:-4]+'_expression')