#!/usr/bin/env python

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_combined_history(history_df: str, save_as):
    """
    Generate a single figure with subplots for combined training loss
    from the model run csv file.
    """
    sns.set_theme()
    sns.set_context('talk')
    sns.set(style="darkgrid")
    plt.ion()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))

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
    plt.savefig(save_as+'_combined_loss.pdf', format='pdf')

def plot_mlm_history(history_df: str, save_as):
    """
    Generate a single figure with subplots for training loss and training accuracy
    from the model run csv file.
    """
    sns.set_theme()
    sns.set_context('talk')
    sns.set(style="darkgrid")
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16, 18))

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
    plt.savefig(save_as+'_loss_acc.pdf', format='pdf')

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
    sns.lineplot(data=history_df, x=history_df.index, y='test_blstm_rmse', color='tab:blue', linewidth=0.5, ax=ax) # add label='Test RMSE' for legend
    sns.lineplot(data=history_df, x=history_df.index, y='train_blstm_rmse', color='tab:orange', linewidth=0.5,ax=ax) # add label='Train RMSE' for legend
    
    # Set the font size
    font_size = 8
    ax.set_xlabel('Epoch', fontsize=font_size)
    ax.set_ylabel(f'RMSE', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    # ax.legend(fontsize=font_size)

    # Skipping every other y-, x-axis tick mark
    ax_yticks = ax.get_yticks()
    ax.set_ylim(-0.1, 1.8)

    ax_xticks = ax.get_xticks()
    new_xlabels = ['' if i % 2 else label for i, label in enumerate(ax.get_xticklabels())]
    ax.set_xticks(ax_xticks)
    ax.set_xticklabels(new_xlabels)
    ax.set_xlim(-100, 5000)

    plt.tight_layout()
    plt.savefig(save_as + '_rmse.pdf', format='pdf')

def plot_all_loss_history(history_df, save_as:str):
    """ Plot error1 (MLM), error2 (BLSTM), and total_error training and testing history per epoch. """
    sns.set_theme()
    sns.set_context('talk')
    sns.set(style="darkgrid")
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.rcParams['font.family'] = 'sans-serif'
    fontsize = 20

    # Color mapping
    palette = sns.color_palette("Paired", 10)
    
    # Plot
    sns.lineplot(data=history_df, x=history_df.index, y='test_mlm_loss', label='Test Error 1', color=palette[0], linewidth=2, ax=ax) 
    sns.lineplot(data=history_df, x=history_df.index, y='train_mlm_loss', label='Train Error 1', color=palette[1], linewidth=2, ax=ax)
    sns.lineplot(data=history_df, x=history_df.index, y='test_blstm_rmse', label='Test Error 2', color=palette[4], linewidth=2, ax=ax)
    sns.lineplot(data=history_df, x=history_df.index, y='train_blstm_rmse', label='Train Error 2', color=palette[5], linewidth=2, ax=ax)
    sns.lineplot(data=history_df, x=history_df.index, y='test_combined_loss', label='Test Combined Error', color=palette[8], linewidth=2, ax=ax)
    sns.lineplot(data=history_df, x=history_df.index, y='train_combined_loss', label='Train Combined Error', color=palette[9], linewidth=2, ax=ax)

    ax.set_xlim(-100, 5000)
    ax_yticks = ax.get_yticks()
    ax.set_yticks(ax_yticks[::2])  # Keep every other tick
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.yaxis.get_offset_text().set_fontsize(fontsize) 

    ax.legend(fontsize=fontsize)
    ax.set_xlabel('Epoch', fontsize=fontsize)
    ax.set_ylabel('Loss', fontsize=fontsize)

    plt.style.use('ggplot')
    plt.tight_layout()
    plt.savefig(save_as + '_all_loss.pdf', format='pdf')

def calc_train_test_history(metrics_csv: str, n_train: int, n_test: int, save_as: str):
    """ Calculate the average mse per item and rmse """

    history_df = pd.read_csv(metrics_csv, sep=',', header=0)

    history_df['train_blstm_loss_per'] = history_df['train_blstm_loss']/n_train  # average mse per item
    history_df['test_blstm_loss_per'] = history_df['test_blstm_loss']/n_test

    history_df['train_blstm_rmse'] = np.sqrt(history_df['train_blstm_loss_per'].values)  # rmse
    history_df['test_blstm_rmse'] = np.sqrt(history_df['test_blstm_loss_per'].values)

    history_df.to_csv(save_as+'_metrics_per.csv', index=False)
    plot_mlm_history(history_df, save_as)
    plot_rmse_history(history_df, save_as)
    plot_combined_history(history_df, save_as)