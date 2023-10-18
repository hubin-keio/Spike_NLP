#!/usr/bin/env python

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame  # Add missing import

def calc_train_test_history(metrics: dict, n_train: int, n_test: int, embedding_method: str, save_as: str):
    """ Calculate the average mse per item and rmse """

    history_df = pd.DataFrame(metrics)
    history_df['train_loss'] = history_df['train_loss']/n_train  # average mse per item
    history_df['test_loss'] = history_df['test_loss']/n_test

    history_df['train_rmse'] = np.sqrt(history_df['train_loss'])  # rmse
    history_df['test_rmse'] = np.sqrt(history_df['test_loss'])

    plot_history_rmse_only(embedding_method, history_df, save_as)
    plot_history(embedding_method, history_df, save_as)

def plot_history_rmse_only(embedding_method: str, history_df: DataFrame, save_as: str):
    """ Plot training and testing history per epoch. """  
    
    sns.set_theme()
    sns.set_context('talk')
    plt.figure(figsize=(12, 6))

    # Single line plot for train and test RMSE
    sns.lineplot(data=history_df[['train_rmse', 'test_rmse']], dashes=False)
    
    plt.title(f'RMSE Over Epochs using {embedding_method} Embedding')
    plt.xlabel('Epochs')
    plt.ylabel('Average RMSE per sample')
    plt.tight_layout()
    plt.savefig(save_as + '-rmse.png')
    history_df.to_csv(save_as + '.csv', index=False)

def plot_history(embedding_method: str, history_df: DataFrame, save_as: str):
    """ Plot training and testing history per epoch. """
    
    sns.set_theme()
    sns.set_context('talk')
    plt.ion()
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))

    sns.lineplot(data=history_df, x=history_df.index, y='train_loss', label='training', ax=axes[0])
    sns.lineplot(data=history_df, x=history_df.index, y='test_loss', label='testing', ax=axes[0])
    axes[0].set(xlabel='Epochs', ylabel='Average MSE per sample')

    sns.lineplot(data=history_df, x=history_df.index, y='train_rmse', label='training', ax=axes[1])
    sns.lineplot(data=history_df, x=history_df.index, y='test_rmse', label='testing', ax=axes[1])
    axes[1].set(xlabel='Epochs', ylabel='Average RMSE per sample')

    plt.suptitle(f'MSE and RMSE Over Epochs Using {embedding_method} Embedding')
    plt.tight_layout()
    plt.savefig(save_as + '-mse_rmse.png')
