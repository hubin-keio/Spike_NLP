#!/usr/bin/env python

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_rmse_distribution(df, save_as:str):
    """ 
    Plots the RMSE density distribution for repeated sequences. 
    """
    sns.set_theme()
    sns.set_context('talk')
    sns.set(style='white')
    plt.figure(figsize=(16, 9))
    plt.rcParams['font.family'] = 'sans-serif'

    # Create histogram and kde plot
    sns.histplot(df['rmse'], kde=False, stat='density', bins=100, color='dimgrey', edgecolor='black')
    sns.kdeplot(df['rmse'], color='red', linewidth=2)

    plt.tight_layout()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('RMSE', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.savefig(save_as+'_rmse_density.png', format='png')
    plt.savefig(save_as+'_rmse_density.pdf', format='pdf')

def plot_value_distribution(df, value_col:str, save_as:str):
    """ 
    Plots the measured value (binding log10Ka or expression ML_meanF) distribution. 
    """
    sns.set_theme()
    sns.set_context('talk')
    sns.set(style='white')
    plt.figure(figsize=(16, 9))
    plt.rcParams['font.family'] = 'sans-serif'

    # Create histogram and kde plot
    sns.histplot(df[value_col], kde=False, stat='density', bins=100, color='dimgrey', edgecolor='black')
    sns.kdeplot(df[value_col], color='red', linewidth=2)

    plt.tight_layout()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(f'Measured {value_col[5:]}', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.savefig(save_as+f'_{value_col[5:]}_density.png', format='png')
    plt.savefig(save_as+f'_{value_col[5:]}_density.pdf', format='pdf')

if __name__ == '__main__':

    data_dir = os.path.join(os.path.dirname(__file__), '../../../data')  
    results_dir =  os.path.join(os.path.dirname(__file__), '../../../results/plots') 
    binding_csv = os.path.join(data_dir, 'dms_duplicate_mutation_binding_Kds.csv')
    expression_csv = os.path.join(data_dir, 'dms_duplicate_mutation_expression_meanFs.csv')

    # Mean log10KA and RMSE calculation 
    binding_df = pd.read_csv(binding_csv, sep=',', header=0)
    binding_df = binding_df[['labels', 'log10Ka', 'sequence']].copy()
    binding_df['mean_log10Ka'] = binding_df.groupby('labels')['log10Ka'].transform('mean')
    binding_df['squared_difference'] = (binding_df['log10Ka'] - binding_df['mean_log10Ka'])**2
    binding_df['mse'] = binding_df.groupby('labels')['squared_difference'].transform('mean')
    binding_df['rmse'] = np.sqrt(binding_df['mse'])
    unique_binding_df = binding_df.drop_duplicates(subset='labels', keep='first')
    save_as = os.path.join(results_dir, 'dms_mutation_binding_Kds')
    plot_rmse_distribution(unique_binding_df, save_as)
    plot_value_distribution(unique_binding_df, 'mean_log10Ka', save_as)

    # Mean ML_meanF and RMSE calculation
    expression_df = pd.read_csv(expression_csv, sep=',', header=0)
    expression_df = expression_df[['labels', 'ML_meanF', 'sequence']].copy()
    expression_df['mean_ML_meanF'] = expression_df.groupby('labels')['ML_meanF'].transform('mean')
    expression_df['squared_difference'] = (expression_df['ML_meanF'] - expression_df['mean_ML_meanF'])**2
    expression_df['mse'] = expression_df.groupby('labels')['squared_difference'].transform('mean')
    expression_df['rmse'] = np.sqrt(expression_df['mse'])
    unique_expression_df = expression_df.drop_duplicates(subset='labels', keep='first')
    save_as = os.path.join(results_dir, 'dms_mutation_expression_meanFs')
    plot_rmse_distribution(unique_expression_df, save_as)
    plot_value_distribution(unique_expression_df, 'mean_ML_meanF', save_as)


    