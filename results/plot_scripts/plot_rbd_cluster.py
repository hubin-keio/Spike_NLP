#!/usr/bin/env python
"""
Plot 2d UMAP, and t-SNE plots from RBD embeddings generated from cluster_blstm.py.
"""

import os
import datetime
import logging
import pickle
import psutil
import umap
import umap.plot
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from contextlib import redirect_stdout
from tqdm import tqdm
from PIL import Image

logger = logging.getLogger(__name__)

def memory_usage() -> str:
    """ 
    Returns string of current memory usage of Python process. 
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB"

def extract_embedding_pickle(pickle_file:str):
    """
    Extracts seq_ids, variants, and embeddings from embedding pickle.
    """
    with open(pickle_file, 'rb') as f:
        pkl_seq_ids, pkl_variants, pkl_embeddings = pickle.load(f)

    df = pd.DataFrame({"seq_id": pkl_seq_ids,
                       "variant": pkl_variants,
                       "embedding": pkl_embeddings})
    return df

def sample_embedding_pickle(run_dir:str, df, whole:bool, ado:bool, iteration:str, rnd_seed=None) -> tuple:
    """
    From extracted embedding dataframe, filters the data and prepares for
    UMAP and T-SNE.

    If ado, data is filtered to just 'Alpha', 'Delta', 'Omicron' variants 
    and sampled accordingly.

    You can set the fraction of total Alpha, Delta, and/or Omicron that you want.
    """
    save_as = os.path.join(run_dir, f"rbd_variants_clustering_bert_blstm")

    if ado:
        ado_df = df[df['variant'].isin(["Alpha", "Delta", "Omicron"])]

        if not whole:
            sample_sizes = {"Alpha": int(1 * (df['variant'] == 'Alpha').sum()), 
                            "Delta": int(0.2 * (df['variant'] == 'Delta').sum()),  
                            "Omicron": int(0.14 * (df['variant'] == 'Omicron').sum())} 

            save_as = save_as + f"_a{sample_sizes['Alpha']}_d{sample_sizes['Delta']}_o{sample_sizes['Omicron']}"

            sampled_dfs = [ado_df[ado_df['variant'] == variant].sample(n=sample_sizes[variant], random_state=rnd_seed)
                           for variant in sample_sizes.keys()]
    
            df = pd.concat(sampled_dfs)
        
        else:
            df = ado_df
    
    embedding_matrix = np.vstack(df['embedding'])
    info_df = df[["seq_id", "variant"]].copy().reset_index(drop=True)
        
    return save_as, info_df, embedding_matrix

def generate_umap_embedding(save_as, info_df, embedding_matrix, rnd_seed) -> str:    
    """
    Generates UMAP embedding from cluster_blstm.py embeddings.
    """
    # Generate UMAP embeddings
    umap_embeddings = umap.UMAP(n_neighbors=25, 
                       n_components=2,
                       min_dist=1, 
                       init='random', 
                       random_state=rnd_seed,
                       verbose=True).fit(embedding_matrix).embedding_
    
    # Save UMAP embeddings
    umap_embeddings_df = pd.DataFrame(umap_embeddings, columns=['DIM_1', 'DIM_2']).reset_index(drop=True)
    umap_df = pd.concat([info_df, umap_embeddings_df], axis=1)
    save_as = save_as + f"_umap_coordinates.csv"
    umap_df.to_csv(save_as, index=False)

    return save_as

def generate_tsne_embedding(save_as, info_df, embedding_matrix, rnd_seed) -> str:
    """
    Generates t-SNE embedding from cluster_blstm.py embeddings.
    """
    # Generate t-SNE embeddings
    tsne_embeddings = TSNE(n_components=2, 
                           learning_rate='auto', 
                           perplexity=200,
                           random_state=rnd_seed,
                           verbose=1).fit_transform(embedding_matrix)
    
    # Save t-SNE embeddings
    tsne_embeddings_df = pd.DataFrame(tsne_embeddings, columns=['DIM_1', 'DIM_2']).reset_index(drop=True)
    tsne_df = pd.concat([info_df, tsne_embeddings_df], axis=1)
    save_as = save_as + f"_tsne_coordinates.csv"
    tsne_df.to_csv(save_as, index=False)

    return save_as

def plot_from_embedding(csv_file, type):
    df = pd.read_csv(csv_file, sep=',', header=0)

    # Custom color mapping for specified variants
    cmap = {'Alpha': 'black', 'Delta': 'tab:blue', 'Omicron': 'tab:green'}
    variant_labels = sorted(df["variant"].unique())

    variant_colors = {}
    color_index = 0
    for variant in variant_labels:
        if variant in cmap:
            variant_colors[variant] = cmap[variant]

    df['colors'] = [variant_colors[variant] for variant in df['variant']]

    # Legend handles
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=variant_colors[variant], label=variant) for variant in variant_labels]

    # Create a scatter plot
    plt.figure(figsize=(16,9))
    plt.rcParams['font.family'] = 'sans-serif'
    plt.scatter(df['DIM_1'], df['DIM_2'], c=df['colors'], s=60, edgecolor='w', alpha=0.5)
    plt.xlabel(f'{type} Dimension 1')
    plt.ylabel(f'{type} Dimension 2')
    plt.legend(handles=legend_handles, loc='upper right')
    plt.tight_layout()
    plt.savefig(csv_file.replace('_coordinates.csv', '_plot.png'), format='png')
    plt.savefig(csv_file.replace('_coordinates.csv', '_plot.pdf'), format='pdf')

def plot_12_embeddings(csv_files, type):
    """
    Plots 12 UMAP or t-SNE subplots into a single plot from generated embeddings.
    """
    n_rows, n_cols = 3, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16*n_rows, 9*n_cols))
    plt.rcParams['font.family'] = 'sans-serif'
    axes = axes.flatten()

    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file, sep=',', header=0)

        # Custom color mapping for specified variants
        cmap = {'Alpha': 'black', 'Delta': 'tab:blue', 'Omicron': 'tab:green'}
        variant_labels = sorted(df["variant"].unique())

        variant_colors = {}
        color_index = 0
        for variant in variant_labels:
            if variant in cmap:
                variant_colors[variant] = cmap[variant]

        df['colors'] = [variant_colors[variant] for variant in df['variant']]

        # Legend handles
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=variant_colors[variant], label=variant) for variant in variant_labels]

        # Plot the UMAP embeddings on the subplot
        ax = axes[i]
        scatter = ax.scatter(df['DIM_1'], df['DIM_2'], c=df['colors'], s=40, edgecolor='w', alpha=0.5)

        # Set the font size for axes labels
        if i >= 8: 
            ax.set_xlabel(f'{type} Dimension 1', fontsize=30)
        else:
            ax.set_xticklabels([])

        if i % 4 == 0:
            ax.set_ylabel(f'{type} Dimension 2', fontsize=30)
        else:
            ax.set_yticklabels([])

        ax.tick_params(axis='both', which='major', labelsize=30)
        ax.legend(handles=legend_handles, loc='upper right', fontsize=22)

    plt.tight_layout()
    plt.savefig(csv_files[0].replace(f"_iter0_{type.lower()}_coordinates.csv", f"_12iterations_{type.lower()}_plot.png"), dpi=300, format='png')
    plt.savefig(csv_files[0].replace(f"_iter0_{type.lower()}_coordinates.csv", f"_12iterations_{type.lower()}_plot.pdf"), format='pdf')

if __name__=="__main__":

    # now = datetime.datetime.now()
    # date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    # run_dir = os.path.join(os.path.dirname(__file__), f'../../../results/run_results/clustering/plot_rbd_cluster-{date_hour_minute}')
    # os.makedirs(run_dir, exist_ok = True)

    # Add logging configuration
    # log_file = os.path.join(run_dir, 'memory-usage.log')
    # logging.basicConfig(filename=log_file,
    #                     level=logging.INFO,
    #                     format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    #                     datefmt='%Y-%m-%d %H:%M:%S')

    # logging.info(f"Init memory usage: {memory_usage()}")
    # data_dir = os.path.join(os.path.dirname(__file__), '../../../data/pickles')
    # pickle_file = os.path.join(data_dir, "spikeprot0528.clean.uniq.noX.RBD_variants_clustering_esm_blstm.pkl")
    # pickle_file = os.path.join(data_dir, "spikeprot0528.clean.uniq.noX.RBD_variants_clustering_bert_blstm_dms_binding.pkl")
    # pickle_file = os.path.join(data_dir, "spikeprot0528.clean.uniq.noX.RBD_variants_clustering_bert_blstm_dms_expression.pkl")
    # pickle_file = os.path.join(data_dir, "spikeprot0528.clean.uniq.noX.RBD_variants_clustering_bert_blstm_esm.pkl")
    # logging.info(f"Using this pickle: {pickle_file}")
    
    # print(f"Extracting embeddings")
    # logging.info(f"Pre embedding extraction memory usage: {memory_usage()}")
    # all_data = extract_embedding_pickle(pickle_file)
    # logging.info(f"Post embedding extraction memory usage: {memory_usage()}")

    # # Whole dataset maps
    # ado = True
    # whole = True
    # save_as, info_df, embedding_matrix = sample_embedding_pickle(run_dir, all_data, whole, ado, "whole")

    # print(f"Plotting 2D UMAP - All")
    # logging.info(f"Pre 2D UMAP memory usage: {memory_usage()}")
    # umap_csv = generate_umap_embedding(save_as, info_df, embedding_matrix, 0) # can also set rnd_seed here
    # plot_from_embedding(umap_csv, 'UMAP')
    # logging.info(f"2D UMAP memory usage: {memory_usage()}")

    # print(f"Plotting T-SNE - All")
    # logging.info(f"Pre T-SNE memory usage: {memory_usage()}")
    # tsne_csv = generate_tsne_embedding(save_as, info_df, embedding_matrix, 0)
    # plot_from_embedding(tsne_csv, 'tSNE')
    # logging.info(f"Post T-SNE memory usage: {memory_usage()}")

    # # Iteration maps
    # ado = True
    # whole = False

    # for i in range(12):
    #     rnd_seed = i  # Change the seed for each iteration to ensure different samples
    #     save_as, info_df, embedding_matrix = sample_embedding_pickle(run_dir, all_data, whole, ado, str(i), rnd_seed)
    #     print(info_df["variant"].value_counts())
    #     save_as = save_as + f'_iter{i}'

    #     print(f"Plotting 2D UMAP - Iteration {i}")
    #     logging.info(f"Pre 2D UMAP memory usage: {memory_usage()}")
    #     umap_csv = generate_umap_embedding(save_as, info_df, embedding_matrix, rnd_seed) # can also set rnd_seed here
    #     plot_from_embedding(umap_csv, 'UMAP')
    #     logging.info(f"2D UMAP memory usage: {memory_usage()}")

    #     print(f"Plotting T-SNE - Iteration {i}")
    #     logging.info(f"Pre T-SNE memory usage: {memory_usage()}")
    #     tsne_csv = generate_tsne_embedding(save_as, info_df, embedding_matrix, rnd_seed)
    #     plot_from_embedding(tsne_csv, 'tSNE')
    #     logging.info(f"Post T-SNE memory usage: {memory_usage()}")

    run_dir = os.path.join(os.path.dirname(__file__), f'../run_results/clustering/plot_cluster-2023-11-30_16-44')
    umap_csv = os.path.join(run_dir, "RBD_variants_clustering_esm_blstm_umap_coordinates.csv")
    tsne_csv = os.path.join(run_dir, "RBD_variants_clustering_esm_blstm_tsne_coordinates.csv")
    plot_from_embedding(umap_csv, 'UMAP')
    plot_from_embedding(tsne_csv, 'tSNE')

    umap_list = [os.path.join(run_dir, f"RBD_variants_clustering_esm_blstm_a22073_d21887_o22399_iter{i}_umap_coordinates.csv") for i in range(12)]
    plot_12_embeddings(umap_list, 'UMAP')
    
    tsne_list = [os.path.join(run_dir, f"RBD_variants_clustering_esm_blstm_a22073_d21887_o22399_iter{i}_tsne_coordinates.csv") for i in range(12)]
    plot_12_embeddings(tsne_list, 'tSNE')

