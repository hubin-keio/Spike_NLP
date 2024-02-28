#!/usr/bin/env python
"""
Plot 2d UMAP, and t-SNE plots from AlphaSeq embeddings generated from cluster_blstm.py.
"""

import os
import datetime
import logging
import pickle
import psutil
import umap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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
    cmap = {'Q1': 'black', 
            'Q2': '#1f77b4',    # blue
            'Q3': '#d62728',    # red
            'Q4': '#f7b6d2'}    # pink

    variant_labels = sorted(df["variant"].unique())    
    variant_colors = {}
    for variant in variant_labels:
        if variant in cmap:
            variant_colors[variant] = cmap[variant]
            
    df['colors'] = [variant_colors[variant] for variant in df['variant']]

    # Legend handles
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=variant_colors[variant], label=variant) for variant in variant_labels]

    # Create a scatter plot
    plt.figure(figsize=(16,9))
    plt.rcParams['font.family'] = 'sans-serif'
    plt.scatter(df['DIM_1'], df['DIM_2'], c=df['colors'], s=20, edgecolor='w', alpha=0.7)
    plt.xlabel(f'{type} Dimension 1')
    plt.ylabel(f'{type} Dimension 2')
    plt.legend(handles=legend_handles, loc='upper right')
    plt.tight_layout()
    plt.savefig(csv_file.replace('_coordinates.csv', '_plot.pdf'), format='pdf')

if __name__=="__main__":

    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(os.path.dirname(__file__), f'../run_results/clustering/plot_alphaseq_cluster-{date_hour_minute}')
    os.makedirs(run_dir, exist_ok = True)

    # Add logging configuration
    log_file = os.path.join(run_dir, 'memory-usage.log')
    logging.basicConfig(filename=log_file,
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    logging.info(f"Init memory usage: {memory_usage()}")
    data_dir = os.path.join(os.path.dirname(__file__), '../../data/pickles')
    pickle_file = os.path.join(data_dir, "clean_avg_alpha_seq_clustering_esm_blstm.pkl")
    save_as = os.path.join(run_dir, os.path.basename(pickle_file).replace('.pkl', ''))
    logging.info(f"Using this pickle: {pickle_file}")
    
    print(f"Extracting embeddings")
    logging.info(f"Pre embedding extraction memory usage: {memory_usage()}")
    all_data = extract_embedding_pickle(pickle_file)
    embedding_matrix = np.vstack(all_data['embedding'])
    info_df = all_data[["seq_id", "variant"]].copy().reset_index(drop=True)
    logging.info(f"Post embedding extraction memory usage: {memory_usage()}")

    print(f"Plotting 2D UMAP - All")
    logging.info(f"Pre 2D UMAP memory usage: {memory_usage()}")
    umap_csv = generate_umap_embedding(save_as, info_df, embedding_matrix, 0) # can also set rnd_seed here
    plot_from_embedding(umap_csv, 'UMAP')
    logging.info(f"2D UMAP memory usage: {memory_usage()}")

    print(f"Plotting T-SNE - All")
    logging.info(f"Pre T-SNE memory usage: {memory_usage()}")
    tsne_csv = generate_tsne_embedding(save_as, info_df, embedding_matrix, 0) # can also set rnd_seed here
    plot_from_embedding(tsne_csv, 'tSNE')
    logging.info(f"Post T-SNE memory usage: {memory_usage()}")