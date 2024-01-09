#!/usr/bin/env python
"""
Plot 2d and 3d UMAP, and T-SNE plots from embeddings generateed from clustering.py.
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
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from contextlib import redirect_stdout
from tqdm import tqdm
from PIL import Image
from adjustText import adjust_text

logger = logging.getLogger(__name__)

def memory_usage() -> str:
    """ Returns string of current memory usage of Python process """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB"

def extract_embedding_pickle(pickle_file:str):
    """
    Extracts seq_ids, host, variants, and embeddings from embedding pickle.
    """
    with open(pickle_file, 'rb') as f:
        pkl_seq_ids, pkl_variants, pkl_hosts, pkl_embeddings = pickle.load(f)

    df = pd.DataFrame({"seq_id": pkl_seq_ids,
                       "type": pkl_hosts,
                       "host": pkl_variants,
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
    save_as = save_as = os.path.join(run_dir, f"RBD_variants_clustering_esm_blstm")

    type_labels = sorted(df["type"].unique())
    print(type_labels)
    
    host_labels = sorted(df["host"].unique())
    print(host_labels)
    
    id_labels = sorted(df["seq_id"].unique())
    
    
    embedding_matrix = np.vstack(df['embedding'])
    print(embedding_matrix.shape)

    return save_as, embedding_matrix, df["type"], type_labels, df["host"], host_labels, df["seq_id"], id_labels


def plot_tsne(save_as, embedding_matrix, all_types, type_labels, all_hosts, host_labels, all_id, id_labels, rnd_seed):
    """
    Plots t-SNE of cluster esm and blstm embedding data.
    """
    save_as = save_as + f"_tsne"

    tsne_embeddings = TSNE(n_components=2, 
                           learning_rate='auto', 
                           perplexity=5,
                           random_state=rnd_seed,
                           verbose=1).fit_transform(embedding_matrix)

    # Color mapping based on variant label
    cmap = sns.color_palette("tab10", len(type_labels))
    type_colors = {type: cmap[i] for i, type in enumerate(type_labels)}
    host_markers = {'Sars-Cov-1': '+', 'MERS': 's', 'Bat Virus': '^', 'Pangolin Virus': 'x', 'Sars-Cov-2': 'o', 'Hibecovirus': 'H'}
    markers = [host_markers[host] for host in all_hosts]
    colors = [type_colors[type] for type in all_types]

    # Legend handles for types
    legend_handles_type = [Line2D([0], [0], marker='o', color='w', markerfacecolor=type_colors[type], label=type) for type in type_labels]

    # Legend handles for hosts
    legend_handles_host = [Line2D([0], [0], marker=host_markers[host], color='grey', markerfacecolor='grey', linestyle='', label=host) for host in host_labels]

    # Combine legend handles
    legend_handles = legend_handles_type + legend_handles_host

    # Set the figure size
    figure_width_inches = 180 / 25.4
    plt.figure(figsize=(figure_width_inches, figure_width_inches))
    
    for host_marker in set(markers):
        indices = [i for i, marker in enumerate(markers) if marker == host_marker]
        plt.scatter(tsne_embeddings[indices, 0], tsne_embeddings[indices, 1], c=[colors[i] for i in indices],
                    s=85, edgecolor='w', alpha=0.6, marker=host_marker, label=host_marker)
                    
        # Add labels for each point with type, host, and ID
#        for i in indices:
#            texts = [plt.text(tsne_embeddings[i, 0], tsne_embeddings[i, 1],
#                     f"{all_id[i]}",
#                     fontsize=8, ha='right', va='bottom', color='black')]
#            adjust_text(texts, autoalign=True)
                    
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(handles=legend_handles, loc='lower right')

    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(100))  # Set to 2 for every other tick
    ax.yaxis.set_major_locator(MultipleLocator(80))  # Set to 2 for every other tick

    plt.tight_layout()
    plt.savefig(save_as + ".pdf", format="pdf")
    plt.close()

if __name__=="__main__":

    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    run_dir = '/Users/vli/Desktop/beta_cora/results'
    os.makedirs(run_dir, exist_ok = True)

    # Add logging configuration
    log_file = os.path.join(run_dir, 'memory-usage.log')
    logging.basicConfig(filename=log_file,
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    logging.info(f"Init memory usage: {memory_usage()}")
    pickle_file = '/Users/vli/Desktop/beta_cora/new_hibecovirus.pkl'
    logging.info(f"Using this pickle: {pickle_file}")
    
    print(f"Extracting embeddings")
    logging.info(f"Pre embedding extraction memory usage: {memory_usage()}")
    all_data = extract_embedding_pickle(pickle_file)
    logging.info(f"Post embedding extraction memory usage: {memory_usage()}")

    # Whole dataset maps
    ado = True
    whole = True
    save_as, embedding_matrix, types, type_labels, hosts, host_labels, ids, id_labels = sample_embedding_pickle(run_dir, all_data, whole, ado, "whole")

    print(f"Plotting T-SNE - All")
    logging.info(f"Pre T-SNE memory usage: {memory_usage()}")
    plot_tsne(save_as, embedding_matrix, types, type_labels, hosts, host_labels, ids, id_labels, 0) # can also set rnd_seed here
    logging.info(f"Post T-SNE memory usage: {memory_usage()}")
