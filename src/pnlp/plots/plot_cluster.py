#!/usr/bin/env python
"""
Plot 2d and 3d UMAP, and T-SNE plots from embeddings generateed from clustering.py.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # set which GPU to use for tsnecuda

import datetime
import umap
import umap.plot
import logging
import pickle
import psutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from tsnecuda import TSNE # gpu tsne
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D 

logger = logging.getLogger(__name__)

def extract_embedding_pickle(pickle_file:str, ado:bool):
    """
    Extracts seq_ids, variants, and embeddings from embedding pickle.
    If ado, data is filtered to just 'Alpha', 'Delta', 'Omicron' variants.
    """
    with open(pickle_file, 'rb') as f:
        pkl_seq_ids, pkl_variants, pkl_embeddings = pickle.load(f)

    if ado:
        seq_ids, variants, embeddings = [], [], []
        for i, variant in enumerate(pkl_variants):
            if variant in ['Alpha', 'Delta', 'Omicron']:
                seq_ids.append(pkl_seq_ids[i])
                variants.append(variant)
                embeddings.append(pkl_embeddings[i])
    else:
        seq_ids, variants, embeddings = pkl_seq_ids, pkl_variants, pkl_embeddings

    variant_labels = sorted(set(variants))
    print(variant_labels)
    print(len(seq_ids))
    
    embedding_matrix = np.vstack(embeddings)
    print(embedding_matrix.shape)

    return embedding_matrix, variants, variant_labels

def plot_umap(pickle_file, embedding_matrix, all_variants):
    """
    pickle_file: input data - includes seq_id, variant, and flattened embeddings
    embedding_matrix: stacked flattened embeddings from each entry
    all_variants: variants from pickle_file from each entry 
    """
    mapper = umap.UMAP(n_neighbors=25, 
                       n_components=2,
                       min_dist = 0.1, 
                       init = 'random', 
                       random_state=0).fit(embedding_matrix)

    # Create a scatter plot
    p = umap.plot.points(mapper, labels=np.array(all_variants), color_key_cmap='Paired', background='black')
    plt.title('UMAP Visualization of Embeddings')
    plt.tight_layout()

    save_as = pickle_file.replace(".pkl", "_umap.png")
    plt.savefig(save_as)

def plot_umap_plt(pickle_file, embedding_matrix, all_variants, variant_labels):
    """
    pickle_file: input data - includes seq_id, variant, and flattened embeddings
    embedding_matrix: stacked flattened embeddings from each entry
    all_variants: variants from pickle_file from each entry 
    variant_labels: all the unique possible labels 
    """
    umap_embeddings = umap.UMAP(n_neighbors=50, 
                                n_components=2,
                                min_dist = 0.1, 
                                init = 'random', 
                                random_state=0).fit_transform(embedding_matrix)

    cmap = sns.color_palette("tab20", len(variant_labels))
    variant_colors = {variant: cmap[i] for i, variant in enumerate(variant_labels)}
    colors = [variant_colors[variant] for variant in all_variants]

    # Legend handles
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=variant_colors[variant], label=variant) for variant in variant_labels]

    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=colors, s=20, edgecolor='w', alpha=0.6)
    plt.title('UMAP Visualization of Embeddings')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.16, 1))
    plt.tight_layout()

    save_as = pickle_file.replace(".pkl", "_umap_plt.png")
    plt.savefig(save_as)
    
def plot_3d_umap(pickle_file, embedding_matrix, all_variants, variant_labels):
    """
    pickle_file: input data - includes seq_id, variant, and flattened embeddings
    embedding_matrix: stacked flattened embeddings from each entry
    all_variants: variants from pickle_file from each entry 
    variant_labels: all the unique possible labels 
    """
    umap_embeddings = umap.UMAP(n_neighbors=50, 
                                n_components=3, 
                                min_dist = 0.1,
                                init='random', 
                                random_state=0).fit_transform(embedding_matrix)

    cmap = sns.color_palette("tab20", len(variant_labels))
    variant_colors = {variant: cmap[i] for i, variant in enumerate(variant_labels)}
    colors = [variant_colors[variant] for variant in all_variants]

    # Legend handles
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=variant_colors[variant], label=variant) for variant in variant_labels]

    # Create a scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot
    scatter = ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], umap_embeddings[:, 2], c=colors, s=20, edgecolor='w', alpha=0.6)
    ax.set_title('3D UMAP Visualization of Embeddings')
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.set_zlabel('UMAP Dimension 3')
    plt.legend(handles=legend_handles, loc='upper right')
    plt.tight_layout()
    
    save_as = pickle_file.replace(".pkl", "_3d_umap.png")
    plt.savefig(save_as)

def plot_tsne(pickle_file, embedding_matrix, all_variants, variant_labels):
    """
    pickle_file: input data - includes seq_id, variant, and flattened embeddings
    embedding_matrix: stacked flattened embeddings from each entry
    all_variants: variants from pickle_file from each entry 
    variant_labels: all the unique possible labels 
    """
    # tsne_embeddings = TSNE(n_components=2, 
    #                        learning_rate=0, 
    #                        perplexity=25).fit_transform(embedding_matrix)
    tsne_embeddings = TSNE(n_components=2, 
                           learning_rate='auto', 
                           init='random', 
                           perplexity=3).fit_transform(embedding_matrix)
    
    cmap = sns.color_palette("tab20", len(variant_labels))
    variant_colors = {variant: cmap[i] for i, variant in enumerate(variant_labels)}
    colors = [variant_colors[variant] for variant in all_variants]

    # Legend handles
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=variant_colors[variant], label=variant) for variant in variant_labels]

    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=colors, s=20, edgecolor='w', alpha=0.6)
    plt.title("t-SNE Visualization of Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(handles=legend_handles, loc='upper right')
    plt.tight_layout()
    
    save_as = pickle_file.replace(".pkl", "_tsne.png")
    plt.savefig(save_as)

def memory_usage() -> str:
    """ Returns string of current memory usage of Python process """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB"

if __name__=="__main__":

    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(os.path.dirname(__file__), f'../../../results/clustering/plot_cluster-{date_hour_minute}')
    os.makedirs(run_dir, exist_ok = True)

    # Add logging configuration
    log_file = os.path.join(run_dir, 'memory-usage.log')
    logging.basicConfig(filename=log_file,
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    logging.info(f"Init memory usage: {memory_usage()}")
    data_dir = os.path.join(os.path.dirname(__file__), '../../../data')
    pickle_file = os.path.join(data_dir, "spike_variants/spikeprot0528.clean.uniq.noX.RBD_variants_cluster_320_embedding.pkl")
    logging.info(f"Using this pickle: {pickle_file}")
    
    print(f"Extracting embeddings")
    logging.info(f"Pre embedding extraction memory usage: {memory_usage()}")
    ado = True
    embedding_matrix, variants, variant_labels = extract_embedding_pickle(pickle_file, ado)
    logging.info(f"Post embedding extraction memory usage: {memory_usage()}")

    # print(f"Plotting 2D UMAP")
    # plot_umap(pickle_file, embedding_matrix, variants)
    # logging.info(f"2D UMAP memory usage: {memory_usage()}")
    # plot_umap_plt(pickle_file, embedding_matrix, variants, variant_labels)
    # logging.info(f"2D UMAP memory usage: {memory_usage()}")

    # print(f"Plotting 3D UMAP")
    # plot_3d_umap(pickle_file, embedding_matrix, variants, variant_labels)
    # logging.info(f"3D UMAP memory usage: {memory_usage()}")

    print(f"Plotting T-SNE")
    logging.info(f"Pre T-SNE memory usage: {memory_usage()}")
    plot_tsne(pickle_file, embedding_matrix, variants, variant_labels)
    logging.info(f"Post T-SNE memory usage: {memory_usage()}")
    
