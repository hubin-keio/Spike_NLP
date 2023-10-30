#!/usr/bin/env python
"""
Plot 2d and 3d UMAP, and T-SNE plots from embeddings generateed from clustering.py.
"""

import os
import datetime
import logging
import pickle
import psutil
import numpy as np
import umap
import umap.plot
import seaborn as sns
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from tqdm import tqdm

logger = logging.getLogger(__name__)

def memory_usage() -> str:
    """ Returns string of current memory usage of Python process """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB"

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

def plot_umap(pickle_file, embedding_matrix, all_variants, variant_labels):
    """
    pickle_file: input data - includes seq_id, variant, and flattened embeddings
    embedding_matrix: stacked flattened embeddings from each entry
    all_variants: variants from pickle_file from each entry 
    variant_labels: all the unique possible labels 
    """
    mapper = umap.UMAP(n_neighbors=25, 
                       n_components=2,
                       min_dist=0.1, 
                       init='random', 
                       random_state=0,
                       verbose=True).fit(embedding_matrix)
    
    umap_embeddings = mapper.embedding_   # Fixed this line
    
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

    save_as = pickle_file.replace(".pkl", "_umap.png")
    plt.savefig(save_as)

def plot_tsne(pickle_file, embedding_matrix, all_variants, variant_labels):
    """
    pickle_file: input data - includes seq_id, variant, and flattened embeddings
    embedding_matrix: stacked flattened embeddings from each entry
    all_variants: variants from pickle_file from each entry 
    variant_labels: all the unique possible labels 
    """
    from sklearn.manifold import TSNE

    # TSNE embedding creation
    tsne_embeddings = TSNE(n_components=2, 
                           learning_rate='auto', 
                           perplexity=30,
                           verbose=1).fit_transform(embedding_matrix)

    # Color mapping based on variant label
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

def plot_gpu_tsne(pickle_file, embedding_matrix, all_variants, variant_labels):
    """
    pickle_file: input data - includes seq_id, variant, and flattened embeddings
    embedding_matrix: stacked flattened embeddings from each entry
    all_variants: variants from pickle_file from each entry 
    variant_labels: all the unique possible labels 
    """
    from tsnecuda import TSNE

    # TSNE embedding creation
    tsne_embeddings = TSNE(n_components=2, 
                           learning_rate='auto', 
                           perplexity=30,
                           verbose=1).fit_transform(embedding_matrix)

    # Color mapping based on variant label
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

    print(f"Plotting 2D UMAP")
    logging.info(f"Pre 2D UMAP memory usage: {memory_usage()}")
    plot_umap(pickle_file, embedding_matrix, variants, variant_labels)
    logging.info(f"2D UMAP memory usage: {memory_usage()}")

    print(f"Plotting T-SNE")
    logging.info(f"Pre T-SNE memory usage: {memory_usage()}")
    plot_tsne(pickle_file, embedding_matrix, variants, variant_labels)
    logging.info(f"Post T-SNE memory usage: {memory_usage()}")
    logging.info(f"Pre GPU T-SNE memory usage: {memory_usage()}")
    plot_gpu_tsne(pickle_file, embedding_matrix, variants, variant_labels)
    logging.info(f"Post GPU T-SNE memory usage: {memory_usage()}")
