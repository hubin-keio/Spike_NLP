#!/usr/bin/env python
"""
Plot 2d and 3d UMAP, and T-SNE plots from embeddings generateed from clustering.py.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # set which GPU to use for tsnecuda

import umap
import umap.plot
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tsnecuda import TSNE # gpu tsne
#from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D 

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
    embedding_matrix = embedding_matrix.reshape(embedding_matrix.shape[0], -1)
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
    tsne_embeddings = TSNE(n_components=2, 
                           learning_rate=0, 
                           perplexity=25).fit_transform(embedding_matrix)
    
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
    data_dir = os.path.join(os.path.dirname(__file__), '../../../data')
    pickle_file = os.path.join(data_dir, "spike_variants/spikeprot0528.clean.uniq.noX.RBD_variants_cluster_embedding.pkl")
    print(f"Extracting embeddings")
    ado = True
    embedding_matrix, variants, variant_labels = extract_embedding_pickle(pickle_file, ado)
    print(f"Plotting 2D UMAP")
    plot_umap(pickle_file, embedding_matrix, variants)
    plot_umap_plt(pickle_file, embedding_matrix, variants, variant_labels)
    print(f"Plotting 3D UMAP")
    plot_3d_umap(pickle_file, embedding_matrix, variants, variant_labels)
    print(f"Plotting T-SNE")
    plot_tsne(pickle_file, embedding_matrix, variants, variant_labels)
    print(f"Done!")
