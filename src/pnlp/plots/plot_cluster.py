#!/usr/bin/env python
"""
Plot 2d and 3d UMAP, and T-SNE plots from embeddings generateed from clustering.py.
"""

#import umap
import os
import ast
import umap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.manifold import TSNE

def extract_embedding(tsv_file:str):
    embeddings =  []
    variants = []
    variant_labels = set()

    chunks = pd.read_csv(tsv_file, delimiter="\t", chunksize=100)  # Changed the chunksize to a larger value
    for chunk in chunks:
        print(f"\tLoading in chunk... {chunk['variant'].tolist()}", flush=True)
        embeddings.extend(chunk["embedding"].apply(ast.literal_eval).tolist())
        variants.extend(chunk["variant"].tolist())
        variant_labels.update(chunk["variant"].tolist())

    embedding_matrix = np.stack(embeddings)  
    print(embedding_matrix.shape)

    return embedding_matrix, variants, sorted(list(variant_labels))

def plot_umap(tsv_file, embedding_matrix, all_variants):
    """
    tsv_file: input data - includes seq_id, variant, and flattened embeddings
    embedding_matrix: stacked flattened embeddings from each entry
    all_variants: variants from tsv_file from each entry 
    variant_labels: all the unique possible labels 
    """
    mapper = umap.UMAP(n_neighbors=25, 
                       n_components=2,
                       min_dist = 0.1, 
                       init = 'random', 
                       n_jobs= -1,
                       random_state=0).fit(embedding_matrix)

    # Create a scatter plot
    p = umap.plot.points(mapper, labels=all_variants, theme='fire')
    umap.plot.plt.title('UMAP Visualization of Embeddings')
    save_as = tsv_file.replace(".tsv", "_umap.png")
    umap.plot.plt.savefig(save_as)

def plot_umap_plt(tsv_file, embedding_matrix, all_variants, variant_labels):
    """
    tsv_file: input data - includes seq_id, variant, and flattened embeddings
    embedding_matrix: stacked flattened embeddings from each entry
    all_variants: variants from tsv_file from each entry 
    variant_labels: all the unique possible labels 
    """
    umap_embeddings = umap.UMAP(n_neighbors=50, 
                                n_components=2,
                                min_dist = 0.1, 
                                init = 'random', 
                                n_jobs= -1,
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

    save_as = tsv_file.replace(".tsv", "_umap.png")
    plt.savefig(save_as)
    
def plot_3d_umap_plt(tsv_file, embedding_matrix, all_variants, variant_labels):
    """
    tsv_file: input data - includes seq_id, variant, and flattened embeddings
    embedding_matrix: stacked flattened embeddings from each entry
    all_variants: variants from tsv_file from each entry 
    variant_labels: all the unique possible labels 
    """
    umap_embeddings = umap.UMAP(n_neighbors=50, 
                                n_components=3, 
                                min_dist = 0.1,
                                init='random', 
                                n_jobs= -1,
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
    
    save_as = tsv_file.replace(".tsv", "_3d_umap.png")
    plt.savefig(save_as)

def plot_tsne(tsv_file, embedding_matrix, all_variants, variant_labels):
    """
    tsv_file: input data - includes seq_id, variant, and flattened embeddings
    embedding_matrix: stacked flattened embeddings from each entry
    all_variants: variants from tsv_file from each entry 
    variant_labels: all the unique possible labels 
    """
    tsne_embeddings = TSNE(n_components=2, 
                            learning_rate='auto', 
                            init='random', 
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
    
    save_as = tsv_file.replace(".tsv", "_tsne.png")
    plt.savefig(save_as)

if __name__=="__main__":
    data_dir = os.path.join(os.path.dirname(__file__), '../../../results')
    tsv_file = os.path.join(data_dir, "plot_results/rbd_variant_seq_sampled_ADO_1200_embedding.tsv")
    print(f"Extracting embeddings")
    embedding_matrix, variants, variant_labels = extract_embedding(tsv_file)
    print(f"Plotting 2D UMAP")
    plot_umap(tsv_file, embedding_matrix, variants, variant_labels)
    print(f"Plotting 3D UMAP")
    plot_3d_umap(tsv_file, embedding_matrix, variants, variant_labels)
    print(f"Plotting T-SNE")
    plot_tsne(tsv_file, embedding_matrix, variants, variant_labels)
    print(f"Done!")
