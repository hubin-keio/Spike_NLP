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
from sklearn.manifold import TSNE
from contextlib import redirect_stdout
from tqdm import tqdm

logger = logging.getLogger(__name__)

def memory_usage() -> str:
    """ Returns string of current memory usage of Python process """
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
    save_as = save_as = os.path.join(run_dir, f"RBD_variants_clustering_esm_blstm")

    if not whole:
        save_as = save_as + f"{iteration}"

    if ado:
        ado_df = df[df['variant'].isin(["Alpha", "Delta", "Omicron"])]

        if not whole:
            sample_sizes = {"Alpha": int(1 * (df['variant'] == 'Alpha').sum()), 
                            "Delta": int(0.3 * (df['variant'] == 'Delta').sum()),  
                            "Omicron": int(0.2 * (df['variant'] == 'Omicron').sum())} 

            save_as = save_as + f"_a{sample_sizes['Alpha']}_d{sample_sizes['Delta']}_o{sample_sizes['Omicron']}"

            sampled_dfs = [ado_df[ado_df['variant'] == variant].sample(n=sample_sizes[variant], random_state=rnd_seed)
                        for variant in sample_sizes.keys()]
        
            df = pd.concat(sampled_dfs)
        
        else:
            df = ado_df

    variant_labels = sorted(df["variant"].unique())
    print(variant_labels)
    print(len(df['seq_id'])) 
    
    embedding_matrix = np.vstack(df['embedding'])
    print(embedding_matrix.shape)

    return save_as, embedding_matrix, df["variant"], variant_labels

def plot_umap_with_circle(save_as, embedding_matrix, all_variants, variant_labels, rnd_seed):
    """
    Plots UMAP with circle overlay of cluster esm and blstm embedding data.
    """
    save_as = save_as + f"_umap_circle.png"

    mapper = umap.UMAP(n_neighbors=25, 
                       n_components=2,
                       min_dist=0.1, 
                       init='random', 
                       random_state=rnd_seed,
                       verbose=True).fit(embedding_matrix)
    
    umap_embeddings = mapper.embedding_  

    # Save UMAP embeddings
    umap_df = pd.DataFrame(umap_embeddings, columns=['UMAP_1', 'UMAP_2'])
    umap_df['variant'] = all_variants  
    umap_csv_save_as = save_as + f"_umap_coordinates.csv"
    umap_df.to_csv(umap_csv_save_as, index=False)
    
    # Color mapping based on variant label
    cmap = sns.color_palette("tab20", len(variant_labels))
    variant_colors = {variant: cmap[i] for i, variant in enumerate(variant_labels)}
    colors = [variant_colors[variant] for variant in all_variants]

    # Legend handles
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=variant_colors[variant], label=variant) for variant in variant_labels]
    
    # Equation of the circle: (x)^(2) + (y-8.5)^(2) = 2.5^(2)
    # Generate points for the circle
    circle_theta = np.linspace(0, 2*np.pi, 100)
    circle_x = 0 + 2.5 * np.cos(circle_theta)
    circle_y = 8.5 + 2.5 * np.sin(circle_theta)

    # Create a scatter plot w/ circle overlay
    plt.figure(figsize=(10, 8))
    plt.plot(circle_x, circle_y, color='red', linestyle='--', linewidth=2)
    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=colors, s=20, edgecolor='w', alpha=0.6)
    plt.title('UMAP With Circle Visualization of Embeddings')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.16, 1))
    plt.tight_layout()
    plt.savefig(save_as)
    plt.close()

def plot_umap(save_as, embedding_matrix, all_variants, variant_labels, rnd_seed):
    """
    Plots UMAP of cluster esm and blstm embedding data.
    """
    save_as = save_as + f"_umap.png"

    mapper = umap.UMAP(n_neighbors=25, 
                       n_components=2,
                       min_dist=0.1, 
                       init='random', 
                       random_state=rnd_seed,
                       verbose=True).fit(embedding_matrix)
    
    umap_embeddings = mapper.embedding_  

    # Save UMAP embeddings
    umap_df = pd.DataFrame(umap_embeddings, columns=['UMAP_1', 'UMAP_2'])
    umap_df['variant'] = all_variants  
    umap_csv_save_as = save_as + f"_umap_coordinates.csv"
    umap_df.to_csv(umap_csv_save_as, index=False)
    
    # Color mapping based on variant label
    cmap = sns.color_palette("tab20", len(variant_labels))
    variant_colors = {variant: cmap[i] for i, variant in enumerate(variant_labels)}
    colors = [variant_colors[variant] for variant in all_variants]

    # Legend handles
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=variant_colors[variant], label=variant) for variant in variant_labels]

    # Create a scatter plot w/ circle overlay
    plt.figure(figsize=(10, 8))
    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=colors, s=20, edgecolor='w', alpha=0.6)
    plt.title('UMAP Visualization of Embeddings')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.16, 1))
    plt.tight_layout()
    plt.savefig(save_as)
    plt.close()

def plot_tsne(save_as, embedding_matrix, all_variants, variant_labels, rnd_seed):
    """
    Plots t-SNE of cluster esm and blstm embedding data.
    """
    save_as = save_as + f"_tsne.png"

    tsne_embeddings = TSNE(n_components=2, 
                           learning_rate='auto', 
                           perplexity=200,
                           random_state=rnd_seed,
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
    plt.savefig(save_as)
    plt.close()

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
    pickle_file = os.path.join(data_dir, "spike_variants/spikeprot0528.clean.uniq.noX.RBD_variants_clustering_esm_blstm.pkl")
    logging.info(f"Using this pickle: {pickle_file}")
    
    print(f"Extracting embeddings")
    logging.info(f"Pre embedding extraction memory usage: {memory_usage()}")
    all_data = extract_embedding_pickle(pickle_file)
    logging.info(f"Post embedding extraction memory usage: {memory_usage()}")

    # # Whole dataset maps
    # ado = True
    # whole = True
    # save_as, embedding_matrix, variants, variant_labels = sample_embedding_pickle(run_dir, all_data, whole, ado, "whole")

    # print(f"Plotting 2D UMAP - All")
    # logging.info(f"Pre 2D UMAP memory usage: {memory_usage()}")
    # plot_umap_with_circle(save_as, embedding_matrix, variants, variant_labels, 0) # can also set rnd_seed here
    # logging.info(f"2D UMAP memory usage: {memory_usage()}")

    # print(f"Plotting T-SNE - All")
    # logging.info(f"Pre T-SNE memory usage: {memory_usage()}")
    # plot_tsne(save_as, embedding_matrix, variants, variant_labels, 0) # can also set rnd_seed here
    # logging.info(f"Post T-SNE memory usage: {memory_usage()}")

    # Iteration maps
    ado = True
    whole = False
    iterations = 10

    for i in range(iterations):
        rnd_seed = i  # Change the seed for each iteration to ensure different samples
        save_as, embedding_matrix, variants, variant_labels = sample_embedding_pickle(run_dir, all_data, whole, ado, str(i), rnd_seed)

        print(f"Plotting 2D UMAP - Iteration {i}")
        logging.info(f"Pre 2D UMAP memory usage: {memory_usage()}")
        plot_umap(save_as, embedding_matrix, variants, variant_labels, 0) # can also set rnd_seed here
        logging.info(f"2D UMAP memory usage: {memory_usage()}")

        print(f"Plotting T-SNE - Iteration {i}")
        logging.info(f"Pre T-SNE memory usage: {memory_usage()}")
        plot_tsne(save_as, embedding_matrix, variants, variant_labels, 0) # can also set rnd_seed here
        logging.info(f"Post T-SNE memory usage: {memory_usage()}")