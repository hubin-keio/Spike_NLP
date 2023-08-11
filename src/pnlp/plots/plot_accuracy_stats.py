"""Plot accuracy stats for model prediction csv file"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def plot_top_predictions(pred_dict:dict, csv_name:str, save: bool=True) -> tuple:
    """
    Generate figure object of accuracy stats using model incorrect prediction dictionary.
    """
    fig = plt.figure(figsize=(10, 6))

    # Extract the epoch numbers for x-axis labels
    epochs = list(range(1, len(pred_dict[next(iter(pred_dict))]) + 1))

    # Sort each epoch column in descending order and take the top 5 rows
    for epoch in range(len(epochs)):
        sorted_data = sorted(pred_dict.items(), key=lambda item: item[1][epoch], reverse=True)
        top_evp_keys = [item[0] for item in sorted_data[:5]] # change 5 to be top whatever
        for evp in top_evp_keys:
            counts = pred_dict[evp]
            plt.plot(epochs, counts, marker='o', label=evp)

    # Plotting things
    plt.xlabel('Epoch')
    plt.ylabel('Count')
    plt.title("Top 5 counts per incorrect 'expected_aa->predicted_aa' across epochs")
    plt.xticks(epochs)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.style.use('ggplot')
    plt.tight_layout()
    plt.grid(True)

    if save:
        fname = csv_name.replace('.csv', '.png')
        plt.savefig(fname)
    
    return fig, csv_name 

def plot_pred_hist(pred_dict:dict, csv_name:str, save: bool=True) -> tuple:
    """
    Generate a histogram of the last epoch predictions.
    """
    # Get counts > 0 from last epoch, then organize in descending order
    labels_counts = [(key, pred_dict[key][-1]) for key in pred_dict if pred_dict[key][-1] > 0]
    labels_counts.sort(key=lambda x: x[1], reverse=True) 
    labels = [i[0] for i in labels_counts]
    counts = [i[1] for i in labels_counts]
    
    fig = plt.figure(figsize=(12,6))
 
    # Plotting histogram things
    plt.bar(labels, counts, color='skyblue')
    plt.xlabel('Amino Acid Predictions')
    plt.ylabel('Counts')
    plt.title('Histogram of Incorrect Amino Acid Prediction Counts')
    plt.xticks(rotation=45, ha='right')
    plt.style.use('ggplot')
    plt.tight_layout()

    if save:
        fname = csv_name.replace('.csv', '_last_pred_hist.png')
        plt.savefig(fname)
    
    return fig, csv_name

