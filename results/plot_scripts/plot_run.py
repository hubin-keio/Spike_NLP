"""Plot model run result csv file"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def plot_run(csv_name: str, save: bool = True) -> Figure:
    '''
    Generate a single figure with subplots for training loss and training accuracy
    from the model run csv file.
    '''
    df = pd.read_csv(csv_name)
    df.columns = df.columns.str.strip()

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))

    # Plot Training Loss
    train_loss_line = ax1.plot(df['epoch'], df['train_loss'],
                               color='red', label='Train Loss')
    test_loss_line = ax1.plot(df['epoch'], df['test_loss'],
                              color='orange', label='Test Loss')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')

    # Plot Training Accuracy
    train_accuracy_line = ax2.plot(df['epoch'], df['train_accuracy'],
                                   color='blue', label='Train Accuracy')
    test_accuracy_line = ax2.plot(df['epoch'], df['test_accuracy'],
                                  color='green', label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1) 
    ax2.legend(loc='lower right')

    plt.style.use('ggplot')
    plt.tight_layout()

    if save:
        combined_fname = csv_name.replace('.csv', '_loss_acc.png')
        plt.savefig(combined_fname)

    return fig