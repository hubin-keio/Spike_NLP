"""Plot model run result csv file"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def plot_run(csv_name:str, save: bool=True) -> Figure:
    '''
    Generate figure object of loss and accuracy from both the train and loss in
    the model run csv file.
    '''
    df = pd.read_csv(csv_name)
    df.columns = df.columns.str.strip()

    # Set ggplot style
    plt.style.use('ggplot')

    fig, ax1 = plt.subplots()

    # plot loss
    train_loss_line = ax1.plot(df['epoch'], df['train_loss'],
                               color='red', label='Train Loss')
    test_loss_line = ax1.plot(df['epoch'], df['test_loss'],
                              color='orange', label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    # plot accuracy
    ax2 = ax1.twinx()
    train_accuracy_line = ax2.plot(df['epoch'], df['train_accuracy'],
                                   color='blue', label='Train Accuracy')
    test_accuracy_line = ax2.plot(df['epoch'], df['test_accuracy'],
                                  color='green', label='Test Accuracy')
    ax2.set_ylabel('Accuracy')

    # legend
    all_lines = train_loss_line + test_loss_line + train_accuracy_line + test_accuracy_line
    all_labels = [line.get_label() for line in all_lines]
    ax1.legend(all_lines, all_labels, loc='upper center')

    if save:
        fname = csv_name.replace('.csv', '.png')        
        plt.savefig(fname)

    return fig
