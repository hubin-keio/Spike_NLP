"""Plot epoch-loss csv file"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_epoch_loss(csv_name):
    df = pd.read_csv(csv_name)
    df.columns = df.columns.str.strip()
    print(df)# Set ggplot style

    plt.style.use('ggplot')
    fig, ax1 = plt.subplots()
    
    # plot loss
    ax1.plot(df['epoch'], df['loss'], color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='red')

    # plot accuracy
    ax2 = ax1.twinx()
    ax2.plot(df['epoch'], df['accuracy'], color='blue')
    ax2.set_ylabel('Accuracy', color='blue')
    
    plt.show()


# if __name__ == '__main__':
#     f_name = os.path.dirname(__file__)
#     f_name = os.path.join(f_name, '../../pn
