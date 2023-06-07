import os
import torch
from datetime import date
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split
from load_model import AlphpaSeqDataset

class RNN(nn.Module):
    """RNN implementation"""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)  # additional linear layer
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = F.relu(self.fc1(out[:, -1, :]))  # use ReLU activation function after first FC layer
        out = self.fc2(out)
        return out
    
def run_rnn(model: RNN,
             train_set: Dataset,
             test_set: Dataset,
             n_epochs: int,
             batch_size: int,
             device: str,
             save_as: str):
    
    """Run RNN model

    model: RNN,
    train_set: training set dataset
    test_set: test det dataset
    n_epochs: number of epochs
    batch_size: batch size
    device: 'gpu' or 'cpu'
    save_as: path and file name to save the model results
    """

    L_RATE = 1e-5               # learning rate
    model = model.to(device)



    loss_fn = nn.MSELoss(reduction='sum').to(device)  # MSE loss with sum
    optimizer = torch.optim.SGD(model.parameters(), L_RATE)  # SGD optimizer

    train_loss_history = []
    test_loss_history = []
    for epoch in range(1, n_epochs + 1):
        train_loss = 0
        test_loss = 0

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)


        for batch, (label, feature, target) in enumerate(train_loader):
            optimizer.zero_grad()
            feature, target = feature.to(device), target.to(device)
            pred = model.forward(feature).flatten()
            batch_loss = loss_fn(pred, target)        # MSE loss at batch level
            train_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()

        for batch, (label, feature, target) in enumerate(test_loader):
            feature, target = feature.to(device), target.to(device)

            with torch.no_grad():
                pred = model.forward(feature).flatten()
                batch_loss = loss_fn(pred, target)
                test_loss += batch_loss.item()
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        if epoch < 11:
            print(f'Epoch {epoch}, Train MSE: {train_loss}, Test MSE: {test_loss}')
        elif epoch%10 == 0:
            print(f'Epoch {epoch}, Train MSE: {train_loss}, Test MSE: {test_loss}')

        save_model(model, optimizer, epoch, save_as + '.model_save')

    return train_loss_history, test_loss_history

def save_model(model: RNN, optimizer: torch.optim.SGD, epoch: int, save_as: str):
    """Save model parameters.

    model: a RNN model object
    optimizer: model optimizer
    epoch: number of epochs in the end of the model running
    save_as: file name for saveing the model.
    """
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
               save_as)

def plot_history(train_losses: list, n_train: int, test_losses: list,
                 n_test: int, save_as: str):
    """Plot training and testing history per epoch

    train_losses: a list of per epoch error from the training set
    n_train: number of items in the training set
    test_losses: a list of per epoch error from the test set
    n_test: number of items in the test set
    """
    history_df = pd.DataFrame(list(zip(train_losses, test_losses)),
                              columns = ['training','testing'])

    history_df['training'] = history_df['training']/n_train  # average error per item
    history_df['testing'] = history_df['testing']/n_test

    print(history_df)

    sns.set_theme()
    sns.set_context('talk')

    plt.ion()
    fig = plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(data=history_df, x=history_df.index, y='training', label='training')
    sns.scatterplot(data=history_df, x=history_df.index, y='testing', label='testing')
    ax.set(xlabel='Epochs', ylabel='Average MSE per sample')

    fig.savefig(save_as + '.png')
    history_df.to_csv(save_as + '.csv')
    
    
    
if __name__=='__main__':
    ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    ROOT_DIR = os.path.abspath(ROOT_DIR)
    DATA_DIR = os.path.join(ROOT_DIR, 'data')


    KD_CSV = os.path.join(DATA_DIR, 'clean_avg_alpha_seq.csv')
    
    # Run setup
    DEVICE = 'cpu'
    BATCH_SIZE = 32
    N_EPOCHS = 20

    RNN_INPUT_SIZE = 28    # RNN_input_size
    RNN_HIDDEN_SIZE = 50      # RNN_hidden_size
    RNN_OUTPUT_SIZE = 1       # RNN_output_size
    
    # pretrained model
    model_path = os.path.abspath('results/2023-05-28_02_26_model_weights.pth')
    data_set = AlphpaSeqDataset(KD_CSV, model_path)
    
    TRAIN_SIZE = int(0.8 * len(data_set))  # 80% goes to training.
    TEST_SIZE = len(data_set) - TRAIN_SIZE
    train_set, test_set = random_split(data_set, (TRAIN_SIZE, TEST_SIZE))

    model = RNN(RNN_INPUT_SIZE,
                RNN_HIDDEN_SIZE,
                RNN_OUTPUT_SIZE)

    model_result = f'RNN_{TRAIN_SIZE}_test_{TEST_SIZE}_{date.today()}'
    model_result = os.path.join(ROOT_DIR, f'plots/{model_result}')
    train_losses, test_losses = run_rnn(model, train_set, test_set,
                                         N_EPOCHS, BATCH_SIZE, DEVICE, model_result)
    plot_history(train_losses, TRAIN_SIZE, test_losses, TEST_SIZE, model_result)
