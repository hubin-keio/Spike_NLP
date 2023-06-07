import os
import re
import sys
import torch
from datetime import date
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import seaborn as sns
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split
from prettytable import PrettyTable
from load_model import AlphpaSeqDataset
import pandas as pd


class BLSTM(nn.Module):
    """Bidirectional LSTM
    """
    def __init__(self,
                 batch_size,         # Batch size of the tensor
                 lstm_input_size,    # The number of expected features.
                 lstm_hidden_size,   # The number of features in hidden state h.
                 lstm_num_layers,    # Number of recurrent layers in LSTM.
                 lstm_bidirectional, # Bidrectional LSTM.
                 fcn_hidden_size,    # The number of features in hidden layer of CN.
                 device):            # Device ('cpu' or 'cuda')
        super().__init__()
        self.batch_size = batch_size
        self.device = device


        # LSTM layer
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            bidirectional=lstm_bidirectional,
                            batch_first=True)               # batch first

        # FCN fcn layer
        if lstm_bidirectional:
            self.fcn = nn.Linear(2 * lstm_hidden_size, fcn_hidden_size)
        else:
            self.fcn = nn.Linear(lstm_hidden_size, fcn_hidden_size)

        # FCN output layer
        self.out = nn.Linear(fcn_hidden_size, 1)

    def forward(self, x):
        # Initialize hidden and cell states to zeros.
        num_directions = 2 if self.lstm.bidirectional else 1
        h_0 = torch.zeros(num_directions * self.lstm.num_layers,
                          x.size(0),
                          self.lstm.hidden_size).to(self.device)
        c_0 = torch.zeros(num_directions * self.lstm.num_layers,
                          x.size(0),
                          self.lstm.hidden_size).to(self.device)

        # call lstm with input, hidden state, and internal state
        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        h_n.detach()
        c_n.detach()
        lstm_final_out = lstm_out[:,-1,:]  # last hidden state from every batch. size: N*H_cell
        lstm_final_state = lstm_final_out.to(self.device)
        fcn_out = self.fcn(lstm_final_state)
        prediction = self.out(fcn_out)
        
        return prediction
    
def run_lstm(model: BLSTM,
             train_set: Dataset,
             test_set: Dataset,
             n_epochs: int,
             batch_size: int,
             device: str,
             save_as: str):
    """Run LSTM model

    model: BLSTM,
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

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)


        for batch, (label, feature, target) in enumerate(train_loader):
            optimizer.zero_grad()
            feature, target = feature.to(device), target.to(device)
            pred = model(feature).flatten()
            batch_loss = loss_fn(pred, target)        # MSE loss at batch level
            train_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()


        for batch, (label, feature, target) in enumerate(test_loader):
            feature, target = feature.to(device), target.to(device)

            with torch.no_grad():
                pred = model(feature).flatten()
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


def save_model(model: BLSTM, optimizer: torch.optim.SGD, epoch: int, save_as: str):
    """Save model parameters.

    model: a BLSTM model object
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

def count_parameters(model):
    """Count model parameters and print a summary

    A nice hack from:
    https://stackoverflow.com/a/62508086/1992369
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}\n")
    return total_params

if __name__=='__main__':
    ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    ROOT_DIR = os.path.abspath(ROOT_DIR)
    DATA_DIR = os.path.join(ROOT_DIR, 'data')


    KD_CSV = os.path.join(DATA_DIR, 'clean_avg_alpha_seq.csv')
    
    # Run setup
    DEVICE = 'cpu'
    BATCH_SIZE = 32
    N_EPOCHS = 20

    LSTM_INPUT_SIZE = 28       # lstm_input_size
    LSTM_HIDDEN_SIZE = 50       # lstm_hidden_size
    LSTM_NUM_LAYERS = 1         # lstm_num_layers
    LSTM_BIDIRECTIONAL = True   # lstm_bidrectional
    FCN_HIDDEN_SIZE = 100        # fcn_hidden_size
    
    # pretrained model
    model_path = os.path.abspath('results/2023-05-28_02_26_model_weights.pth')
    data_set = AlphpaSeqDataset(KD_CSV, model_path)
    
    TRAIN_SIZE = int(0.8 * len(data_set))  # 80% goes to training.
    TEST_SIZE = len(data_set) - TRAIN_SIZE
    train_set, test_set = random_split(data_set, (TRAIN_SIZE, TEST_SIZE))

    model = BLSTM(BATCH_SIZE,
                  LSTM_INPUT_SIZE,
                  LSTM_HIDDEN_SIZE,
                  LSTM_NUM_LAYERS,
                  LSTM_BIDIRECTIONAL,
                  FCN_HIDDEN_SIZE,
                  DEVICE)

    count_parameters(model)
    model_result = f'blstm_TAPE_train_{TRAIN_SIZE}_test_{TEST_SIZE}_{date.today()}'
    model_result = os.path.join(ROOT_DIR, f'plots/{model_result}')
    train_losses, test_losses = run_lstm(model, train_set, test_set,
                                         N_EPOCHS, BATCH_SIZE, DEVICE, model_result)
    plot_history(train_losses, TRAIN_SIZE, test_losses, TEST_SIZE, model_result)
    