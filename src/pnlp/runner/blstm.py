#!/usr/bin/env python

import os
import tqdm
import torch
import datetime
from torch import nn
from torch.utils.data import DataLoader, random_split
from typing import Union
from matplotlib import pyplot as plt
import seaborn as sns
from prettytable import PrettyTable
from load_dms import DMS_Dataset, load_model
import pandas as pd

class BLSTM(nn.Module):
    """ Bidirectional LSTM """

    def __init__(self, batch_size, lstm_input_size, lstm_hidden_size,
                 lstm_num_layers, lstm_bidirectional, fcn_hidden_size, device):
        super().__init__()
        self.batch_size = batch_size
        self.device = device

        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers, bidirectional=lstm_bidirectional,
                            batch_first=True)

        if lstm_bidirectional:
            self.fcn = nn.Linear(2 * lstm_hidden_size, fcn_hidden_size)
        else:
            self.fcn = nn.Linear(lstm_hidden_size, fcn_hidden_size)

        self.out = nn.Linear(fcn_hidden_size, 1)

    def forward(self, x):
        num_directions = 2 if self.lstm.bidirectional else 1
        h_0 = torch.zeros(num_directions * self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(self.device)
        c_0 = torch.zeros(num_directions * self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(self.device)

        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        h_n.detach()
        c_n.detach()
        lstm_final_out = lstm_out[:, -1, :]
        lstm_final_state = lstm_final_out.to(self.device)
        fcn_out = self.fcn(lstm_final_state)
        prediction = self.out(fcn_out)

        return prediction

def run_lstm(model, train_set, test_set, n_epochs: int, batch_size: int, max_batch: Union[int, None], device: str):
    """ Run LSTM model """
    
    if not max_batch:
        max_batch = len(train_loader)

    model = model.to(device)
    lr = 1e-5
    loss_fn = nn.MSELoss(reduction='sum').to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr)

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size, shuffle=True)

    train_loss_history = []
    test_loss_history = []

    for epoch in range(1, n_epochs + 1):
        train_loss = epoch_iteration(model, loss_fn, optimizer, train_loader, epoch, max_batch, device, train=True)
        test_loss = epoch_iteration(model, loss_fn, optimizer, test_loader, epoch, max_batch, device, train=False)

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        if epoch < 11 or epoch % 10 == 0:
            print(f'Epoch {epoch}, Train MSE: {train_loss}, Test MSE: {test_loss}')

    return train_loss_history, test_loss_history

def epoch_iteration(model, loss_fn, optimizer, data_loader, num_epochs: int, max_batch: int, device: str, train=True):
    """ Used in run_lstm """
    
    mode = 'train' if train else 'test'
    model.train() if train else model.eval()
    loss = 0

    data_iter = tqdm.tqdm(enumerate(data_loader),
                          desc=f'Epoch_{mode}: {num_epochs}',
                          total=len(data_loader),
                          bar_format='{l_bar}{r_bar}')

    for batch, batch_data in data_iter:
        if max_batch > 0 and batch >= max_batch:
            break 
        
        _, feature, target = batch_data
        feature, target = feature.to(device), target.to(device)

        if mode == 'train':
            optimizer.zero_grad()
            pred = model(feature).flatten()
            batch_loss = loss_fn(pred, target)
            loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                pred = model(feature).flatten()
                batch_loss = loss_fn(pred, target)
                loss += batch_loss.item()

    return loss

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

    DATA_DIR = os.path.join(os.path.join(os.path.dirname(__file__), '../../../data'))
    RESULTS_DIR = os.path.join(os.path.join(os.path.dirname(__file__), '../../../results'))

    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    DIR = os.path.join(RESULTS_DIR, f"blstm-{date_hour_minute}")
    os.makedirs(DIR, exist_ok = True)

    # Data file, reference sequence
    dms_csv = os.path.join(DATA_DIR, 'mutation_binding_Kds.csv')
    refseq = 'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'

    # Load pretrained spike model
    model_pth = os.path.join(RESULTS_DIR, 'ddp-2023-08-14_15-10/ddp-2023-08-14_15-10_best_model_weights.pth')
    model = load_model(model_pth)

    # Dataset, training and test dataset set up
    dataset = DMS_Dataset(dms_csv, refseq, model)
    train_size = int(0.8 * len(dataset)) 
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, (train_size, test_size))

    # Run setup
    device = 'cuda'
    n_epochs = 20
    batch_size = 32
    max_batch = -1

    lstm_input_size = 28       
    lstm_hidden_size = 50      
    lstm_num_layers = 1        
    lstm_bidrectional = True   
    fcn_hidden_size = 100       

    model = BLSTM(batch_size, lstm_input_size, lstm_hidden_size, lstm_num_layers, lstm_bidrectional, fcn_hidden_size,  device)

    count_parameters(model)
    train_losses, test_losses = run_lstm(model, train_set, test_set, n_epochs, batch_size, max_batch, device)
    model_result = os.path.join(DIR, f"blstm-{date_hour_minute}_train_{train_size}_test_{test_size}")
    plot_history(train_losses, train_size, test_losses, test_size, model_result)
    