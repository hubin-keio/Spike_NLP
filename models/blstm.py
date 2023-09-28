#!/usr/bin/env python

import os
import tqdm
import torch
import datetime
import seaborn as sns
import pandas as pd
from collections import defaultdict
from torch import nn
from torch.utils.data import DataLoader, random_split
from typing import Union
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from torcheval.metrics.functional import r2_score
from load_dms import DMS_Dataset, load_model
from pnlp.embedding.tokenizer import ProteinTokenizer, token_to_index, index_to_token

class BLSTM(nn.Module):
    """ Bidirectional LSTM """

    def __init__(self, batch_size, lstm_input_size, lstm_hidden_size,
                 lstm_num_layers, lstm_bidirectional, fcn_hidden_size, device, fcn_hidden_size2=50):
        super().__init__()
        self.batch_size = batch_size
        self.device = device

        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers, bidirectional=lstm_bidirectional,
                            batch_first=True)

        if lstm_bidirectional:
            lstm_output_size = 2 * lstm_hidden_size
        else:
            lstm_output_size = lstm_hidden_size

        self.fcn = nn.Sequential(nn.Linear(lstm_output_size, fcn_hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(fcn_hidden_size, fcn_hidden_size2),
                                           nn.ReLU())
        self.out = nn.Linear(fcn_hidden_size2, 1)

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

def run_lstm(model, train_set, test_set, n_epochs: int, batch_size: int, max_batch: Union[int, None], device: str, save_as: str):
    """ Run LSTM model """
    
    if not max_batch:
        max_batch = len(train_set)

    model = model.to(device)
    lr = 1e-5
    loss_fn = nn.MSELoss(reduction='sum').to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr)

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size, shuffle=True)
    
    metrics = defaultdict(list)

    for epoch in range(1, n_epochs + 1):
        train_loss, train_r2 = epoch_iteration(model, loss_fn, optimizer, train_loader, epoch, max_batch, device, mode='train')
        test_loss, test_r2 = epoch_iteration(model, loss_fn, optimizer, test_loader, epoch, max_batch, device, mode='test')

        keys = ['train_loss','test_loss','train_r2','test_r2']
        for key in keys:
            metrics[key].append(locals()[key])

        print(f'\n'
              f'Epoch {epoch} | Train MSE: {train_loss:.4f}, Train R2: {train_r2:.4f}\n'
              f'{" "*(len(str(epoch))+8)} Test MSE: {test_loss:.4f}, Test R2: {test_r2:.4f}')

        save_model(model, optimizer, epoch, save_as + '.model_save')

    return metrics

def epoch_iteration(model, loss_fn, optimizer, data_loader, num_epochs: int, max_batch: int, device: str, mode: str):
    """ Used in run_lstm """
    
    model.train() if mode=='train' else model.eval()
    loss = 0
    r2 = 0

    data_iter = tqdm.tqdm(enumerate(data_loader),
                          desc=f'Epoch_{mode}: {num_epochs}',
                          total=len(data_loader),
                          bar_format='{l_bar}{r_bar}')

    for batch, batch_data in data_iter:
        if max_batch > 0 and batch >= max_batch:
            break 
        
        _, target, _, feature = batch_data
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

        # R2 value 
        r2 = r2_score(pred, target).cpu()

    return loss, r2

def save_model(model: BLSTM, optimizer: torch.optim.SGD, epoch: int, save_as: str):
    """
    Save model parameters.

    model: a BLSTM model object
    optimizer: model optimizer
    epoch: number of epochs in the end of the model running
    save_as: file name for saveing the model.
    """
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                save_as)

def plot_history(metrics: dict, n_train: int, n_test: int, save_as: str):
    """ Plot training and testing history per epoch. """

    history_df = pd.DataFrame(metrics)
    history_df['train_loss'] = history_df['train_loss']/n_train  # average error per item
    history_df['test_loss'] = history_df['test_loss']/n_test
    
    sns.set_theme()
    sns.set_context('talk')

    plt.ion()
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))

    sns.scatterplot(data=history_df, x=history_df.index, y='train_loss', label='training', ax=axes[0])
    sns.scatterplot(data=history_df, x=history_df.index, y='test_loss', label='testing', ax=axes[0])
    axes[0].set(xlabel='Epochs', ylabel='Average MSE per sample')

    sns.scatterplot(data=history_df, x=history_df.index, y='train_r2', label='training', ax=axes[1])
    sns.scatterplot(data=history_df, x=history_df.index, y='test_r2', label='testing', ax=axes[1])
    axes[1].set(xlabel='Epochs', ylabel='R-squared')

    plt.tight_layout()
    fig.savefig(save_as + '.png')
    history_df.to_csv(save_as + '.csv')

def count_parameters(model):
    """
    Count model parameters and print a summary

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
    RESULTS_DIR = os.path.join(os.path.join(os.path.dirname(__file__), '../../../results/blstm'))

    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    DIR = os.path.join(RESULTS_DIR, f"blstm-{date_hour_minute}")
    os.makedirs(DIR, exist_ok = True)

    # Data file, reference sequence
    dms_csv = os.path.join(DATA_DIR, 'mutation_binding_Kds.csv')
    refseq = 'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'

    # Load pretrained spike model
    model_pth = os.path.join(RESULTS_DIR, 'ddp-2023-08-16_08-41/ddp-2023-08-16_08-41_best_model_weights.pth')
    model = load_model(model_pth)

    # Dataset, training and test dataset set up
    dataset = DMS_Dataset(dms_csv, refseq, model)
    train_size = int(0.8 * len(dataset)) 
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, (train_size, test_size))

    # Run setup
    device = "cuda"
    n_epochs = 20 #20
    batch_size = 32
    max_batch = 100

    lstm_input_size = 768       
    lstm_hidden_size = 768     
    lstm_num_layers = 1        
    lstm_bidrectional = True   
    fcn_hidden_size = 768       

    model = BLSTM(batch_size, lstm_input_size, lstm_hidden_size, lstm_num_layers, lstm_bidrectional, fcn_hidden_size,  device)

    count_parameters(model)
    model_result = os.path.join(DIR, f"blstm-{date_hour_minute}_train_{train_size}_test_{test_size}")
    metrics  = run_lstm(model, train_set, test_set, n_epochs, batch_size, max_batch, device, model_result)
    plot_history(metrics, train_size, test_size, model_result)
    