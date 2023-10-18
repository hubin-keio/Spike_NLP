#!/usr/bin/env python

import os
import sys
import tqdm
import torch
import datetime
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from torch import nn
from torch.utils.data import DataLoader
from typing import Union
from prettytable import PrettyTable
from pnlp.embedding.tokenizer import ProteinTokenizer, token_to_index, index_to_token

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.load_dms import PKL_Loader
from src.pnlp.plots.plot_mse_rmse_history import calc_train_test_history

class FCN(nn.Module):
    """ Fully Connected Network """

    def __init__(self,
                 fcn_input_size,     # The number of input features
                 fcn_hidden_size,    # The number of features in hidden layer of FCN.
                 device):            # Device ('cpu' or 'cuda')
        super().__init__()
        self.device = device

        # FCN layers
        self.fcn = nn.Sequential(nn.Linear(fcn_input_size, fcn_hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(fcn_hidden_size, 1))  # Adjust this line based on the required output size

    def forward(self, x):
        fcn_out = self.fcn(x)
        fcn_final_out = fcn_out[:, -1, :]
        prediction = fcn_final_out.to(self.device)

        return prediction
 
def run_fcn(model, train_set, test_set, n_epochs: int, batch_size: int, lr:float, max_batch: Union[int, None], device: str, save_as: str):    
    """ Run FCN model """
    
    if not max_batch:
        max_batch = len(train_set)

    model = model.to(device)
    loss_fn = nn.MSELoss(reduction='sum').to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr)

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size, shuffle=True)
    
    metrics = defaultdict(list)

    for epoch in range(1, n_epochs + 1):
        train_loss = epoch_iteration(model, loss_fn, optimizer, train_loader, epoch, max_batch, device, mode='train')
        test_loss = epoch_iteration(model, loss_fn, optimizer, test_loader, epoch, max_batch, device, mode='test')

        keys = ['train_loss','test_loss'] # to add more metrics, add more keys
        for key in keys:
            metrics[key].append(locals()[key])

        print(f'\n'
              f'Epoch {epoch} | Train MSE: {train_loss:.4f}\n'
              f'{" "*(len(str(epoch))+8)} Test MSE: {test_loss:.4f}')

        save_model(model, optimizer, epoch, save_as + '.model_save')

    return metrics

def epoch_iteration(model, loss_fn, optimizer, data_loader, num_epochs: int, max_batch: int, device: str, mode: str):
    """ Used in run_lstm """
    
    model.train() if mode=='train' else model.eval()
    loss = 0

    data_iter = tqdm.tqdm(enumerate(data_loader),
                          desc=f'Epoch_{mode}: {num_epochs}',
                          total=len(data_loader),
                          bar_format='{l_bar}{r_bar}')

    for batch, batch_data in data_iter:
        if max_batch > 0 and batch >= max_batch:
            break 
        
        label, feature, target = batch_data
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

def save_model(model: FCN, optimizer: torch.optim.SGD, epoch: int, save_as: str):
    """
    Save model parameters.

    model: a FCN model object
    optimizer: model optimizer
    epoch: number of epochs in the end of the model running
    save_as: file name for saveing the model.
    """
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                save_as)

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

def load_embedding_pkl(embedding_method: str, device: str) -> tuple:
    """
    Load in embedding pkl based on embedding type, and set up the pkl loader.
    Based on the embedding type, returns the 
        - pkl loaders, 
        - length of the loaders, and
        - the dimension size needed for running the fcn model.
    """
    data_dir = os.path.join(os.path.dirname(__file__), '../data/dms')
    
    file_names = {"rbd_learned": ("mutation_binding_Kds_train_rbd_learned_embedded.pkl", 
                                  "mutation_binding_Kds_test_rbd_learned_embedded.pkl", 
                                  [201, 320]),
                  "rbd_bert": ("mutation_binding_Kds_train_rbd_bert_embedded.pkl", 
                               "mutation_binding_Kds_test_rbd_bert_embedded.pkl", 
                               [201, 320]),
                  "esm": ("mutation_binding_Kds_train_esm_embedded.pkl", 
                          "mutation_binding_Kds_test_esm_embedded.pkl", 
                          [203, 320])}
    
    data_files = file_names.get(embedding_method.lower())
    if not data_files:
        raise ValueError("Invalid embedding type. Choose from 'rbd_learned', 'rbd_bert', or 'esm'.")
    
    train_file, test_file, input_shape = data_files
    embedded_train_pkl = os.path.join(data_dir, train_file)
    embedded_test_pkl = os.path.join(data_dir, test_file)
    
    train_pkl_loader = PKL_Loader(embedded_train_pkl, device) 
    test_pkl_loader = PKL_Loader(embedded_test_pkl, device)    
    
    return train_pkl_loader, test_pkl_loader, len(train_pkl_loader), len(test_pkl_loader), input_shape[1]

if __name__=='__main__':

    results_dir = os.path.join(os.path.dirname(__file__), '../results/fcn')
    embedding_method = "esm"

    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(results_dir, f"fcn_{embedding_method}-{date_hour_minute}")
    os.makedirs(run_dir, exist_ok = True)

    # Run setup
    n_epochs = 5000
    batch_size = 32
    max_batch = -1
    lr = 1e-5
    device = torch.device("cuda:0")

    train_pkl_loader, test_pkl_loader, train_size, test_size, fcn_size = load_embedding_pkl(embedding_method, device)

    fcn_input_size = fcn_size      
    fcn_hidden_size = fcn_size 
    model = FCN(fcn_input_size, fcn_hidden_size, device)

    count_parameters(model)
    model_result = os.path.join(run_dir, f"fcn{date_hour_minute}_train_{train_size}_test_{test_size}")
    metrics  = run_fcn(model, train_pkl_loader, test_pkl_loader, n_epochs, batch_size, lr, max_batch, device, model_result)   
    calc_train_test_history(metrics, train_size, test_size, embedding_method, model_result)   
    