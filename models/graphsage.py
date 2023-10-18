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
from torch_geometric.nn import SAGEConv
from torch.utils.data import DataLoader, random_split
from typing import Union
from prettytable import PrettyTable

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.load_dms import PKL_Loader

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, 16)
        self.conv2 = SAGEConv(16, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

def run_graphsage(model, train_set, test_set, n_epochs: int, batch_size: int, lr:float, max_batch: Union[int, None], device: str, save_as: str):
    """ Run GraphSAGE model """
    
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
    """ Used in run_graphsage """

    model.train() if mode=='train' else model.eval()
    loss = 0

    data_iter = tqdm.tqdm(enumerate(data_loader),
                          desc=f'Epoch_{mode}: {num_epochs}',
                          total=len(data_loader),
                          bar_format='{l_bar}{r_bar}')

    for batch, batch_data in data_iter:
        if max_batch > 0 and batch >= max_batch:
            break 
        
        label, feature, edge_index, target = batch_data  # Assume edge_index is available
        feature, edge_index, target = feature.to(device), edge_index.to(device), target.to(device) 

        if mode == 'train':
            optimizer.zero_grad()
            pred = model(feature, edge_index).flatten()
            batch_loss = loss_fn(pred, target)
            loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()

        else:
            with torch.no_grad():
                pred = model(feature, edge_index).flatten()
                batch_loss = loss_fn(pred, target)
                loss += batch_loss.item()

    return loss

def save_model(model: GraphSAGE, optimizer: torch.optim.SGD, epoch: int, save_as: str):  
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
        - the dimension size needed for running the blstm model.
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
        raise ValueError("Invalid embedding type. Choose from 'rbd_learned', 'rbd_bert', 'esm', or 'one_hot'.")
    
    train_file, test_file, input_shape = data_files
    embedded_train_pkl = os.path.join(data_dir, train_file)
    embedded_test_pkl = os.path.join(data_dir, test_file)
    
    train_pkl_loader = PKL_Loader(embedded_train_pkl, device) 
    test_pkl_loader = PKL_Loader(embedded_test_pkl, device)    
    
    return train_pkl_loader, test_pkl_loader, len(train_pkl_loader), len(test_pkl_loader), input_shape[1]

if __name__=='__main__':

    results_dir = os.path.join(os.path.dirname(__file__), '../results/blstm')
    embedding_method = "rbd_learned"

    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(results_dir, f"blstm_{embedding_method}-{date_hour_minute}")
    os.makedirs(run_dir, exist_ok = True)

    # Run setup
    n_epochs = 10
    batch_size = 32
    max_batch = 10
    lr = 1e-5
    device = "cuda:0"

    train_pkl_loader, test_pkl_loader, train_size, test_size, input_channels = load_embedding_pkl(embedding_method, device)
   
    out_channels = 1  # For regression output
    model = GraphSAGE(input_channels, out_channels).to(device)

    run_graphsage(model, train_pkl_loader, test_pkl_loader, n_epochs, batch_size, lr, max_batch, device)
