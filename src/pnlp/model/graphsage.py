#!/usr/bin/env python
""" 
GraphSAGE model, utilizing the DMS binding or expression datasets.

The aim is to utilize the binding affinity or expression value for a 
sequence as the target value for the model to predict. The each sequence
has an embedding representation, which is then used as a node feature
in the graph that gets utilized by the GraphSAGE model. These node features
are connected to each other by edges.
"""

import os
import sys
import tqdm
import torch
import pickle
import datetime
from typing import Union
from collections import defaultdict
from torch import nn
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from model_util import calc_train_test_history

class EmbeddedDMSDataset(Dataset):
    """ Binding or Expression DMS Dataset """
    
    def __init__(self, pickle_file:str):
        """
        Load from pickle file:
        - sequence label (seq_id), 
        - binding or expression numerical target (log10Ka or ML_meanF), and 
        - embeddings
        """
        with open(pickle_file, 'rb') as f:
            dms_list = pickle.load(f)
        
            self.labels = [entry['seq_id'] for entry in dms_list]
            self.numerical = [entry["log10Ka" if "binding" in pickle_file else "ML_meanF"] for entry in dms_list]
            self.embeddings = [entry['embedding'] for entry in dms_list]
 
    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert to pytorch geometric graph
        edges = [(i, i+1) for i in range(self.embeddings[idx].size(0) - 1)]
        edge_index = torch.tensor(edges, dtype=torch.int64).t().contiguous()
        y = torch.tensor([self.numerical[idx]], dtype=torch.float32).view(-1, 1)
        
        return Data(x=self.embeddings[idx], edge_index=edge_index, y=y)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, 16)
        self.conv2 = SAGEConv(16, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return x

def run_graphsage(model, train_set, test_set, n_epochs: int, batch_size: int, lr:float, max_batch: Union[int, None], device: str):
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

    return metrics

def epoch_iteration(model, loss_fn, optimizer, data_loader, num_epochs: int, max_batch: int, device: str, mode: str):
    """ Used in run_graphsage """

    model.train() if mode=='train' else model.eval()
    loss = 0

    data_iter = tqdm.tqdm(enumerate(data_loader),
                          desc=f'Epoch_{mode}: {num_epochs}',
                          total=len(data_loader),
                          bar_format='{l_bar}{r_bar}')

    for batch, data in data_iter:
        if max_batch > 0 and batch >= max_batch:
            break 

        data = data.to(device)

        if mode == 'train':
            optimizer.zero_grad()
            pred = model(data.x, data.edge_index, data.batch)
            batch_loss = loss_fn(pred, data.y)
            loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()

        else:
            with torch.no_grad():
                pred = model(data.x, data.edge_index, data.batch)
                batch_loss = loss_fn(pred, data.y)
                loss += batch_loss.item()

    return loss

if __name__=='__main__':

    dataset_folder = "expression" # specify
    embed_method = "rbd_learned" # specify
    dimension = "320"
    unique_or_duplicate = "unique" # specify

    # Data/results directoriesdata_dir = os.path.join(os.path.dirname(__file__), f'../data/dms/{dataset_folder}/{unique_or_duplicate}')
    results_dir = os.path.join(os.path.dirname(__file__), f'../results/graphsage/dms/{dataset_folder}/{unique_or_duplicate}')
    
    # Create run directory for results
    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(results_dir, f"graphsage_{embed_method}-{date_hour_minute}")
    os.makedirs(run_dir, exist_ok = True)

    # Run setup
    n_epochs = 2500
    batch_size = 32
    max_batch = -1
    lr = 1e-5
    device = torch.device("cuda:0")

    # Load in train and test pickle
    embedded_train_pkl = os.path.join(data_dir, f"{unique_or_duplicate}_mutation_{dataset_folder}_{'Kds' if 'binding' in dataset_folder else 'meanFs'}_train_{embed_method}_embedded{f'_{dimension}'}.pkl")
    train_dataset = EmbeddedDMSDataset(embedded_train_pkl)
    embedded_test_pkl = os.path.join(data_dir, f"{unique_or_duplicate}_mutation_{dataset_folder}_{'Kds' if 'binding' in dataset_folder else 'meanFs'}_test_{embed_method}_embedded{f'_{dimension}'}.pkl")
    test_dataset = EmbeddedDMSDataset(embedded_test_pkl)

    input_channels = train_dataset.embeddings[0].size(1) # number of input channels (dimensions of the embeddings)
    out_channels = 1  # For regression output
    model = GraphSAGE(input_channels, out_channels).to(device)

    model_result = os.path.join(run_dir, f"graphsage-{date_hour_minute}_train_{len(train_dataset)}_test_{len(test_dataset)}")
    metrics = run_graphsage(model, train_dataset, test_dataset, n_epochs, batch_size, lr, max_batch, device)
    calc_train_test_history(metrics, len(train_dataset), len(test_dataset), embed_method, dataset_folder, "graphsage", str(lr), model_result)
