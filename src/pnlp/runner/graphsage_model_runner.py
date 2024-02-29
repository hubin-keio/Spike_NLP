#!/usr/bin/env python
""" GraphSAGE model. """

import os
import sys
import tqdm
import torch
import pickle
import datetime
from typing import Union
from torch import nn
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from runner_util import save_model, count_parameters, calc_train_test_history

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.graphsage import GraphSAGE

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
        embedding = self.embeddings[idx]
        edges = [(i, i+1) for i in range(embedding.size(0) - 1)]
        edge_index = torch.tensor(edges, dtype=torch.int64).t().contiguous()
        y = torch.tensor([self.numerical[idx]], dtype=torch.float32).view(-1, 1)
        
        return Data(x=embedding, edge_index=edge_index, y=y)

def run_model(model, train_set, test_set, n_epochs: int, batch_size: int, lr:float, max_batch: Union[int, None], device: str, save_as: str):
    """ Run a model through train and test epochs. """
    
    if not max_batch:
        max_batch = len(train_set)

    model = model.to(device)
    loss_fn = nn.MSELoss(reduction='sum').to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr)

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size, shuffle=True)
    
    metrics_csv = save_as + "_metrics.csv"

    with open(metrics_csv, "w") as fh:
        fh.write(f"epoch,"
                 f"train_loss,test_loss\n")

        for epoch in range(1, n_epochs + 1):
            train_loss = epoch_iteration(model, loss_fn, optimizer, train_loader, epoch, max_batch, device, mode='train')
            test_loss = epoch_iteration(model, loss_fn, optimizer, test_loader, epoch, max_batch, device, mode='test')

            print(f'Epoch {epoch} | Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}\n')
           
            fh.write(f"{epoch},"
                     f"{train_loss},{test_loss}\n")
            fh.flush()
                
            save_model(model, optimizer, epoch, save_as + '.model_save')

    return metrics_csv

def epoch_iteration(model, loss_fn, optimizer, data_loader, num_epochs: int, max_batch: int, device: str, mode: str):
    """ Used in run_model. """

    model.train() if mode=='train' else model.eval()

    data_iter = tqdm.tqdm(enumerate(data_loader),
                          desc=f'Epoch_{mode}: {num_epochs}',
                          total=len(data_loader),
                          bar_format='{l_bar}{r_bar}')

    total_loss = 0

    for batch, batch_data in data_iter:
        if max_batch > 0 and batch >= max_batch:
            break

        batch_data = batch_data.to(device)

        if mode == 'train':
            optimizer.zero_grad()
            pred = model(batch_data.x, batch_data.edge_index, batch_data.batch)
            batch_loss = loss_fn(pred, batch_data.y)
            batch_loss.backward()
            optimizer.step()

        else:
            with torch.no_grad():
                pred = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                batch_loss = loss_fn(pred, batch_data.y)

        total_loss += batch_loss.item()

    return total_loss

if __name__=='__main__':

    # Data/results directories
    result_tag = 'graphsage-esm_dms_binding' # specify rbd_learned or esm, and expression or binding
    data_dir = os.path.join(os.path.dirname(__file__), f'../../../data/pickles')
    results_dir = os.path.join(os.path.dirname(__file__), f'../../../results/run_results/graphsage')
    
    # Create run directory for results
    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(results_dir, f"{result_tag}-{date_hour_minute}")
    os.makedirs(run_dir, exist_ok = True)

    # Load in data
    # embedded_train_pkl = os.path.join(data_dir, 'dms_mutation_expression_meanFs_train_esm_embedded.pkl') # graphsage-esm_dms_expression
    # embedded_test_pkl = os.path.join(data_dir, 'dms_mutation_expression_meanFs_test_esm_embedded.pkl')
    embedded_train_pkl = os.path.join(data_dir, 'dms_mutation_binding_Kds_train_esm_embedded.pkl') # graphsage-esm_dms_binding
    embedded_test_pkl = os.path.join(data_dir, 'dms_mutation_binding_Kds_test_esm_embedded.pkl')
    # embedded_train_pkl = os.path.join(data_dir, 'dms_mutation_expression_meanFs_train_rbd_learned_embedded_320.pkl') # graphsage-rbd_learned_320_dms_expression
    # embedded_test_pkl = os.path.join(data_dir, 'dms_mutation_expression_meanFs_test_rbd_learned_embedded_320.pkl')
    # embedded_train_pkl = os.path.join(data_dir, 'dms_mutation_binding_Kds_train_rbd_learned_embedded_320.pkl') # graphsage-rbd_learned_320_dms_binding
    # embedded_test_pkl = os.path.join(data_dir, 'dms_mutation_binding_Kds_test_rbd_learned_embedded_320.pkl')
    train_dataset = EmbeddedDMSDataset(embedded_train_pkl)
    test_dataset = EmbeddedDMSDataset(embedded_test_pkl)

    # Run setup
    n_epochs = 5000
    batch_size = 32
    max_batch = -1
    lr = 1e-5
    device = torch.device("cuda:0")

    # GraphSAGE input
    input_channels = train_dataset.embeddings[0].size(1) # number of input channels (dimensions of the embeddings)
    out_channels = 1  # For regression output
    model = GraphSAGE(input_channels, out_channels).to(device)

    # Run
    count_parameters(model)
    model_result = os.path.join(run_dir, f"{result_tag}-{date_hour_minute}_train_{len(train_dataset)}_test_{len(test_dataset)}")
    metrics_csv = run_model(model, train_dataset, test_dataset, n_epochs, batch_size, lr, max_batch, device, model_result)
    calc_train_test_history(metrics_csv, len(train_dataset), len(test_dataset), model_result)
