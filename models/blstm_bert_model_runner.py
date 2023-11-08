#!/usr/bin/env python
"""
Model runner for bert_blstm.py
TODO: Have not made any changes to this, not updated for bert_blstm.py yet
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
from torch.utils.data import Dataset, DataLoader
from model_util import save_model, count_parameters, calc_train_test_history
from bert_blstm import BERT_BLSTM

class EmbeddedDMSDataset(Dataset):
    """ Binding or Expression DMS Dataset """
    
    def __init__(self, pickle_file:str, device:str):
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
 
        self.device = device

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        # label, feature, target
        return self.labels[idx], self.embeddings[idx].to(self.device), self.numerical[idx]

def run_model(model, train_set, test_set, n_epochs: int, batch_size: int, lr:float, max_batch: Union[int, None], device: str, save_as: str):
    """ Run a model through train and test epochs"""
    
    if not max_batch:
        max_batch = len(train_set)

    model = model.to(device)
    blstm_loss_fn = nn.MSELoss(reduction='sum').to(device)
    bert_loss = torch.nn.CrossEntropyLoss(reduction='sum').to(device)
    
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
    """ Used in run_model """
    
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
        target = target.float()

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
 
if __name__=='__main__':

    dataset_folder = "expression" # specify
    embed_method = "rbd_learned" # specify

    # Data/results directories
    data_dir = os.path.join(os.path.dirname(__file__), f'../data/dms/{dataset_folder}')
    results_dir = os.path.join(os.path.dirname(__file__), f'../results/bert_blstm/dms/{dataset_folder}')
    
    # Create run directory for results
    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(results_dir, f"bert_blstm_{embed_method}-{date_hour_minute}")
    os.makedirs(run_dir, exist_ok = True)

    # Run setup
    n_epochs = 1
    batch_size = 32
    max_batch = -1
    lr = 1e-5
    device = torch.device("cuda:0")

    # Load in train and test pickle
    embedded_train_pkl = os.path.join(data_dir, f"mutation_{dataset_folder}_{'Kds' if 'binding' in dataset_folder else 'meanFs'}_train_{embed_method}_embedded_320.pkl")
    train_dataset = EmbeddedDMSDataset(embedded_train_pkl, device)
    embedded_test_pkl = os.path.join(data_dir, f"mutation_{dataset_folder}_{'Kds' if 'binding' in dataset_folder else 'meanFs'}_test_{embed_method}_embedded_320.pkl")
    test_dataset = EmbeddedDMSDataset(embedded_test_pkl, device)

    lstm_input_size = train_dataset.embeddings[0].size(1)      
    lstm_hidden_size = lstm_input_size   
    lstm_num_layers = 1        
    lstm_bidrectional = True   
    fcn_hidden_size = lstm_input_size
    model = BERT_BLSTM(batch_size, lstm_input_size, lstm_hidden_size, lstm_num_layers, lstm_bidrectional, fcn_hidden_size, device)

    count_parameters(model)
    model_result = os.path.join(run_dir, f"bert_blstm-{date_hour_minute}_train_{len(train_dataset)}_test_{len(test_dataset)}")
    metrics  = run_model(model, train_dataset, test_dataset, n_epochs, batch_size, lr, max_batch, device, model_result)
    calc_train_test_history(metrics, len(train_dataset), len(test_dataset), embed_method, dataset_folder, "bert_blstm", str(lr), model_result)
