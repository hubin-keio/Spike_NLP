#!/usr/bin/env python
"""
Model runner for blstm.py, where ESM and RBD Learned embeddings are fixed.

TODO: 
- Add blstm and bert_blstm to pnlp module? To avoid sys pathing hack
"""

import os
import sys
import tqdm
import torch
import pickle
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union
from collections import defaultdict
from torch import nn
from torch.utils.data import Dataset, DataLoader
from runner_util import save_model, count_parameters, calc_train_test_history
from transformers import AutoTokenizer, EsmModel 
from pnlp.embedding.tokenizer import ProteinTokenizer, token_to_index
from pnlp.model.language import BERT

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.blstm import BLSTM

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

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        # label, feature, target
        return self.labels[idx], self.embeddings[idx].to(device), self.numerical[idx]

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
        
        label, feature, target = batch_data
        feature, target = feature.to(device), target.to(device) 
        target = target.float()

        if mode == 'train':
            optimizer.zero_grad()
            pred = model(feature).flatten()
            batch_loss = loss_fn(pred, target)
            batch_loss.backward()
            optimizer.step()

        else:
            with torch.no_grad():
                pred = model(feature).flatten()
                batch_loss = loss_fn(pred, target)
                
        total_loss += batch_loss.item()

    return total_loss
 
if __name__=='__main__':

    # Data/results directories
    result_tag = 'blstm_fixed-rbd_learned_320_dms_binding' # specify expression or binding, esm or rbd_learned
    data_dir = os.path.join(os.path.dirname(__file__), f'../../../data/pickles')
    results_dir = os.path.join(os.path.dirname(__file__), f'../../../results/run_results/blstm')
    
    # Create run directory for results
    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(results_dir, f"{result_tag}-{date_hour_minute}")
    os.makedirs(run_dir, exist_ok = True)

    # Load in data (from pickle)
    # embedded_train_pkl = os.path.join(data_dir, 'dms_mutation_expression_meanFs_train_esm_embedded.pkl') # blstm_fixed-esm_dms_expression
    # embedded_test_pkl = os.path.join(data_dir, 'dms_mutation_expression_meanFs_test_esm_embedded.pkl')
    # embedded_train_pkl = os.path.join(data_dir, 'dms_mutation_binding_Kds_train_esm_embedded.pkl') # blstm_fixed-esm_dms_binding
    # embedded_test_pkl = os.path.join(data_dir, 'dms_mutation_binding_Kds_test_esm_embedded.pkl')
    # embedded_train_pkl = os.path.join(data_dir, 'dms_mutation_expression_meanFs_train_rbd_learned_embedded_320.pkl') # blstm_fixed-rbd_learned_320_dms_expression
    # embedded_test_pkl = os.path.join(data_dir, 'dms_mutation_expression_meanFs_test_rbd_learned_embedded_320.pkl')
    embedded_train_pkl = os.path.join(data_dir, 'dms_mutation_binding_Kds_train_rbd_learned_embedded_320.pkl') # blstm_fixed-rbd_learned_320_dms_binding
    embedded_test_pkl = os.path.join(data_dir, 'dms_mutation_binding_Kds_test_rbd_learned_embedded_320.pkl')

    device = torch.device("cuda:3")
    train_dataset = EmbeddedDMSDataset(embedded_train_pkl, device)
    test_dataset = EmbeddedDMSDataset(embedded_test_pkl, device)

    # Run setup
    n_epochs = 1
    batch_size = 32
    max_batch = -1
    lr = 1e-5

    # BLSTM input
    lstm_input_size = 320
    lstm_hidden_size = 320
    lstm_num_layers = 1        
    lstm_bidrectional = True   
    fcn_hidden_size = 320
    model = BLSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers, lstm_bidrectional, fcn_hidden_size)
 
    # Run
    count_parameters(model)
    model_result = os.path.join(run_dir, f"{result_tag}-{date_hour_minute}_train_{len(train_dataset)}_test_{len(test_dataset)}")
    metrics_csv = run_model(model, train_dataset, test_dataset, n_epochs, batch_size, lr, max_batch, device, model_result)
    calc_train_test_history(metrics_csv, len(train_dataset), len(test_dataset), model_result)
