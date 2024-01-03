#!/usr/bin/env python
"""
Model runner for blstm.py, where ESM embeddings get updated.

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

class DMSDataset(Dataset):
    """ Binding or Expression DMS Dataset, not from pickle! """
    
    def __init__(self, csv_file:str):
        """
        Load from csv file into pandas:
        - sequence label ('labels'), 
        - binding or expression numerical target ('log10Ka' or 'ML_meanF'), and 
        - 'sequence'
        """
        try:
            self.full_df = pd.read_csv(csv_file, sep=',', header=0)
            self.target = 'log10Ka' if 'binding' in csv_file else 'ML_meanF'
        except (FileNotFoundError, pd.errors.ParserError, Exception) as e:
            print(f"Error reading in .csv file: {csv_file}\n{e}", file=sys.stderr)
            sys.exit(1)

    def __len__(self) -> int:
        return len(self.full_df)

    def __getitem__(self, idx):
        # label, seq, target
        return self.full_df['labels'][idx], self.full_df['sequence'][idx], self.full_df[self.target][idx]

class ESM_BLSTM(nn.Module):
    def __init__(self, esm, blstm):
        super().__init__()
        self.esm = esm
        self.blstm = blstm

    def forward(self, tokenized_seqs):
        with torch.set_grad_enabled(self.training):  # Enable gradients, managed by model.eval() or model.train() in epoch_iteration
            esm_output = self.esm(**tokenized_seqs).last_hidden_state
            reshaped_output = esm_output.squeeze(0)  
            output = self.blstm(reshaped_output)
        return output

def run_model(model, tokenizer, train_set, test_set, n_epochs: int, batch_size: int, lr:float, max_batch: Union[int, None], device: str, save_as: str):
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
            train_loss = epoch_iteration(model, tokenizer, loss_fn, optimizer, train_loader, epoch, max_batch, device, mode='train')
            test_loss = epoch_iteration(model, tokenizer, loss_fn, optimizer, test_loader, epoch, max_batch, device, mode='test')

            print(f'Epoch {epoch} | Train BLSTM Loss: {train_loss:.4f}, Test BLSTM Loss: {test_loss:.4f}\n')
           
            fh.write(f"{epoch},"
                     f"{train_loss},{test_loss}\n")
            fh.flush()
                
            save_model(model, optimizer, epoch, save_as + '.model_save')

    return metrics_csv

def epoch_iteration(model, tokenizer, loss_fn, optimizer, data_loader, num_epochs: int, max_batch: int, device: str, mode: str):
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

        labels, seqs, targets = batch_data
        targets = targets.to(device).float()
        tokenized_seqs = tokenizer(seqs,return_tensors="pt").to(device)
   
        if mode == 'train':
            optimizer.zero_grad()
            pred = model(tokenized_seqs).flatten()
            batch_loss = loss_fn(pred, targets)
            batch_loss.backward()
            optimizer.step()

        else:
            with torch.no_grad():
                pred = model(tokenized_seqs).flatten()
                batch_loss = loss_fn(pred, targets)

        total_loss += batch_loss.item()

    return total_loss
 
if __name__=='__main__':

    # Data/results directories
    result_tag = 'blstm_updating-esm_dms_binding' # specify expression or binding
    data_dir = os.path.join(os.path.dirname(__file__), f'../../../data')
    results_dir = os.path.join(os.path.dirname(__file__), f'../../../results/run_results/blstm')
    
    # Create run directory for results
    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(results_dir, f"{result_tag}-{date_hour_minute}")
    os.makedirs(run_dir, exist_ok = True)

    # Load in data (from csv)
    dms_train_csv = os.path.join(data_dir, 'dms_mutation_expression_meanFs_train.csv') # blstm_updating-esm_dms_expression
    dms_test_csv = os.path.join(data_dir, 'dms_mutation_expression_meanFs_test.csv') 
    dms_train_csv = os.path.join(data_dir, 'dms_mutation_binding_Kds_train.csv') # blstm_updating-esm_dms_binding
    dms_test_csv = os.path.join(data_dir, 'dms_mutation_binding_Kds_test.csv') 

    train_dataset = DMSDataset(dms_train_csv)
    test_dataset = DMSDataset(dms_test_csv)

    # Run setup
    n_epochs = 5000
    batch_size = 32
    max_batch = -1
    lr = 1e-5
    device = torch.device("cuda:0")

    # ESM input
    esm = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    # BLSTM input
    lstm_input_size = 320
    lstm_hidden_size = 320
    lstm_num_layers = 1        
    lstm_bidrectional = True   
    fcn_hidden_size = 320
    blstm = BLSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers, lstm_bidrectional, fcn_hidden_size)

    model = ESM_BLSTM(esm, blstm)
 
    # Run
    count_parameters(model)
    model_result = os.path.join(run_dir, f"{result_tag}-{date_hour_minute}_train_{len(train_dataset)}_test_{len(test_dataset)}")
    metrics_csv = run_model(model, tokenizer, train_dataset, test_dataset, n_epochs, batch_size, lr, max_batch, device, model_result)
    calc_train_test_history(metrics_csv, len(train_dataset), len(test_dataset), model_result)
