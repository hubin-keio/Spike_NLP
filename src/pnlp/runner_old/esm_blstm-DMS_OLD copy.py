#!/usr/bin/env python
"""
Model runner for ESM-BLSTM model (single target).
"""

import os
import re
import sys
import math
import tqdm
import torch
import time
import datetime
import numpy as np
import pandas as pd
from typing import Union
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmModel, EsmConfig
import seaborn as sns
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from dms_runner_util import (
    DMSDataset,
    ScheduledOptim,
    count_parameters,
    save_model,
    load_model,
    load_model_checkpoint,
    plot_log_file,
)


# BLSTM
class BLSTM(nn.Module):
    """ Bidirectional LSTM. Output is embedding layer, not prediction value."""

    def __init__(self,
                 lstm_input_size,    # The number of expected features.
                 lstm_hidden_size,   # The number of features in hidden state h.
                 lstm_num_layers,    # Number of recurrent layers in LSTM.
                 lstm_bidirectional, # Bidrectional LSTM.
                 fcn_hidden_size):   # The number of features in hidden layer of CN.
        super().__init__()

        # LSTM layer
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            bidirectional=lstm_bidirectional,
                            batch_first=True)           

        # FCN
        if lstm_bidirectional:
            self.fcn = nn.Sequential(nn.Linear(2 * lstm_hidden_size, fcn_hidden_size),
                                     nn.ReLU())
        else:
            self.fcn = nn.Sequential(nn.Linear(lstm_hidden_size, fcn_hidden_size),
                                     nn.ReLU())

        # FCN output layer
        self.out = nn.Linear(fcn_hidden_size, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add a sequence length dimension of 1, now [batch_size, sequence_length, features]

        num_directions = 2 if self.lstm.bidirectional else 1
        h_0 = torch.zeros(num_directions * self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
        c_0 = torch.zeros(num_directions * self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)

        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        lstm_final_out = lstm_out[:, -1, :]
        fcn_out = self.fcn(lstm_final_out)
        prediction = self.out(fcn_out)  # [batch_size, 1]

        return prediction

# ESM-BLSTM
class ESM_BLSTM(nn.Module):
    def __init__(self, esm, blstm):
        super().__init__()
        self.esm = esm
        self.blstm = blstm

    def forward(self, tokenized_seqs):
        with torch.set_grad_enabled(self.training):  # Enable gradients, managed by model.eval() or model.train() in epoch_iteration
            esm_last_hidden_state = self.esm(**tokenized_seqs).last_hidden_state # shape: [batch_size, sequence_length, embedding_dim]
            esm_aa_embedding = esm_last_hidden_state[:, 1:-1, :] # Amino Acid-level representations, [batch_size, sequence_length-2, embedding_dim], excludes 1st and last tokens
            output = self.blstm(esm_aa_embedding).squeeze(1) # [batch_size]
        return output


# MODEL RUNNING
def run_model(model, tokenizer, train_data_loader, test_data_loader, n_epochs: int, lr:float, max_batch: Union[int, None], device: str, run_dir: str, save_as: str, saved_model_pth:str=None, from_checkpoint:bool=False):
    """ Run a model through train and test epochs. """

    model = model.to(device)
    loss_fn = nn.MSELoss(reduction='sum').to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr)
    scheduler = ScheduledOptim(
        optimizer, 
        d_model=model.blstm.lstm.hidden_size, 
        n_warmup_steps=(len(train_data_loader.dataset) / train_data_loader.batch_size) * 0.1
    ) 

    metrics_csv = os.path.join(run_dir, f"{save_as}_metrics.csv")
    metrics_img = os.path.join(run_dir, f"{save_as}_metrics.pdf")

    starting_epoch = 1
    best_rmse = float('inf')

    # Load saved model
    if saved_model_pth is not None and os.path.exists(saved_model_pth):
        if from_checkpoint:
            model_state, optimizer_state, scheduler_state, starting_epoch, best_rmse = load_model(saved_model_pth, device)

            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
            starting_epoch += 1
            
            if starting_epoch > n_epochs:
                raise ValueError(f"Starting epoch ({starting_epoch}) is greater than the total number of epochs to run ({n_epochs}). Adjust the number of epochs, 'n_epochs'.")
        
        else:
            model_state, _, _, _ = load_model(saved_model_pth, device)
            model.load_state_dict(model_state)

    with open(metrics_csv, "a") as fa:
        if from_checkpoint: load_model_checkpoint(saved_model_pth, metrics_csv, starting_epoch)
        else: fa.write(f"Epoch,Train RMSE,Test RMSE\n")

    # Running
    start_time = time.time()

    for epoch in range(starting_epoch, n_epochs + 1):
        train_mse, train_rmse = epoch_iteration(model, tokenizer, loss_fn, scheduler, train_data_loader, epoch, max_batch, device, mode='train')
        test_mse, test_rmse = epoch_iteration(model, tokenizer, loss_fn, scheduler, test_data_loader, epoch, max_batch, device, mode='test')

        print(f'Epoch {epoch} | Train MSE Loss: {train_mse:.4f}, Train RMSE Loss: {train_rmse:.4f}')
        print(f'{" "*(7+len(str(epoch)))}| Test MSE Loss: {test_mse:.4f}, Test RMSE Loss: {test_rmse:.4f}\n')  

        with open(metrics_csv, "a") as fa:        
            fa.write(f"{epoch},{train_mse},{train_rmse},{test_mse},{test_rmse}\n")
            fa.flush()

        # Save best
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            model_path = os.path.join(run_dir, f'best_saved_model.pth')
            print(f"NEW BEST model: RMSE loss {best_rmse:.4f}")
            save_model(model, optimizer, scheduler, model_path, epoch, test_rmse)
        
        # Save every 10 epochs
        if epoch > 0 and epoch % 10 == 0:
            model_path = os.path.join(run_dir, f'saved_model-epoch_{epoch}.pth')
            save_model(model, optimizer, scheduler, model_path, epoch, test_rmse)

        # Save checkpoint 
        model_path = os.path.join(run_dir, f'checkpoint_saved_model.pth')
        save_model(model, optimizer, scheduler, model_path, epoch, test_rmse)
            
        print("")
        
    plot_log_file(metrics_csv, metrics_img)

    # End timer and print duration
    end_time = time.time()
    duration = end_time - start_time
    print(f'Training and testing complete in {duration:.2f} seconds.')

def epoch_iteration(model, tokenizer, loss_fn, scheduler, data_loader, epoch, max_batch, device, mode):
    """ Used in run_model. """
    
    model.train() if mode=='train' else model.eval()

    data_iter = tqdm.tqdm(enumerate(data_loader),
                          desc=f'Epoch_{mode}: {epoch}',
                          total=len(data_loader),
                          bar_format='{l_bar}{r_bar}')

    total_loss = 0
    total_items = 0

    # Set max_batch if None
    if not max_batch:
        max_batch = len(data_loader)

    for batch, batch_data in data_iter:
        if max_batch > 0 and batch >= max_batch:
            break

        seq_ids, seqs, targets = batch_data
        targets = targets.to(device).float()
        tokenized_seqs = tokenizer(seqs, return_tensors="pt").to(device)
   
        if mode == 'train':
            scheduler.zero_grad()
            preds = model(tokenized_seqs)
            batch_loss = loss_fn(preds, targets)
            batch_loss.backward()
            scheduler.step_and_update_lr()

        else:
            with torch.no_grad():
                preds = model(tokenized_seqs)
                batch_loss = loss_fn(preds, targets)

        total_loss += batch_loss.item()
        total_items += targets.size(0)

        # # Check if gradients are being computed for ESM parameters
        # for name, param in model.esm.named_parameters():
        #     if param.grad is None:
        #         print(f'No gradient for {name}')
        #     else:
        #         print(f'Gradient computed for {name}')
    
    # total loss is the sum of squared errors over items encountered
    # so divide by the number of items encountered
    # we get mse and rmse per item
    mse = total_loss/total_items
    rmse = np.sqrt(mse)

    return mse, rmse 

if __name__=='__main__':

    # Data/results directories
    result_tag = 'expression' # specify expression or binding
    data_dir = os.path.join(os.path.dirname(__file__), f'../../../data/dms') 
    results_dir = os.path.join(os.path.dirname(__file__), f'../../../results/run_results/esm_blstm')

    # Create run directory for results
    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(results_dir, f"esm_blstm-DMS_OLD-{result_tag}-{date_hour_minute}")
    os.makedirs(run_dir, exist_ok = True)

    # Run setup
    n_epochs = 6
    batch_size = 64
    max_batch = -1
    num_workers = 64
    lr = 1e-5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create Dataset and DataLoader
    torch.manual_seed(0)

    train_dataset = DMSDataset(os.path.join(data_dir, "mutation_combined_DMS_OLD_train.csv"), result_tag)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=True)

    test_dataset = DMSDataset(os.path.join(data_dir, "mutation_combined_DMS_OLD_test.csv"), result_tag)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)

    # ESM input
    esm_version = "facebook/esm2_t6_8M_UR50D" 
    esm = EsmModel.from_pretrained(esm_version, cache_dir='../../../../model_downloads').to(device)
    tokenizer = AutoTokenizer.from_pretrained(esm_version, cache_dir='../../../../model_downloads')

    # BLSTM input
    size = 320
    lstm_input_size = size
    lstm_hidden_size = size
    lstm_num_layers = 1        
    lstm_bidrectional = True   
    fcn_hidden_size = size
    blstm = BLSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers, lstm_bidrectional, fcn_hidden_size)

    model = ESM_BLSTM(esm, blstm)

    # Run
    count_parameters(model)
    saved_model_pth = "../../../results/run_results/esm_blstm/esm_blstm-DMS_OLD-expression-2024-10-07_20-44/checkpoint_saved_model.pth"
    from_checkpoint = True
    save_as = f"esm_blstm-DMS_OLD_{result_tag}-train_{len(train_dataset)}_test_{len(test_dataset)}"
    run_model(model, tokenizer, train_data_loader, test_data_loader, n_epochs, lr, max_batch, device, run_dir, save_as, saved_model_pth, from_checkpoint)