#!/usr/bin/env python
"""
Model runner for ESM-BLSTM model (multi target for both binding and expression).
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

from runner_util_dms import (
    DMSDataset_BE,
    count_parameters,
    save_model,
    load_model,
    load_model_checkpoint,
    plot_log_file_BE,
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

        # FCN output layers - two separate heads for binding and expression
        self.binding_head = nn.Linear(fcn_hidden_size, 1)
        self.expression_head = nn.Linear(fcn_hidden_size, 1)

    def forward(self, x):
        num_directions = 2 if self.lstm.bidirectional else 1
        h_0 = torch.zeros(num_directions * self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
        c_0 = torch.zeros(num_directions * self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        lstm_final_out = lstm_out[:, -1, :] 
        fcn_out = self.fcn(lstm_final_out)

        # Task-specific predictions
        binding_pred = self.binding_head(fcn_out).squeeze(1) # [batch_size]
        expression_pred = self.expression_head(fcn_out).squeeze(1) # [batch_size]

        return binding_pred, expression_pred

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
            binding_preds, expression_preds = self.blstm(esm_aa_embedding)
        return binding_preds, expression_preds


# MODEL RUNNING
def run_model(model, tokenizer, train_data_loader, test_data_loader, n_epochs: int, lr:float, max_batch: Union[int, None], device: str, run_dir: str, save_as: str, saved_model_pth:str=None, from_checkpoint:bool=False):
    """ Run a model through train and test epochs. """

    model = model.to(device)
    loss_fn = nn.MSELoss(reduction='sum').to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr)

    metrics_csv = os.path.join(run_dir, f"{save_as}_metrics.csv")
    metrics_img = os.path.join(run_dir, f"{save_as}_metrics.pdf")

    starting_epoch = 1
    best_rmse = float('inf')

    # Load saved model
    if saved_model_pth is not None and os.path.exists(saved_model_pth):
        if from_checkpoint:
            model_state, optimizer_state, starting_epoch, best_rmse = load_model(saved_model_pth, device)

            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
            starting_epoch += 1
            
            if starting_epoch > n_epochs:
                raise ValueError(f"Starting epoch ({starting_epoch}) is greater than the total number of epochs to run ({n_epochs}). Adjust the number of epochs, 'n_epochs'.")
        
        else:
            model_state, _, _, _ = load_model(saved_model_pth, device)
            model.load_state_dict(model_state)

    with open(metrics_csv, "a") as fa:
        if from_checkpoint: load_model_checkpoint(saved_model_pth, metrics_csv, starting_epoch)
        else: 
            fa.write((
                "Epoch,"
                "Train Binding MSE,Train Binding RMSE,Train Expression MSE,Train Expression RMSE,Train MSE,Train RMSE,"
                "Test Binding MSE,Test Binding RMSE,Test Expression MSE,Test Expression RMSE,Test MSE,Test RMSE\n"
            ))

    # Running
    start_time = time.time()

    for epoch in range(starting_epoch, n_epochs + 1):
        train_binding_mse, train_binding_rmse, train_expression_mse, train_expression_rmse, train_mse, train_rmse = epoch_iteration(model, tokenizer, loss_fn, optimizer, train_data_loader, epoch, max_batch, device, mode='train')
        test_binding_mse, test_binding_rmse, test_expression_mse, test_expression_rmse, test_mse, test_rmse = epoch_iteration(model, tokenizer, loss_fn, optimizer, test_data_loader, epoch, max_batch, device, mode='test')

        print(f'Epoch {epoch} | Train Binding RMSE: {train_binding_rmse:.4f}, Train Expression RMSE: {train_expression_rmse:.4f}, Train RMSE: {train_rmse:.4f}') 
        print(f'{" "*(8+len(str(epoch)))} Test Binding RMSE: {test_binding_rmse:.4f}, Test Expression RMSE: {test_expression_rmse:.4f}, Test RMSE: {test_rmse:.4f}') 

        with open(metrics_csv, "a") as fa:         
            fa.write((
                f"{epoch}," 
                f"{train_binding_mse}, {train_binding_rmse}, {train_expression_mse}, {train_expression_rmse}, {train_mse}, {train_rmse},"
                f"{test_binding_mse}, {test_binding_rmse}, {test_expression_mse}, {test_expression_rmse}, {test_mse}, {test_rmse}\n"
            ))                
            fa.flush()

        # Save best
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            model_path = os.path.join(run_dir, f'best_saved_model.pth')
            print(f"NEW BEST model: RMSE loss {best_rmse:.4f}")
            save_model(model, optimizer, model_path, epoch, test_rmse)
        
        # Save every 10 epochs
        if epoch > 0 and epoch % 10 == 0:
            model_path = os.path.join(run_dir, f'saved_model-epoch_{epoch}.pth')
            save_model(model, optimizer, model_path, epoch, test_rmse)

        # Save checkpoint 
        model_path = os.path.join(run_dir, f'checkpoint_saved_model.pth')
        save_model(model, optimizer, model_path, epoch, test_rmse)
            
        print("")
        
    plot_log_file_BE(metrics_csv, metrics_img)

    # End timer and print duration
    end_time = time.time()
    duration = end_time - start_time
    formatted_duration = str(datetime.timedelta(seconds=duration))
    print(f'Training and testing complete in: {formatted_duration} (D day(s), H:MM:SS.microseconds)')

def epoch_iteration(model, tokenizer, loss_fn, optimizer, data_loader, epoch, max_batch, device, mode):
    """ Used in run_model. """
    
    model.train() if mode=='train' else model.eval()

    data_iter = tqdm.tqdm(enumerate(data_loader),
                          desc=f'Epoch_{mode}: {epoch}',
                          total=len(data_loader),
                          bar_format='{l_bar}{r_bar}')

    total_binding_loss, total_expression_loss, total_loss = 0, 0, 0
    total_items = 0

    # Set max_batch if None
    if not max_batch:
        max_batch = len(data_loader)

    for batch, batch_data in data_iter:
        if max_batch > 0 and batch >= max_batch:
            break

        seq_ids, seqs, binding_targets, expression_targets = batch_data
        binding_targets, expression_targets = binding_targets.to(device).float(), expression_targets.to(device).float()
        tokenized_seqs = tokenizer(seqs, return_tensors="pt").to(device)
   
        if mode == 'train':
            optimizer.zero_grad()
            binding_preds, expression_preds = model(tokenized_seqs)
            binding_loss = loss_fn(binding_preds, binding_targets)
            expression_loss = loss_fn(expression_preds, expression_targets)
            batch_loss = binding_loss + expression_loss
            batch_loss.backward()
            optimizer.step()

        else:
            with torch.no_grad():
                binding_preds, expression_preds = model(tokenized_seqs)
                binding_loss = loss_fn(binding_preds, binding_targets)
                expression_loss = loss_fn(expression_preds, expression_targets)
                batch_loss = binding_loss + expression_loss

        total_binding_loss += binding_loss.item()
        total_expression_loss += expression_loss.item()
        total_loss += batch_loss.item()
        total_items += binding_targets.size(0)  # same size as expression_targets.size(0)
    
    # total loss is the sum of squared errors over items encountered
    # so divide by the number of items encountered
    # we get mse and rmse per item
    binding_mse = total_binding_loss/total_items
    expression_mse = total_expression_loss/total_items
    mse = total_loss/total_items

    binding_rmse = np.sqrt(binding_mse)
    expression_rmse = np.sqrt(expression_mse)
    rmse = np.sqrt(mse)

    return binding_mse, binding_rmse, expression_mse, expression_rmse, mse, rmse

if __name__=='__main__':

    # Data/results directories
    data_dir = os.path.join(os.path.dirname(__file__), f'../../../data/dms') 
    results_dir = os.path.join(os.path.dirname(__file__), f'../../../results/run_results/esm_blstm')

    # Create run directory for results
    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(results_dir, f"esm_blstm-DMS_OLD_BE-{date_hour_minute}")
    os.makedirs(run_dir, exist_ok = True)

    # Run setup
    n_epochs = 5000
    batch_size = 64
    max_batch = -1
    num_workers = 64
    lr = 1e-5
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Create Dataset and DataLoader
    torch.manual_seed(0)

    train_dataset = DMSDataset_BE(os.path.join(data_dir, "mutation_combined_DMS_OLD_train.csv"))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=True)

    test_dataset = DMSDataset_BE(os.path.join(data_dir, "mutation_combined_DMS_OLD_test.csv"))
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
    saved_model_pth = None
    from_checkpoint = False
    save_as = f"esm_blstm-DMS_OLD_BE-train_{len(train_dataset)}_test_{len(test_dataset)}"
    run_model(model, tokenizer, train_data_loader, test_data_loader, n_epochs, lr, max_batch, device, run_dir, save_as, saved_model_pth, from_checkpoint)