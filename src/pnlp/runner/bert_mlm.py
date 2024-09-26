#!/usr/bin/env python
"""
Model runner for BERT-MLM model (NOT ESM-initialized).
"""

import os
import re
import sys
import math
import tqdm
import time
import torch
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import Union
from prettytable import PrettyTable
from collections import defaultdict

from pnlp.model.language import BERT, ProteinLM
from pnlp.embedding.tokenizer import ProteinTokenizer, token_to_index

from bert_mlm_runner_util import (
    RBDDataset,
    ScheduledOptim,
    count_parameters,
    save_model,
    load_model,
    load_model_checkpoint,
    plot_log_file,
    plot_aa_preds_heatmap
)


# MODEL RUNNING
def run_model(model, tokenizer, train_data_loader, test_data_loader, n_epochs: int, lr:float, max_batch: Union[int, None], device: str, run_dir: str, save_as: str, saved_model_pth:str=None, from_checkpoint:bool=False):
    """ Run a model through train and test epochs. """

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss(reduction='sum').to(device)  # sum of CEL at batch level.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = ScheduledOptim(
        optimizer, 
        d_model=model.bert.embedding_dim, 
        n_warmup_steps=(len(train_data_loader.dataset) / train_data_loader.batch_size) * 0.1
    ) 

    metrics_csv = os.path.join(run_dir, f"{save_as}_metrics.csv")
    metrics_img = os.path.join(run_dir, f"{save_as}_metrics.pdf")
    preds_csv = os.path.join(run_dir, f"{save_as}_predictions.csv")
    preds_img = os.path.join(run_dir, f"{save_as}_predictions.pdf")

    starting_epoch = 1
    best_accuracy = 0
    best_loss = float('inf')
    aa_preds_tracker = {}
    
    # Load saved model
    if saved_model_pth is not None and os.path.exists(saved_model_pth):
        if from_checkpoint:
            model_state, optimizer_state, scheduler_state, starting_epoch, best_accuracy, best_loss = load_model(saved_model_pth, device)

            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
            starting_epoch += 1
            
            if starting_epoch > n_epochs:
                raise ValueError(f"Starting epoch ({starting_epoch}) is greater than the total number of epochs to run ({n_epochs}). Adjust the number of epochs, 'n_epochs'.")
        
        else:
            model_state, _, _, _, _ = load_model(saved_model_pth, device)
            model.load_state_dict(model_state)

    with open(metrics_csv, "a") as fa:
        if from_checkpoint: load_model_checkpoint(saved_model_pth, metrics_csv, starting_epoch)
        else: fa.write(f"Epoch,Train Accuracy,Train Loss,Test Accuracy,Test Loss\n")

    # Running
    start_time = time.time()

    for epoch in range(starting_epoch, n_epochs + 1):
        train_accuracy, train_loss = epoch_iteration(model, tokenizer, loss_fn, scheduler, train_data_loader, epoch, max_batch, device, mode='train')
        test_accuracy, test_loss, aa_pred_counter = epoch_iteration(model, tokenizer, loss_fn, scheduler, test_data_loader, epoch, max_batch, device, mode='test')

        print(f'Epoch {epoch} | Train Accuracy: {train_accuracy:.4f}, Train Loss: {train_loss:.4f}')
        print(f'{" "*(7+len(str(epoch)))}| Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}\n') 
        
        with open(metrics_csv, "a") as fa:         
            fa.write(f"{epoch},{train_accuracy},{train_loss},{test_accuracy},{test_loss}\n")
            fa.flush()

        for key, value in aa_pred_counter.items():
            if key not in aa_preds_tracker:
                aa_preds_tracker[key] = {}
            aa_preds_tracker[key][epoch] = value

        # Save best
        if test_accuracy > best_accuracy or (test_accuracy == best_accuracy and test_loss < best_loss):
            best_accuracy = test_accuracy
            best_loss = test_loss
            model_path = os.path.join(run_dir, f'best_saved_model.pth')
            print(f"NEW BEST model: accuracy {best_accuracy:.4f} and loss {best_loss:.4f}")
            save_model(model, optimizer, scheduler, model_path, epoch, test_accuracy, test_loss)
        
        # Save every 10 epochs
        if epoch > 0 and epoch % 10 == 0:
            model_path = os.path.join(run_dir, f'saved_model-epoch_{epoch}.pth')
            save_model(model, optimizer, scheduler, model_path, epoch, test_accuracy, test_loss)

        # Save checkpoint 
        model_path = os.path.join(run_dir, f'checkpoint_saved_model.pth')
        save_model(model, optimizer, scheduler, model_path, epoch, test_accuracy, test_loss)
            
        print("")

    # Write amino acid predictions
    with open(preds_csv, 'w') as fb:
        header = ", ".join(f"epoch {epoch}" for epoch in range(1, n_epochs + 1))
        fb.write(f"expected_aa->predicted_aa, {header}\n")

        for key, values in aa_preds_tracker.items():
            predictions_per_epoch = [values.get(epoch, 0) for epoch in range(1, n_epochs + 1)]
            data_row = ", ".join(map(str, predictions_per_epoch))
            fb.write(f"{key}, {data_row}\n")

    plot_log_file(metrics_csv, metrics_img)
    plot_aa_preds_heatmap(preds_csv, preds_img)

    # End timer and print duration
    end_time = time.time()
    duration = end_time - start_time
    formatted_duration = str(datetime.timedelta(seconds=duration))
    print(f'Training and testing complete in: {formatted_duration} (D day(s), H:MM:SS.microseconds)')

def epoch_iteration(model, tokenizer, loss_fn, scheduler, data_loader, epoch, max_batch, device, mode):
    """ Used in run_model. """
    
    model.train() if mode=='train' else model.eval()

    data_iter = tqdm.tqdm(enumerate(data_loader),
                          desc=f'Epoch_{mode}: {epoch}',
                          total=len(data_loader),
                          bar_format='{l_bar}{r_bar}')

    total_loss = 0
    total_masked = 0
    correct_predictions = 0
    aa_pred_counter = defaultdict(int)

    # Set max_batch if None
    if not max_batch:
        max_batch = len(data_loader)

    for batch, batch_data in data_iter:
        if max_batch > 0 and batch >= max_batch:
            break

        seq_ids, seqs = batch_data
        masked_tokenized_seqs = tokenizer(seqs).to(device) 
        unmasked_tokenized_seqs = tokenizer._batch_pad(seqs).to(device)
   
        if mode == 'train':
            scheduler.zero_grad()
            preds = model(masked_tokenized_seqs)
            batch_loss = loss_fn(preds.transpose(1, 2), unmasked_tokenized_seqs)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scheduler.step_and_update_lr()

        else:
            with torch.no_grad():
                preds = model(masked_tokenized_seqs)
                batch_loss = loss_fn(preds.transpose(1, 2), unmasked_tokenized_seqs)

        # Loss
        total_loss += batch_loss.item()

        # Accuracy
        predicted_tokens = torch.max(preds, dim=-1)[1]
        masked_locations = torch.nonzero(torch.eq(masked_tokenized_seqs, token_to_index['<MASK>']), as_tuple=True)
        correct_predictions += torch.eq(predicted_tokens[masked_locations], unmasked_tokenized_seqs[masked_locations]).sum().item()
        total_masked += masked_locations[0].numel()     

        # Track amino acid predictions
        if mode == "test":
            token_to_aa = {i:aa for i, aa in enumerate('ACDEFGHIKLMNPQRSTUVWXY')}                
            aa_keys = [f"{token_to_aa.get(token.item())}->{token_to_aa.get(pred_token.item())}" for token, pred_token in zip(unmasked_tokenized_seqs[masked_locations], predicted_tokens[masked_locations])]
            for aa_key in aa_keys:
                aa_pred_counter[aa_key] += 1

    # Average accuracy/loss per masked token
    avg_loss = total_loss / total_masked
    avg_accuracy = (correct_predictions / total_masked) * 100

    if mode == "train": return avg_accuracy, avg_loss
    else: return avg_accuracy, avg_loss, aa_pred_counter

if __name__=='__main__':

    # Data/results directories
    data_dir = os.path.join(os.path.dirname(__file__), f'../../../data/rbd')
    results_dir = os.path.join(os.path.dirname(__file__), f'../../../results/run_results/bert_mlm')

    # Create run directory for results
    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(results_dir, f"bert_mlm-rbd-{date_hour_minute}")
    os.makedirs(run_dir, exist_ok = True)

    # Run setup
    n_epochs = 100
    batch_size = 64
    max_batch = -1
    num_workers = 64
    lr = 1e-5
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # Create Dataset and DataLoader
    torch.manual_seed(0)

    train_dataset = RBDDataset(os.path.join(data_dir, "spikeprot0528.clean.uniq.noX.RBD.metadata.variants_train.csv"))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=True)

    test_dataset = RBDDataset(os.path.join(data_dir, "spikeprot0528.clean.uniq.noX.RBD.metadata.variants_test.csv"))
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)

    # BERT input
    max_len = 280
    mask_prob = 0.15
    embedding_dim = 320 
    dropout = 0.1
    n_transformer_layers = 12
    n_attn_heads = 10

    bert = BERT(embedding_dim, dropout, max_len, mask_prob, n_transformer_layers, n_attn_heads)
    tokenizer = ProteinTokenizer(max_len, mask_prob)

    model = ProteinLM(bert, vocab_size=len(token_to_index))

    # Run
    count_parameters(model)
    saved_model_pth = "Spike_NLP_kaetlyn/results/run_results/bert_mlm/bert_mlm-rbd-2024-09-24_20-31/best_saved_model.pth"
    from_checkpoint = False
    save_as = f"bert_mlm-RBD-train_{len(train_dataset)}_test_{len(test_dataset)}"
    run_model(model, tokenizer, train_data_loader, test_data_loader, n_epochs, lr, max_batch, device, run_dir, save_as, saved_model_pth, from_checkpoint)