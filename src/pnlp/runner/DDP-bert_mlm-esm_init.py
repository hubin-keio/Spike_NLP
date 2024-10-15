#!/usr/bin/env python
"""
Model runner for ESM-initialized BERT-MLM model. We utilize DistributedDataParallel.

*GPU ONLY* to avoid device assignment issues.

Usage:
    torchrun
    > --standalone: utilize single node
    > --nproc_per_node: number of processes/gpus

    Example equivalent commands to run (single node, 4 gpu; top is for bio-lambda cluster, bottom is generic):
        /data/miniconda3/envs/spike_env/bin/time -v torchrun --standalone --nproc_per_node=4 DDP-bert_mlm-esm_init.py 
        /usr/bin/time -v torchrun --standalone --nproc_per_node=4 DDP-bert_mlm-esm_init.py 
"""
import os
import re
import sys
import math
import tqdm
import time
import torch
import random
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

from runner_util_rbd_bert_mlm import (
    RBDDataset,
    ScheduledOptim,
    count_parameters,
    save_model,
    load_model,
    load_model_checkpoint,
    plot_log_file,
    plot_aa_preds_heatmap
)

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record

# MODEL RUNNING
def run_model(model, tokenizer, train_data_loader, test_data_loader, n_epochs: int, lr:float, max_batch: Union[int, None], device: str, run_dir: str, save_as: str, saved_model_pth:str=None, from_checkpoint:bool=False):
    """ Run a model through train and test epochs. """

    loss_fn = nn.CrossEntropyLoss(reduction='sum').to(device)  # sum of CEL at batch level.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr*dist.get_world_size(), betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = ScheduledOptim(
        optimizer, 
        d_model=model.bert.embedding_dim, 
        n_warmup_steps=(len(train_data_loader.dataset) / (train_data_loader.batch_size/dist.get_world_size())) * 0.1
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
        
        dist.barrier()

    with open(metrics_csv, "a") as fa:
        if from_checkpoint: 
            load_model_checkpoint(saved_model_pth, metrics_csv, starting_epoch)
        else: 
            if dist.get_rank() == 0: fa.write(f"Epoch,Train Accuracy,Train Loss,Test Accuracy,Test Loss\n")

    # Wrap model, sync proccesses - DDP 
    model = DDP(model.to(device), device_ids=[device], output_device=device, find_unused_parameters=True)
    dist.barrier()

    # Running
    start_time = time.time()

    for epoch in range(starting_epoch, n_epochs + 1):
        train_data_loader.sampler.set_epoch(epoch)
        test_data_loader.sampler.set_epoch(epoch)

        train_accuracy, train_loss = epoch_iteration(model, tokenizer, loss_fn, scheduler, train_data_loader, epoch, max_batch, device, mode='train')
        dist.barrier()
        test_accuracy, test_loss, aa_pred_counter = epoch_iteration(model, tokenizer, loss_fn, scheduler, test_data_loader, epoch, max_batch, device, mode='test')
        dist.barrier()

        # Predictions from all processes - DDP
        aa_pred_counter_list = [None] * dist.get_world_size()       
        dist.all_gather_object(aa_pred_counter_list, aa_pred_counter)

        if dist.get_rank() == 0:
            print(f'Epoch {epoch} | Train Accuracy: {train_accuracy:.4f}, Train Loss: {train_loss:.4f}')
            print(f'{" "*(7+len(str(epoch)))}| Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}\n') 
            
            with open(metrics_csv, "a") as fa:         
                fa.write(f"{epoch},{train_accuracy},{train_loss},{test_accuracy},{test_loss}\n")
                fa.flush()

            # Combine predictions from all processes - DDP
            for counter in aa_pred_counter_list:
                for key, value in counter.items():
                    if key not in aa_preds_tracker:
                        aa_preds_tracker[key] = {}
                    aa_preds_tracker[key][epoch] = value

            # Save best
            if test_accuracy > best_accuracy or (test_accuracy == best_accuracy and test_loss < best_loss):
                best_accuracy = test_accuracy
                best_loss = test_loss
                model_path = os.path.join(run_dir, f'best_saved_model.pth')
                print(f"NEW BEST model: accuracy {best_accuracy:.4f} and loss {best_loss:.4f}")
                save_model(model, optimizer, scheduler, model_path, epoch, best_accuracy, best_loss)
            
            # Save every 10 epochs
            if epoch > 0 and epoch % 10 == 0:
                model_path = os.path.join(run_dir, f'saved_model-epoch_{epoch}.pth')
                save_model(model, optimizer, scheduler, model_path, epoch, test_accuracy, test_loss)

            # Save checkpoint 
            model_path = os.path.join(run_dir, f'checkpoint_saved_model.pth')
            save_model(model, optimizer, scheduler, model_path, epoch, test_accuracy, test_loss)

            print("")
    
    if dist.get_rank() == 0:
        # Write amino acid predictions
        with open(preds_csv, 'w') as fb:
            header = ", ".join(f"epoch {epoch}" for epoch in range(1, n_epochs + 1))
            fb.write(f"expected_aa->predicted_aa, {header}\n")

            # Write the data for each key in aa_preds_tracker
            for key, epoch_values in aa_preds_tracker.items():
                # Ensure that all epochs are covered and fill missing epochs with 0
                predictions_per_epoch = [epoch_values.get(epoch, 0) for epoch in range(1, n_epochs + 1)]
                data_row = ", ".join(map(str, predictions_per_epoch))
                fb.write(f"{key}, {data_row}\n")

        plot_log_file(metrics_csv, metrics_img)
        plot_aa_preds_heatmap(preds_csv, preds_img)

    # End timer and print duration
    end_time = time.time()
    duration = end_time - start_time
    formatted_duration = str(datetime.timedelta(seconds=duration))
    if dist.get_rank() == 0:
        print(f'Training and testing complete in: {formatted_duration} (D day(s), H:MM:SS.microseconds)')

def epoch_iteration(model, tokenizer, loss_fn, scheduler, data_loader, epoch, max_batch, device, mode):
    """ Used in run_model. """
    
    model.train() if mode == 'train' else model.eval()

    if dist.get_rank() == 0:
        data_iter = tqdm.tqdm(enumerate(data_loader),
                            desc=f'Epoch_{mode}: {epoch}',
                            total=len(data_loader),
                            bar_format='{l_bar}{r_bar}')
    else:
        data_iter = enumerate(data_loader)

    total_loss = 0
    total_masked = 0
    correct_predictions = 0
    aa_pred_counter = defaultdict(int)

    # Set max_batch if None
    if not max_batch:
        max_batch = len(data_loader)

    for batch_idx, batch_data in data_iter:
        if max_batch > 0 and batch_idx >= max_batch:
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
    
    # Reduce metrics across all processes (sum them) to the main process (dst=0) - DDP
    total_loss = torch.tensor(total_loss).to(device)
    correct_predictions = torch.tensor(correct_predictions).to(device)
    total_masked = torch.tensor(total_masked).to(device)

    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_predictions, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_masked, op=dist.ReduceOp.SUM)

    # Initialize avg_accuracy and avg_loss to None for non-rank 0 processes - DDP
    avg_loss = None
    avg_accuracy = None

    # Calculate average loss and accuracy per masked token (only on rank 0) - DDP
    if dist.get_rank() == 0:
        avg_loss = total_loss.item() / total_masked.item()
        avg_accuracy = (correct_predictions.item() / total_masked.item()) * 100

    if mode == "train": return avg_accuracy, avg_loss
    else: return avg_accuracy, avg_loss, aa_pred_counter

@record
def main():
    # Initialize process group - DDP
    dist.init_process_group(
        backend='nccl', 
        timeout=datetime.timedelta(seconds=5400), 
        rank=int(os.environ['RANK']), 
        world_size=int(os.environ['WORLD_SIZE']) 
    )

    # Data/results directories
    data_dir = os.path.join(os.path.dirname(__file__), f'../../../data/rbd')
    results_dir = os.path.join(os.path.dirname(__file__), f'../../../results/run_results/DDP-bert_mlm-esm_init')

    # Create run directory for results
    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(results_dir, f"DDP-bert_mlm-esm_init-RBD-{date_hour_minute}")
    os.makedirs(run_dir, exist_ok = True)

    # Run setup
    n_epochs = 10
    batch_size = 64
    max_batch = -1
    num_workers = 64
    lr = 1e-5

    # Check if it can run on GPU - DDP
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else sys.exit(1)
    torch.cuda.set_device(local_rank)

    # Create Dataset and DataLoader, use DistributedSampler - DDP
    seed = 0
    torch.manual_seed(seed)

    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id + local_rank * num_workers  # Unique seed per worker and rank
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_dataset = RBDDataset(os.path.join(data_dir, "spikeprot0528.clean.uniq.noX.RBD.metadata.variants_train.csv"))
    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=seed)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers, worker_init_fn=worker_init_fn, pin_memory=True, sampler=train_sampler)

    test_dataset = RBDDataset(os.path.join(data_dir, "spikeprot0528.clean.uniq.noX.RBD.metadata.variants_test.csv"))
    test_sampler = DistributedSampler(test_dataset, shuffle=False, seed=seed)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, worker_init_fn=worker_init_fn, pin_memory=True, sampler=test_sampler)

    # BERT input
    max_len = 280
    mask_prob = 0.15
    embedding_dim = 320 
    dropout = 0.1
    n_transformer_layers = 12
    n_attn_heads = 10

    bert = BERT(embedding_dim, dropout, max_len, mask_prob, n_transformer_layers, n_attn_heads)
    bert.embedding.load_pretrained_embeddings(os.path.join(data_dir, 'esm_weights-embedding_dim320.pth'), no_grad=False)
    tokenizer = ProteinTokenizer(max_len, mask_prob)

    model = ProteinLM(bert, vocab_size=len(token_to_index))

    # Run
    if dist.get_rank() == 0: 
        count_parameters(model)
    saved_model_pth = None
    from_checkpoint = False
    save_as = f"DDP-bert_mlm-esm_init-RBD-train_{len(train_dataset)}_test_{len(test_dataset)}"
    run_model(model, tokenizer, train_data_loader, test_data_loader, n_epochs, lr, max_batch, device, run_dir, save_as, saved_model_pth, from_checkpoint)

    # Clean up - DDP
    dist.destroy_process_group() 

if __name__=='__main__':
    main()
