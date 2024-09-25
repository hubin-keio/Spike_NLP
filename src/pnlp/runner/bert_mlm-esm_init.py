#!/usr/bin/env python
"""
Model runner for ESM-initialized BERT-MLM model.
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


# SCHEDULER FOR OPTIMIZER
class ScheduledOptim():
    """A simple wrapper class for learning rate scheduling from BERT-pytorch.

    Author: codertimo
    https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/trainer/optim_schedule.py
    """

    def __init__(self, optimizer, d_model: int, n_warmup_steps):
        self._optimizer=optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        """Learning rate scheduling per step"""
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

# DATASET    
class RBDDataset(Dataset):
    def __init__(self, csv_file:str):

        try:
            self.full_df = pd.read_csv(csv_file, sep=',', header=0)
        except (FileNotFoundError, pd.errors.ParserError, Exception) as e:
            print(f"Error reading in .csv file: {csv_file}\n{e}", file=sys.stderr)
            sys.exit(1)

    def __len__(self) -> int:
        return len(self.full_df)

    def __getitem__(self, idx):
        # label, seq, target
        return self.full_df['seq_id'][idx], self.full_df['sequence'][idx]

# HELPER FUNCTIONS
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

def save_model(model, optimizer, path_to_pth, epoch, accuracy, loss):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'accuracy': accuracy,
        'loss': loss}, path_to_pth
    )
    print(f"Model and optimizer state saved to {path_to_pth}")

def load_model(model, optimizer, saved_model_pth, device):
    saved_state = torch.load(saved_model_pth, map_location=device)
    model.load_state_dict(saved_state['model_state_dict'])
    optimizer.load_state_dict(saved_state['optimizer_state_dict'])
    epoch = saved_state['epoch']
    accuracy = saved_state['accuracy']
    loss = saved_state['loss']
    print(f"Loaded model from {saved_model_pth} at epoch {epoch}, accuracy {accuracy}, loss {loss}")
    return model.to(device), optimizer, epoch, accuracy, loss

def load_model_checkpoint(path_to_pth, metrics_csv, starting_epoch):
    """ Load model data csv, and model pth. """
    folder_path = os.path.dirname(path_to_pth)
    files_in_folder = os.listdir(folder_path)
    metrics_file_name = [file for file in files_in_folder if file.endswith("_metrics.csv")][0]
    saved_metrics_file = os.path.join(folder_path, metrics_file_name)

    with open(saved_metrics_file, "r") as fa, open(metrics_csv, "w") as fb:
        header = fa.readline()
        fb.write(header)

        for line in fa:
            epoch = int(line.split(',')[0])
            if epoch == starting_epoch:
                break
            fb.write(line)

def plot_log_file(metrics_csv, metrics_img):
    df = pd.read_csv(metrics_csv)

    sns.set_theme(style="darkgrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 18))  # 2 rows, 1 column
    fontsize = 28

    # Plot Loss
    ax1.plot(df['Epoch'], df['Train Loss'], label='Train Loss', color='tab:red', linewidth=3)
    ax1.plot(df['Epoch'], df['Test Loss'], label='Test Loss', color='tab:orange', linewidth=3)
    ax1.tick_params(axis='x', labelsize=fontsize)
    ax1.set_ylim(-0.5e6, 14e6) 
    ax1.set_ylabel('Loss', fontsize=fontsize)
    ax1.tick_params(axis='y', labelsize=fontsize)
    ax1.legend(loc='upper right', fontsize=fontsize)

    # Plot Accuracy
    ax2.plot(df['Epoch'], df['Train Accuracy'], label='Train Accuracy', color='tab:blue', linewidth=3)
    ax2.plot(df['Epoch'], df['Test Accuracy'], label='Test Accuracy', color='tab:green', linewidth=3)
    ax2.set_xlabel('Epochs', fontsize=fontsize)
    ax2.tick_params(axis='x', labelsize=fontsize)
    ax2.set_ylim(0, 100) 
    ax2.set_ylabel('Accuracy', fontsize=fontsize)
    ax2.tick_params(axis='y', labelsize=fontsize)
    ax2.legend(loc='lower right', fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(metrics_img, format='pdf')
    plt.savefig(metrics_img.replace('.pdf', '.png'), format='png')

def plot_aa_preds_heatmap(preds_csv, preds_img):
    """ Plots heatmap of expected vs predicted amino acid incorrect prediction counts. Expected on x axis. """
    df = pd.read_csv(preds_csv)

    # Create a DataFrame with all possible amino acid combinations
    ALL_AAS = 'ACDEFGHIKLMNPQRSTVWY'
    all_combinations = [(e_aa, p_aa) for e_aa in ALL_AAS for p_aa in ALL_AAS]
    all_df = pd.DataFrame(all_combinations, columns=["Expected", "Predicted"])

    # Split 'expected_aa->predicted_aa' into separate columns
    df[['Expected', 'Predicted']] = df['expected_aa->predicted_aa'].str.split('->', expand=True)

    # Ensure that the epoch columns are numeric and fill any NaNs with 0
    epoch_columns = df.columns[1:-2]  # Assuming epoch columns start at index 1 and go up to the second last column
    df[epoch_columns] = df[epoch_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Sum the counts across all epochs to get the total error count for each expected->predicted pair
    df['Total Count'] = df[epoch_columns].sum(axis=1)

    # Merge with all possible amino acid combinations so missing pairs get a count of 0
    df = pd.merge(all_df, df[['Expected', 'Predicted', 'Total Count']], how="left", on=["Expected", "Predicted"])
    df["Total Count"].fillna(0, inplace=True)

    # Calculate the total counts for each expected amino acid
    total_counts = df.groupby("Expected")["Total Count"].sum()
    df["Expected Total"] = df["Expected"].map(total_counts)

    # Calculate error percentage
    df["Error Percentage"] = (df["Total Count"] / df["Expected Total"]) * 100
    df["Error Percentage"].fillna(0, inplace=True)

    # Pivot the DataFrame to create a heatmap data structure
    heatmap_data = df.pivot_table(index="Predicted", columns="Expected", values="Error Percentage")

    # Set figure size
    plt.figure(figsize=(16, 9))
    fontsize=16

    # Plot
    cmap = sns.color_palette("rocket_r", as_cmap=True)
    heatmap = sns.heatmap(
        heatmap_data,
        annot=True, fmt=".2f",
        linewidth=.5,
        cmap=cmap, vmin=0, vmax=100,
        annot_kws={"size": 13},
        cbar_kws={'drawedges': False, 'label': 'Prediction Rate (%)'}
    )

    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=fontsize)  # Set colorbar tick label size
    colorbar.set_label('Prediction Rate (%)', size=fontsize)  # Set colorbar label size

    plt.ylabel('Predicted Amino Acid', fontsize=fontsize)
    plt.xlabel('Expected Amino Acid', fontsize=fontsize)
    plt.xticks(rotation=0, fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)

    plt.tight_layout()
    plt.savefig(preds_img, format='pdf')
    plt.savefig(preds_img.replace('.pdf', '.png'), format='png')

# MODEL RUNNING
def run_model(model, tokenizer, train_data_loader, test_data_loader, n_epochs: int, lr:float, max_batch: Union[int, None], device: str, run_dir: str, save_as: str, saved_model_pth:str=None, from_checkpoint:bool=False):
    """ Run a model through train and test epochs. """

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss(reduction='sum').to(device)  # sum of CEL at batch level.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    optimizer_scheduler = ScheduledOptim(
        optimizer, 
        d_model=model.bert.embedding_dim, 
        n_warmup_steps=(len(train_data_loader.dataset) / train_data_loader.batch_size) * 0.1
    ) 

    starting_epoch = 1
    best_accuracy = 0
    best_loss = float('inf')

    # Load saved model if provided
    if saved_model_pth is not None:
        model, optimizer, starting_epoch, best_accuracy, best_loss = load_model(model, optimizer, saved_model_pth, device)
        starting_epoch += 1

    start_time = time.time()
    
    metrics_csv = os.path.join(run_dir, f"{save_as}_metrics.csv")
    metrics_img = os.path.join(run_dir, f"{save_as}_metrics.pdf")
    preds_csv = os.path.join(run_dir, f"{save_as}_predictions.csv")
    preds_img = os.path.join(run_dir, f"{save_as}_predictions.pdf")
    aa_preds_tracker = {}
    
    # If saved model, skip to predictions 
    # Except when from checkpoint (still need saved model)
    if saved_model_pth is None or (from_checkpoint and saved_model_pth is not None):

        if starting_epoch > n_epochs:
            raise ValueError(f"Starting epoch ({starting_epoch}) is greater than the total number of epochs to run ({n_epochs}). Adjust the number of epochs, 'n_epochs'.")
        
        # Write metrics
        with open(metrics_csv, "a") as fa:
            if from_checkpoint: load_model_checkpoint(saved_model_pth, metrics_csv, starting_epoch)
            else: fa.write(f"Epoch,Train Accuracy,Train Loss,Test Accuracy,Test Loss\n")

            for epoch in range(starting_epoch, n_epochs + 1):
                if not max_batch:
                    max_batch = len(train_data_loader)

                train_accuracy, train_loss = epoch_iteration(model, tokenizer, loss_fn, optimizer_scheduler, train_data_loader, epoch, max_batch, device, mode='train')
                test_accuracy, test_loss, aa_pred_counter = epoch_iteration(model, tokenizer, loss_fn, optimizer_scheduler, test_data_loader, epoch, max_batch, device, mode='test')

                print(f'Epoch {epoch} | Train Accuracy: {train_accuracy:.4f}, Train Loss: {train_loss:.4f}')
                print(f'{" "*(7+len(str(epoch)))}| Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}\n')          
                fa.write(f"{epoch},{train_accuracy},{train_loss},{test_accuracy},{test_loss}\n")
                fa.flush()

                for key in aa_pred_counter:
                    if key not in aa_preds_tracker :
                        aa_preds_tracker[key] = {}
                    aa_preds_tracker[key][epoch] = aa_pred_counter.get(key)

                # Save best
                if test_accuracy > best_accuracy or (test_accuracy == best_accuracy and test_loss < best_loss):
                    best_accuracy = test_accuracy
                    best_loss = test_loss
                    model_path = os.path.join(run_dir, f'best_saved_model.pth')
                    print(f"NEW BEST model: accuracy {best_accuracy:.4f} and loss {best_loss:.4f}")
                    save_model(model, optimizer, model_path, epoch, best_accuracy, best_loss)
                
                # Save every 10 epochs
                if epoch > 0 and epoch % 10 == 0:
                    model_path = os.path.join(run_dir, f'saved_model-epoch_{epoch}.pth')
                    save_model(model, optimizer, model_path, epoch, test_accuracy, test_loss)

                # Save checkpoint 
                model_path = os.path.join(run_dir, f'checkpoint_saved_model.pth')
                save_model(model, optimizer, model_path, epoch, test_accuracy, test_loss)
                    
                print("")
        
        plot_log_file(metrics_csv, metrics_img)

        # Write Predictions
        with open(preds_csv, 'w') as fb:
            header = ", ".join(f"epoch {epoch}" for epoch in range(1, n_epochs + 1))
            header = f"expected_aa->predicted_aa, {header}\n"
            fb.write(header)

            for key in aa_preds_tracker:
                aa_preds_tracker[key] = [aa_preds_tracker[key].get(epoch, 0) for epoch in range(1, n_epochs + 1)]
                data_row = ", ".join(str(val) for val in aa_preds_tracker[key])
                fb.write(f"{key}, {data_row}\n")

        plot_aa_preds_heatmap(preds_csv, preds_img)

    # End timer and print duration
    end_time = time.time()
    duration = end_time - start_time
    formatted_duration = str(datetime.timedelta(seconds=duration))
    days, hrs, minutes = formatted_duration.split(":")
    print(f'Training and testing complete in: {days} days, {hrs} hours, {minutes} minutes.')

def epoch_iteration(model, tokenizer, loss_fn, optimizer_scheduler, data_loader, epoch, max_batch, device, mode):
    """ Used in run_model. """
    
    model.train() if mode=='train' else model.eval()

    data_iter = tqdm.tqdm(enumerate(data_loader),
                          desc=f'Epoch_{mode}: {epoch}',
                          total=len(data_loader),
                          bar_format='{l_bar}{r_bar}')

    total_loss = 0
    total_items = 0
    total_masked = 0
    correct_predictions = 0
    aa_pred_counter = defaultdict(int)

    for batch, batch_data in data_iter:
        if max_batch > 0 and batch >= max_batch:
            break

        seq_ids, seqs = batch_data
        masked_tokenized_seqs = tokenizer(seqs).to(device) 
        unmasked_tokenized_seqs = tokenizer._batch_pad(seqs).to(device)
   
        if mode == 'train':
            optimizer_scheduler.zero_grad()
            preds = model(masked_tokenized_seqs)
            batch_loss = loss_fn(preds.transpose(1, 2), unmasked_tokenized_seqs)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_scheduler.step_and_update_lr()

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

        # STATS
        if mode == "test":
            token_to_aa = {i:aa for i, aa in enumerate('ACDEFGHIKLMNPQRSTUVWXY')}                
            # Create a list of keys from masked_loactions in format "expected_aa -> predicted_aa" where expected_aa != predicted_aa
            aa_keys = [f"{token_to_aa.get(token.item())}->{token_to_aa.get(pred_token.item())}" for token, pred_token in zip(unmasked_tokenized_seqs[masked_locations], predicted_tokens[masked_locations])]
            # Update the tracker as going through keys (counting occurences)
            aa_pred_counter.update((aa_key, aa_pred_counter[aa_key] + 1) for aa_key in aa_keys)  

    accuracy = (correct_predictions / total_masked) * 100
    if mode == "train": return accuracy, total_loss
    else: return accuracy, total_loss, aa_pred_counter

if __name__=='__main__':

    # Data/results directories
    data_dir = os.path.join(os.path.dirname(__file__), f'../../../data/rbd')
    results_dir = os.path.join(os.path.dirname(__file__), f'../../../results/run_results/bert_mlm-esm_init')

    # Create run directory for results
    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(results_dir, f"bert_mlm-esm_init-rbd-{date_hour_minute}")
    os.makedirs(run_dir, exist_ok = True)

    # Run setup
    n_epochs = 100
    batch_size = 64
    max_batch = -1
    num_workers = 64
    lr = 1e-5
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # Create Dataset and DataLoader
    torch.manual_seed(0)

    train_dataset = RBDDataset(os.path.join(data_dir, "spikeprot0528.clean.uniq.noX.RBD.metadata.variants_train.csv"))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=True)

    test_dataset = RBDDataset(os.path.join(data_dir, "spikeprot0528.clean.uniq.noX.RBD.metadata.variants_test.csv"))
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=True)

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

    model = ProteinLM(bert, vocab_size=27)

    # Run
    count_parameters(model)
    saved_model_pth = None
    from_checkpoint = False
    save_as = f"bert_mlm-esm_init-RBD-train_{len(train_dataset)}_test_{len(test_dataset)}"
    run_model(model, tokenizer, train_data_loader, test_data_loader, n_epochs, lr, max_batch, device, run_dir, save_as, saved_model_pth, from_checkpoint)