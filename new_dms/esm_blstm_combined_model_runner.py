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
from transformers import AutoTokenizer, EsmModel 
import seaborn as sns
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# BLSTM
class BLSTM(nn.Module):
    """ Bidirectional LSTM with two separate heads for binding and expression. """

    def __init__(self,
                 lstm_input_size,    # The number of expected features.
                 lstm_hidden_size,   # The number of features in hidden state h.
                 lstm_num_layers,    # Number of recurrent layers in LSTM.
                 lstm_bidirectional, # Bidirectional LSTM.
                 fcn_hidden_size):   # The number of features in hidden layer of FCN.
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

        # Two separate heads for binding and expression
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
        binding_pred = self.binding_head(fcn_out).squeeze(1) # (batch_size)
        expression_pred = self.expression_head(fcn_out).squeeze(1) # (batch_size)

        return binding_pred, expression_pred

# ESM-BLSTM
class ESM_BLSTM(nn.Module):
    def __init__(self, esm, blstm):
        super().__init__()
        self.esm = esm
        self.blstm = blstm

    def forward(self, tokenized_seqs):
        with torch.set_grad_enabled(self.training):  # Enable gradients, managed by model.eval() or model.train() in epoch_iteration
            esm_output = self.esm(**tokenized_seqs).last_hidden_state  
            binding_preds, expression_preds = self.blstm(esm_output)
        return binding_preds, expression_preds

# DATASET    
class DMSDataset(Dataset):
    """ Binding or Expression DMS Dataset, not from pickle! """
    
    def __init__(self, csv_file:str):
        """
        Load from csv file into pandas:
        - sequence label ('labels'), 
        - 'sequence',
        - binding target,
        - expression target
        """
        try:
            self.full_df = pd.read_csv(csv_file, sep=',', header=0)
        except (FileNotFoundError, pd.errors.ParserError, Exception) as e:
            print(f"Error reading in .csv file: {csv_file}\n{e}", file=sys.stderr)
            sys.exit(1)

    def __len__(self) -> int:
        return len(self.full_df)

    def __getitem__(self, idx):
        # label, seq, binding target, expression target
        return self.full_df['label'][idx], self.full_df['sequence'][idx], self.full_df['ACE2-binding_affinity'][idx], self.full_df['RBD_expression'][idx]

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

def save_model(model, optimizer, path_to_pth, epoch, loss):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }, path_to_pth
    )
    print(f"Model and optimizer state saved to {path_to_pth}")

def load_model(model, optimizer, saved_model_pth, device, continue_from_checkpoint):
    saved_state = torch.load(saved_model_pth, map_location=device)
    model.load_state_dict(saved_state['model_state_dict'])
    epoch = saved_state['epoch']
    loss = saved_state['loss']
    print(f"Loaded model from {saved_model_pth} at epoch {epoch}, loss {loss}")

    if continue_from_checkpoint:
        optimizer.load_state_dict(saved_state['optimizer_state_dict'])
        return model.to(device), optimizer, epoch, loss
    else:
        return model.to(device)

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

    fontsize = 28
    plt.subplots(figsize=(16, 9))
    plt.plot(df['Epoch'], df['Test RMSE'], label='Test RMSE', color='tab:blue', linewidth=3)
    plt.plot(df['Epoch'], df['Train RMSE'], label='Train RMSE', color='tab:orange', linewidth=3)
    plt.xlabel('Epochs', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    #plt.xlim(-250, 5250)
    plt.ylabel('Loss', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    #plt.ylim(0.05, 0.79)
    plt.legend(loc='upper right', fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(metrics_img, format='pdf')

def plot_all_log_file(metrics_csv, metrics_img):
    df = pd.read_csv(metrics_csv)

    sns.set_theme(style="darkgrid")

    fontsize = 28
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 18))  # 2 rows, 1 column

    # Plot 1: Training Metrics (binding, expression, combined RMSE)
    ax1.plot(df['Epoch'], df['Train Binding RMSE'], label='Train Binding RMSE', color='tab:blue', linewidth=3)
    ax1.plot(df['Epoch'], df['Train Expression RMSE'], label='Train Expression RMSE', color='tab:green', linewidth=3)
    ax1.plot(df['Epoch'], df['Train RMSE'], label='Train RMSE (Combined)', color='tab:orange', linewidth=3)
    ax1.set_xlabel('Epochs', fontsize=fontsize)
    ax1.set_xticks(range(0, len(df['Epoch']), max(1, len(df['Epoch']) // 10)))
    ax1.tick_params(axis='x', labelsize=fontsize)
    #ax1.set_xlim(-250, 5250)
    ax1.set_ylabel('Loss', fontsize=fontsize)
    ax1.tick_params(axis='y', labelsize=fontsize)
    #ax1.set_ylim(0.05, 0.79)
    ax1.legend(loc='upper right', fontsize=fontsize)
    ax1.set_title('Training Metrics', fontsize=fontsize)

    # Plot 2: Testing Metrics (binding, expression, combined RMSE)
    ax2.plot(df['Epoch'], df['Test Binding RMSE'], label='Test Binding RMSE', color='tab:blue', linewidth=3)
    ax2.plot(df['Epoch'], df['Test Expression RMSE'], label='Test Expression RMSE', color='tab:green', linewidth=3)
    ax2.plot(df['Epoch'], df['Test RMSE'], label='Test RMSE (Combined)', color='tab:orange', linewidth=3)
    ax2.set_xlabel('Epochs', fontsize=fontsize)
    ax2.set_xticks(range(0, len(df['Epoch']), max(1, len(df['Epoch']) // 10)))
    ax2.tick_params(axis='x', labelsize=fontsize)
    #ax2.set_xlim(-250, 5250)
    ax2.set_ylabel('Loss', fontsize=fontsize)
    ax2.tick_params(axis='y', labelsize=fontsize)
    #ax2.set_ylim(0.05, 0.79)
    ax2.legend(loc='upper right', fontsize=fontsize)
    ax2.set_title('Testing Metrics', fontsize=fontsize)

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(metrics_img, format='pdf')
    plt.close()

# MODEL RUNNING
def run_model(model, tokenizer, train_data_loader, test_data_loader, n_epochs: int, lr:float, max_batch: Union[int, None], device: str, run_dir: str, save_as: str, saved_model_pth:str=None, continue_from_checkpoint:bool=False):
    """ Run a model through train and test epochs. """

    model = model.to(device)
    loss_fn = nn.MSELoss(reduction='sum').to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr)

    starting_epoch = 1
    best_loss = float('inf')

    # Load saved model if provided
    if saved_model_pth is not None:
        if continue_from_checkpoint:
            model, optimizer, starting_epoch, best_loss = load_model(model, optimizer, saved_model_pth, device, continue_from_checkpoint)
            starting_epoch += 1
        else:
            model = load_model(model, optimizer, saved_model_pth, device, continue_from_checkpoint)

    start_time = time.time()
    
    metrics_csv = os.path.join(run_dir, f"{save_as}_metrics.csv")
    metrics_img = os.path.join(run_dir, f"{save_as}_metrics.pdf")
    all_metrics_img = os.path.join(run_dir, f"{save_as}_all_metrics.pdf")

    if starting_epoch > n_epochs:
        raise ValueError(f"Starting epoch ({starting_epoch}) is greater than the total number of epochs to run ({n_epochs}). Adjust the number of epochs, 'n_epochs'.")
    
    with open(metrics_csv, "a") as fa:
        if continue_from_checkpoint: load_model_checkpoint(saved_model_pth, metrics_csv, starting_epoch)
        else:
            fa.write((
                "Epoch,"
                "Train Binding MSE,Train Binding RMSE,Train Expression MSE,Train Expression RMSE,Train MSE,Train RMSE,"
                "Test Binding MSE,Test Binding RMSE,Test Expression MSE,Test Expression RMSE,Test MSE,Test RMSE\n"
            ))
            
        for epoch in range(starting_epoch, n_epochs + 1):
            if not max_batch:
                max_batch = len(train_data_loader)

            train_binding_mse, train_binding_rmse, train_expression_mse, train_expression_rmse, train_mse, train_rmse = epoch_iteration(model, tokenizer, loss_fn, optimizer, train_data_loader, epoch, max_batch, device, mode='train')
            test_binding_mse, test_binding_rmse, test_expression_mse, test_expression_rmse, test_mse, test_rmse = epoch_iteration(model, tokenizer, loss_fn, optimizer, test_data_loader, epoch, max_batch, device, mode='test')

            print(f'Epoch {epoch} | Train Binding MSE: {train_binding_mse:.4f}, Train Expression MSE: {train_expression_mse:.4f}, Train MSE: {train_mse:.4f}') 
            print(f'{" "*(8+len(str(epoch)))} Train Binding RMSE: {train_binding_rmse:.4f}, Train Expression RMSE: {train_expression_rmse:.4f}, Train RMSE: {train_rmse:.4f}') 
            print(f'{" "*(8+len(str(epoch)))} Test Binding MSE: {test_binding_mse:.4f}, Test Expression MSE: {test_expression_mse:.4f}, Test MSE: {test_mse:.4f}') 
            print(f'{" "*(8+len(str(epoch)))} Test Binding RMSE: {test_binding_rmse:.4f}, Test Expression RMSE: {test_expression_rmse:.4f}, Test RMSE: {test_rmse:.4f}') 
            
            fa.write((
                f"{epoch}," 
                f"{train_binding_mse}, {train_binding_rmse}, {train_expression_mse}, {train_expression_rmse}, {train_mse}, {train_rmse},"
                f"{test_binding_mse}, {test_binding_rmse}, {test_expression_mse}, {test_expression_rmse}, {test_mse}, {test_rmse}\n"
            ))                
            fa.flush()

            if test_rmse < best_loss:
                best_loss = test_rmse
                model_path = os.path.join(run_dir, f'best_saved_model.pth')
                print(f"NEW BEST model: loss {best_loss:.4f}")
                save_model(model, optimizer, model_path, epoch, best_loss)
            
            model_path = os.path.join(run_dir, f'checkpoint_saved_model.pth')
            save_model(model, optimizer, model_path, epoch, test_rmse)
            print("")
    
    plot_log_file(metrics_csv, metrics_img)
    plot_all_log_file(metrics_csv, all_metrics_img)

    # End timer and print duration
    end_time = time.time()
    duration = end_time - start_time
    print(f'Training and testing complete in {duration:.2f} seconds.')

def epoch_iteration(model, tokenizer, loss_fn, optimizer, data_loader, epoch, max_batch, device, mode):
    """ Used in run_model. """
    
    model.train() if mode=='train' else model.eval()

    data_iter = tqdm.tqdm(enumerate(data_loader),
                          desc=f'Epoch_{mode}: {epoch}',
                          total=len(data_loader),
                          bar_format='{l_bar}{r_bar}')

    total_binding_loss, total_expression_loss, total_loss = 0, 0, 0
    total_items = 0

    for batch, batch_data in data_iter:
        if max_batch > 0 and batch >= max_batch:
            break

        _, seqs, binding_targets, expression_targets = batch_data
        binding_targets, expression_targets = binding_targets.to(device).float(), expression_targets.to(device).float()
        tokenized_seqs = tokenizer(seqs,return_tensors="pt").to(device)
   
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
    result_tag = 'combined_OLD_DMS_DATA_WEIGHTS_NEW_DMS_DATA_SHORT'
    old_data_dir = os.path.join(os.path.dirname(__file__), f'../results/run_results/esm-blstm')
    data_dir = os.path.join(os.path.dirname(__file__), f'./data/split_processed_dms')
    results_dir = os.path.join(os.path.dirname(__file__), f'./run_results/esm-blstm')

    # Create run directory for results
    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(results_dir, f"esm_blstm-dms_{result_tag}-{date_hour_minute}")
    os.makedirs(run_dir, exist_ok = True)

    # Run setup
    n_epochs = 100
    batch_size = 32
    max_batch = -1
    num_workers = 64
    lr = 1e-5
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # Create Dataset and DataLoader
    torch.manual_seed(0)
    dms_train_csv = os.path.join(data_dir, 'mutation_combined_NEW-DMS_train.csv') 
    dms_test_csv = os.path.join(data_dir, 'mutation_combined_NEW-DMS_test.csv') 

    train_dataset = DMSDataset(dms_train_csv)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=True)

    test_dataset = DMSDataset(dms_test_csv)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=True)

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
    saved_model_pth = None
    continue_from_checkpoint = False     
    save_as = f"esm_blstm-dms_{result_tag}-train_{len(train_dataset)}_test_{len(test_dataset)}"
    run_model(model, tokenizer, train_data_loader, test_data_loader, n_epochs, lr, max_batch, device, run_dir, save_as, saved_model_pth, continue_from_checkpoint)
