#!/usr/bin/env python
"""
Model runner for ESM-FCN model (single target).
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
    DMSDataset,
    count_parameters,
    save_model,
    load_model,
    load_model_checkpoint,
    plot_log_file,
)

class FCN(nn.Module):
    """ Fully Connected Network """

    def __init__(self,
                 fcn_input_size,    # The number of input features
                 fcn_hidden_size,   # The number of features in hidden layer of FCN.
                 fcn_num_layers):   # The number of fcn layers  
        super().__init__()

        # Creating a list of layers for the FCN
        # Subsequent layers after 1st should be equal to hidden_size for input_size
        layers = []
        input_size = fcn_input_size

        for _ in range(fcn_num_layers):
            layers.append(nn.Linear(input_size, fcn_hidden_size))
            layers.append(nn.ReLU())
            input_size = fcn_hidden_size

        # FCN layers
        self.fcn = nn.Sequential(*layers)

        # FCN output layer 
        self.out = nn.Linear(fcn_hidden_size, 1)

    def forward(self, x):
        fcn_out = self.fcn(x)
        prediction = self.out(fcn_out)  # [batch_size, 1]

        return prediction

# ESM-FCN
class ESM_FCN(nn.Module):
    def __init__(self, esm, fcn):
        super().__init__()
        self.esm = esm
        self.fcn = fcn

    def forward(self, tokenized_seqs):
        with torch.set_grad_enabled(self.training):  # Enable gradients, managed by model.eval() or model.train() in epoch_iteration
            esm_last_hidden_state = self.esm(**tokenized_seqs).last_hidden_state # shape: [batch_size, sequence_length, embedding_dim]
            esm_cls_embedding = esm_last_hidden_state[:, 0, :]  # CLS token embedding (sequence-level representations), [batch_size, embedding_dim]
            output = self.fcn(esm_cls_embedding).squeeze(1) # [batch_size]
        return output

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
        else: fa.write(f"Epoch,Train MSE,Train RMSE,Test MSE,Test RMSE\n")

    # Running
    start_time = time.time()

    for epoch in range(starting_epoch, n_epochs + 1):
        train_mse, train_rmse = epoch_iteration(model, tokenizer, loss_fn, optimizer, train_data_loader, epoch, max_batch, device, mode='train')
        test_mse, test_rmse = epoch_iteration(model, tokenizer, loss_fn, optimizer, test_data_loader, epoch, max_batch, device, mode='test')

        print(f'Epoch {epoch} | Test MSE Loss: {test_mse:.4f}, Test RMSE Loss: {test_rmse:.4f}\n')  

        with open(metrics_csv, "a") as fa:        
            fa.write(f"{epoch},{train_mse},{train_rmse},{test_mse},{test_rmse}\n")
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
        
    plot_log_file(metrics_csv, metrics_img)

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
            optimizer.zero_grad()
            preds = model(tokenized_seqs)
            batch_loss = loss_fn(preds, targets)
            batch_loss.backward()
            optimizer.step()

        else:
            with torch.no_grad():
                preds = model(tokenized_seqs)
                batch_loss = loss_fn(preds, targets)

        total_loss += batch_loss.item()
        total_items += targets.size(0)
    
    # total loss is the sum of squared errors over items encountered
    # so divide by the number of items encountered
    # we get mse and rmse per item
    mse = total_loss/total_items
    rmse = np.sqrt(mse)

    return mse, rmse 

if __name__=='__main__':

    # Data/results directories
    result_tag = 'binding' # specify expression or binding
    data_dir = os.path.join(os.path.dirname(__file__), f'../../../data/dms') 
    results_dir = os.path.join(os.path.dirname(__file__), f'../../../results/run_results/esm_fcn')

    # Create run directory for results
    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(results_dir, f"esm_fcn-DMS_OLD-{result_tag}-{date_hour_minute}")
    os.makedirs(run_dir, exist_ok = True)

    # Run setup
    n_epochs = 5000
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

    # FCN input
    size = 320
    fcn_input_size = size  
    fcn_hidden_size = size
    fcn_num_layers = 1
    fcn = FCN(fcn_input_size, fcn_hidden_size, fcn_num_layers)

    model = ESM_FCN(esm, fcn)

    # Run
    count_parameters(model)
    saved_model_pth = None
    from_checkpoint = False
    save_as = f"esm_fcn-DMS_OLD_{result_tag}-train_{len(train_dataset)}_test_{len(test_dataset)}"
    run_model(model, tokenizer, train_data_loader, test_data_loader, n_epochs, lr, max_batch, device, run_dir, save_as, saved_model_pth, from_checkpoint)#!/usr/bin/env python
"""
Model runner for ESM-FCN model (single target).
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
    DMSDataset,
    count_parameters,
    save_model,
    load_model,
    load_model_checkpoint,
    plot_log_file,
)

class FCN(nn.Module):
    """ Fully Connected Network """

    def __init__(self,
                 fcn_input_size,    # The number of input features
                 fcn_hidden_size,   # The number of features in hidden layer of FCN.
                 fcn_num_layers):   # The number of fcn layers  
        super().__init__()

        # Creating a list of layers for the FCN
        # Subsequent layers after 1st should be equal to hidden_size for input_size
        layers = []
        input_size = fcn_input_size

        for _ in range(fcn_num_layers):
            layers.append(nn.Linear(input_size, fcn_hidden_size))
            layers.append(nn.ReLU())
            input_size = fcn_hidden_size

        # FCN layers
        self.fcn = nn.Sequential(*layers)

        # FCN output layer 
        self.out = nn.Linear(fcn_hidden_size, 1)

    def forward(self, x):
        fcn_out = self.fcn(x)
        prediction = self.out(fcn_out)  # [batch_size, 1]

        return prediction

# ESM-FCN
class ESM_FCN(nn.Module):
    def __init__(self, esm, fcn):
        super().__init__()
        self.esm = esm
        self.fcn = fcn

    def forward(self, tokenized_seqs):
        with torch.set_grad_enabled(self.training):  # Enable gradients, managed by model.eval() or model.train() in epoch_iteration
            esm_last_hidden_state = self.esm(**tokenized_seqs).last_hidden_state # shape: [batch_size, sequence_length, embedding_dim]
            esm_cls_embedding = esm_last_hidden_state[:, 0, :]  # CLS token embedding (sequence-level representations), [batch_size, embedding_dim]
            output = self.fcn(esm_cls_embedding).squeeze(1) # [batch_size]
        return output

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
        else: fa.write(f"Epoch,Train MSE,Train RMSE,Test MSE,Test RMSE\n")

    # Running
    start_time = time.time()

    for epoch in range(starting_epoch, n_epochs + 1):
        train_mse, train_rmse = epoch_iteration(model, tokenizer, loss_fn, optimizer, train_data_loader, epoch, max_batch, device, mode='train')
        test_mse, test_rmse = epoch_iteration(model, tokenizer, loss_fn, optimizer, test_data_loader, epoch, max_batch, device, mode='test')

        print(f'Epoch {epoch} | Test MSE Loss: {test_mse:.4f}, Test RMSE Loss: {test_rmse:.4f}\n')  

        with open(metrics_csv, "a") as fa:        
            fa.write(f"{epoch},{train_mse},{train_rmse},{test_mse},{test_rmse}\n")
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
        
    plot_log_file(metrics_csv, metrics_img)

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
            optimizer.zero_grad()
            preds = model(tokenized_seqs)
            batch_loss = loss_fn(preds, targets)
            batch_loss.backward()
            optimizer.step()

        else:
            with torch.no_grad():
                preds = model(tokenized_seqs)
                batch_loss = loss_fn(preds, targets)

        total_loss += batch_loss.item()
        total_items += targets.size(0)
    
    # total loss is the sum of squared errors over items encountered
    # so divide by the number of items encountered
    # we get mse and rmse per item
    mse = total_loss/total_items
    rmse = np.sqrt(mse)

    return mse, rmse 

if __name__=='__main__':

    # Data/results directories
    result_tag = 'binding' # specify expression or binding
    data_dir = os.path.join(os.path.dirname(__file__), f'../../../data/dms') 
    results_dir = os.path.join(os.path.dirname(__file__), f'../../../results/run_results/esm_fcn')

    # Create run directory for results
    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(results_dir, f"esm_fcn-DMS_OLD-{result_tag}-{date_hour_minute}")
    os.makedirs(run_dir, exist_ok = True)

    # Run setup
    n_epochs = 5000
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

    # FCN input
    size = 320
    fcn_input_size = size  
    fcn_hidden_size = size
    fcn_num_layers = 5
    fcn = FCN(fcn_input_size, fcn_hidden_size, fcn_num_layers)

    model = ESM_FCN(esm, fcn)

    # Run
    count_parameters(model)
    saved_model_pth = None
    from_checkpoint = False
    save_as = f"esm_fcn-DMS_OLD_{result_tag}-train_{len(train_dataset)}_test_{len(test_dataset)}"
    run_model(model, tokenizer, train_data_loader, test_data_loader, n_epochs, lr, max_batch, device, run_dir, save_as, saved_model_pth, from_checkpoint)