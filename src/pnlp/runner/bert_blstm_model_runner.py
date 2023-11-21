#!/usr/bin/env python
"""
Model runner for bert_blstm.py

TODO: 
- update embedding weights - split between bert and mlm
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
from model_util import save_model, count_parameters
from pnlp.embedding.tokenizer import ProteinTokenizer, token_to_index
from pnlp.model.language import BERT

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.blstm import BLSTM
from model.bert_blstm import BERT_BLSTM

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

def run_model(model, tokenizer, train_set, test_set, n_epochs: int, batch_size: int, max_batch: Union[int, None], alpha:float, lr:float, device: str, save_as: str):
    """ Run a model through train and test epochs"""
    
    if not max_batch:
        max_batch = len(train_set)

    model = model.to(device)
    regression_loss_fn = nn.MSELoss(reduction='sum').to(device) # blstm
    masked_language_loss_fn = nn.CrossEntropyLoss(reduction='sum').to(device) # mlm
    optimizer = torch.optim.SGD(model.parameters(), lr)

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size, shuffle=True)

    metrics = defaultdict(list)

    for epoch in range(1, n_epochs + 1):
        train_loss = epoch_iteration(model, tokenizer, regression_loss_fn, masked_language_loss_fn, optimizer, train_loader, epoch, max_batch, alpha, device, mode='train')
        test_loss = epoch_iteration(model, tokenizer, regression_loss_fn, masked_language_loss_fn, optimizer, test_loader, epoch, max_batch, alpha, device, mode='test')

        keys = ['train_loss','test_loss'] # to add more metrics, add more keys
        for key in keys:
            metrics[key].append(locals()[key])

        print(f'Epoch {epoch} | Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        save_model(model, optimizer, epoch, save_as + '.model_save')

    return metrics

def epoch_iteration(model, tokenizer, regression_loss_fn, masked_language_loss_fn, optimizer, data_loader, num_epochs: int, max_batch: int, alpha:float, device: str, mode: str):
    """ Used in run_model """
    
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
        masked_tokenized_seqs = tokenizer(seqs).to(device)
        unmasked_tokenized_seqs = tokenizer._batch_pad(seqs).to(device)
        target = targets.to(device).float()

        if mode == 'train':
            optimizer.zero_grad()
            mlm_pred, blstm_pred = model(masked_tokenized_seqs)
            batch_mlm_loss = masked_language_loss_fn(mlm_pred.transpose(1, 2), unmasked_tokenized_seqs)
            batch_blstm_loss = regression_loss_fn(blstm_pred.flatten(), target)
            combined_loss = batch_mlm_loss + (alpha * batch_blstm_loss)
            combined_loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                mlm_pred, blstm_pred = model(masked_tokenized_seqs)
                batch_mlm_loss = masked_language_loss_fn(mlm_pred.transpose(1, 2), unmasked_tokenized_seqs)
                batch_blstm_loss = regression_loss_fn(blstm_pred.flatten(), target)
                combined_loss = batch_mlm_loss + (alpha * batch_blstm_loss)

        total_loss += combined_loss.item()

    return total_loss

def calc_train_test_history(metrics: dict, n_train: int, n_test: int, save_as: str):
    """ Calculate the average mse per item and rmse """

    history_df = pd.DataFrame(metrics)
    history_df['train_loss'] = history_df['train_loss']/n_train  # average mse per item
    history_df['test_loss'] = history_df['test_loss']/n_test

    history_df['train_rmse'] = np.sqrt(history_df['train_loss'].values)  # rmse
    history_df['test_rmse'] = np.sqrt(history_df['test_loss'].values)

    history_df.to_csv(save_as+'.csv', index=False)
    plot_rmse_history(history_df, save_as)

def plot_rmse_history(history_df, save_as: str):
    """ Plot RMSE training and testing history per epoch. """
    
    sns.set_theme()
    sns.set_context('talk')
    palette = sns.color_palette()
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(data=history_df, x=history_df.index, y='test_rmse', label='Testing', color=palette[0], ax=ax)
    sns.lineplot(data=history_df, x=history_df.index, y='train_rmse', label='Training', color=palette[1], ax=ax)
    
    # Skipping every other x-axis tick mark
    xticks = ax.get_xticks()
    ax.set_xticks(xticks[::2])  # Keep every other tick

    # Skipping every other y-axis tick mark
    yticks = ax.get_yticks()
    ax.set_yticks(yticks[::2])  # Keep every other tick
    
    ax.set(xlabel='Epochs', ylabel='Average RMSE per sample')
    plt.tight_layout()
    plt.savefig(save_as + '-rmse.png')
 
if __name__=='__main__':

    # Data/results directories
    dataset = 'dms/binding' # specify
    data_dir = os.path.join(os.path.dirname(__file__), f'../../../data/{dataset}')
    results_dir = os.path.join(os.path.dirname(__file__), f'../../../results')
    
    # Create run directory for results
    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(results_dir, f"bert_blstm/{dataset}/bert_blstm-{date_hour_minute}")
    os.makedirs(run_dir, exist_ok = True)

    # Load in data
    dms_train_csv = os.path.join(data_dir, 'mutation_binding_Kds_train.csv') 
    dms_test_csv = os.path.join(data_dir, 'mutation_binding_Kds_test.csv')
    train_dataset = DMSDataset(dms_train_csv)
    test_dataset = DMSDataset(dms_test_csv)

    # Load pretrained spike model weights for BERT
    model_pth = os.path.join(results_dir, 'ddp_runner/ddp-2023-10-06_20-16/ddp-2023-10-06_20-16_best_model_weights.pth') # 320 dim
    saved_state = torch.load(model_pth, map_location='cuda')
    state_dict = saved_state['model_state_dict']

    # For loading from ddp models, they have 'module.bert.' or 'module.mlm.' in keys of state_dict
    # Also need separated out for each corresponding model part
    bert_state_dict = {key[len('module.bert.'):]: value for key, value in state_dict.items() if key.startswith('module.bert.')}
    mlm_state_dict = {key[len('module.mlm.'):]: value for key, value in state_dict.items() if key.startswith('module.mlm.')}

    # BERT input
    max_len = 280
    mask_prob = 0.15
    embedding_dim = 320 
    dropout = 0.1
    n_transformer_layers = 12
    n_attn_heads = 10
    tokenizer = ProteinTokenizer(max_len, mask_prob)
    bert = BERT(embedding_dim, dropout, max_len, mask_prob, n_transformer_layers, n_attn_heads)
    bert.load_state_dict(bert_state_dict)

    # BLSTM input
    lstm_input_size = 320
    lstm_hidden_size = 320
    lstm_num_layers = 1        
    lstm_bidrectional = True   
    fcn_hidden_size = 320
    blstm = BLSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers, lstm_bidrectional, fcn_hidden_size)

    # BERT_BLSTM input
    vocab_size = len(token_to_index)
    model = BERT_BLSTM(bert, blstm, vocab_size)
    model.mlm.load_state_dict(mlm_state_dict)

    # Run setup
    n_epochs = 200
    batch_size = 32
    max_batch = -1
    alpha = 1000000
    lr = 1e-5
    device = torch.device("cuda:2")

    #count_parameters(model)
    model_result = os.path.join(run_dir, f"bert_blstm-{date_hour_minute}_train_{len(train_dataset)}_test_{len(test_dataset)}")
    metrics  = run_model(model, tokenizer, train_dataset, test_dataset, n_epochs, batch_size, max_batch, alpha, lr, device, model_result)
    calc_train_test_history(metrics, len(train_dataset), len(test_dataset), model_result)
