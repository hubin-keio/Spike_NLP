#!/usr/bin/env python
"""
Model runner for bert_blstm.py

TODO: 
- Move plotting functions to another file for cleanup
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
from runner_util import save_model, count_parameters
from transformers import AutoTokenizer, EsmModel 
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

def generate_esm_embeddings(batch_sequences, esm, esm_tokenizer, device):
    tokenized_batch = esm_tokenizer(batch_sequences, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = esm(**tokenized_batch)
        embeddings = outputs.last_hidden_state
    # embeddings shape: (batch_size, seq_len, embedding_dim)
    return embeddings  

def run_model(model, tokenizer, esm, esm_tokenizer, train_set, test_set, n_epochs: int, batch_size: int, max_batch: Union[int, None], alpha:float, lr:float, device: str, save_as: str):
    """ Run a model through train and test epochs"""
    
    if not max_batch:
        max_batch = len(train_set)

    model = model.to(device)
    regression_loss_fn = nn.MSELoss(reduction='sum').to(device) # blstm
    masked_language_loss_fn = nn.CrossEntropyLoss(reduction='sum').to(device) # mlm
    optimizer = torch.optim.SGD(model.parameters(), lr)

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size, shuffle=True)

    metrics_csv = save_as + "_metrics.csv"

    with open(metrics_csv, "w") as fh:
        fh.write("epoch,train_blstm_loss,test_blstm_loss\n")
        for epoch in range(1, n_epochs + 1):
            train_mlm_accuracy, train_mlm_loss, train_blstm_loss, train_combined_loss = epoch_iteration(model, tokenizer, esm, esm_tokenizer, regression_loss_fn, masked_language_loss_fn, optimizer, train_loader, epoch, max_batch, alpha, device, mode='train')
            test_mlm_accuracy, test_mlm_loss, test_blstm_loss, test_combined_loss = epoch_iteration(model, tokenizer, esm, esm_tokenizer, regression_loss_fn, masked_language_loss_fn, optimizer, test_loader, epoch, max_batch, alpha, device, mode='test')

            print(f'Epoch {epoch} | Train MLM Acc: {train_mlm_accuracy:.4f}, Test MLM Acc: {test_mlm_accuracy:.4f}\n'
                  f'{" "*(len(str(epoch))+7)}| Train MLM Loss: {train_mlm_loss:.4f}, Test MLM Loss: {test_mlm_loss:.4f}\n'
                  f'{" "*(len(str(epoch))+7)}| Train BLSTM Loss: {train_blstm_loss:.4f}, Test BLSTM Loss: {test_blstm_loss:.4f}\n'
                  f'{" "*(len(str(epoch))+7)}| Train Combined Loss: {train_combined_loss:.4f}, Test Combined Loss: {test_combined_loss:.4f}\n')
            
            fh.write(f"{epoch},{train_mlm_accuracy},{test_mlm_accuracy},{train_mlm_loss},{test_mlm_loss},{train_blstm_loss},{test_blstm_loss},{train_combined_loss},{test_combined_loss}\n")
            fh.flush()
            
            save_model(model, optimizer, epoch, save_as + '.model_save')    

    return metrics_csv

def epoch_iteration(model, tokenizer, esm, esm_tokenizer, regression_loss_fn, masked_language_loss_fn, optimizer, data_loader, num_epochs: int, max_batch: int, alpha:float, device: str, mode: str):
    """ Used in run_model """
    
    model.train() if mode=='train' else model.eval()

    data_iter = tqdm.tqdm(enumerate(data_loader),
                          desc=f'Epoch_{mode}: {num_epochs}',
                          total=len(data_loader),
                          bar_format='{l_bar}{r_bar}')

    total_mlm_loss = 0
    total_blstm_loss = 0
    total_combined_loss = 0
    total_masked = 0
    correct_predictions = 0

    for batch, batch_data in data_iter:
        if max_batch > 0 and batch >= max_batch:
            break

        labels, seqs, targets = batch_data
        masked_tokenized_seqs = tokenizer(seqs).to(device)

        print(masked_tokenized_seqs)
        exit()

        unmasked_tokenized_seqs = tokenizer._batch_pad(seqs).to(device)
        target = targets.to(device).float()

        if mode == 'train':
            optimizer.zero_grad()
            mlm_pred, blstm_pred = model(masked_tokenized_seqs)
            batch_mlm_loss = masked_language_loss_fn(mlm_pred.transpose(1, 2), unmasked_tokenized_seqs)
            batch_blstm_loss = regression_loss_fn(blstm_pred.flatten(), target)
            combined_loss = batch_mlm_loss + (batch_blstm_loss * alpha)
            combined_loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                mlm_pred, blstm_pred = model(masked_tokenized_seqs)
                batch_mlm_loss = masked_language_loss_fn(mlm_pred.transpose(1, 2), unmasked_tokenized_seqs)
                batch_blstm_loss = regression_loss_fn(blstm_pred.flatten(), target)
                combined_loss = batch_mlm_loss + (batch_blstm_loss * alpha)

        # Loss
        total_mlm_loss += batch_mlm_loss.item()
        total_blstm_loss += batch_blstm_loss.item()
        total_combined_loss += combined_loss.item()

        # Accuracy
        predicted_tokens = torch.max(mlm_pred, dim=-1)[1]
        masked_locations = torch.nonzero(torch.eq(masked_tokenized_seqs, token_to_index['<MASK>']), as_tuple=True)
        correct_predictions += torch.eq(predicted_tokens[masked_locations], unmasked_tokenized_seqs[masked_locations]).sum().item()
        total_masked += masked_locations[0].numel()

    mlm_accuracy = correct_predictions / total_masked
    return mlm_accuracy, total_mlm_loss, total_blstm_loss, total_combined_loss

def calc_train_test_history(metrics: dict, n_train: int, n_test: int, save_as: str):
    """ Calculate the average mse per item and rmse """

    history_df = pd.DataFrame(metrics)

    history_df['train_blstm_loss_per'] = history_df['train_blstm_loss']/n_train  # average mse per item
    history_df['test_blstm_loss_per'] = history_df['test_blstm_loss']/n_test

    history_df['train_blstm_rmse_per'] = np.sqrt(history_df['train_blstm_loss_per'].values)  # rmse
    history_df['test_blstm_rmse_per'] = np.sqrt(history_df['test_blstm_loss_per'].values)

    history_df.to_csv(save_as+'_metrics_per.csv', index=False)
    plot_mlm_history(history_df, save_as)
    plot_rmse_history(history_df, save_as)
    plot_combined_history(history_df, save_as)

def plot_combined_history(history_df: str, save_as):
    '''
    Generate a single figure with subplots for combined training loss
    from the model run csv file.
    '''
    sns.set_theme()
    sns.set_context('talk')
    sns.set(style="darkgrid")
    plt.ion()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

    # Plot Training Loss
    train_loss_line = ax.plot(history_df['epoch'], history_df['train_combined_loss'], color='tab:orange', label='Train Loss')
    test_loss_line = ax.plot(history_df['epoch'], history_df['test_combined_loss'],color='tab:blue', label='Test Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')

    # Skipping every other y-axis tick mark
    yticks = ax.get_yticks()
    ax.set_yticks(yticks[::2])  # Keep every other tick

    plt.style.use('ggplot')
    plt.tight_layout()
    plt.savefig(save_as+'_combined_loss.png', format='png')
    plt.savefig(save_as+'_combined_loss.pdf', format='pdf')

def plot_mlm_history(history_df: str, save_as):
    '''
    Generate a single figure with subplots for training loss and training accuracy
    from the model run csv file.
    '''
    sns.set_theme()
    sns.set_context('talk')
    sns.set(style="darkgrid")
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

    # Plot Training Loss
    train_loss_line = ax1.plot(history_df['epoch'], history_df['train_mlm_loss'], color='tab:red', label='Train Loss')
    test_loss_line = ax1.plot(history_df['epoch'], history_df['test_mlm_loss'],color='tab:orange', label='Test Loss')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')

    # Plot Training Accuracy
    train_accuracy_line = ax2.plot(history_df['epoch'], history_df['train_mlm_accuracy'], color='tab:blue', label='Train Accuracy')
    test_accuracy_line = ax2.plot(history_df['epoch'], history_df['test_mlm_accuracy'], color='tab:green', label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1) 
    ax2.legend(loc='upper right')

    # Skipping every other y-axis tick mark
    a1_yticks = ax1.get_yticks()
    ax1.set_yticks(a1_yticks[::2])  # Keep every other tick

    plt.style.use('ggplot')
    plt.tight_layout()
    plt.savefig(save_as+'_loss_acc.png', format='png')
    plt.savefig(save_as+'_loss_acc.pdf', format='pdf')

def plot_rmse_history(history_df, save_as: str):
    """ Plot RMSE training and testing history per epoch. """
    
    sns.set_theme()
    sns.set_context('talk')
    sns.set(style="darkgrid")
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.lineplot(data=history_df, x=history_df.index, y='train_blstm_rmse_per', label='Train RMSE', color='tab:orange', ax=ax)
    sns.lineplot(data=history_df, x=history_df.index, y='test_blstm_rmse_per', label='Test RMSE', color='tab:blue', ax=ax)
    
    # Skipping every other y-axis tick mark
    ax_yticks = ax.get_yticks()
    ax.set_yticks(ax_yticks[::2])  # Keep every other tick

    ax.set(xlabel='Epoch', ylabel='Average RMSE Per Sample')
    plt.style.use('ggplot')
    plt.tight_layout()
    plt.savefig(save_as + '_rmse.png', format='png')
    plt.savefig(save_as + '_rmse.pdf', format='pdf') 

if __name__=='__main__':

    # Data/results directories
    result_tag = 'bert_blstm_esm-dms_expression'
    data_dir = os.path.join(os.path.dirname(__file__), f'../../../data')
    results_dir = os.path.join(os.path.dirname(__file__), f'../../../results/run_results/bert_blstm_esm')
    
    # Create run directory for results
    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(results_dir, f"{result_tag}-{date_hour_minute}")
    os.makedirs(run_dir, exist_ok = True)

    # Load in data
    dms_train_csv = os.path.join(data_dir, 'dms_mutation_expression_meanFs_train.csv') 
    dms_test_csv = os.path.join(data_dir, 'dms_mutation_expression_meanFs_test.csv')
    train_dataset = DMSDataset(dms_train_csv)
    test_dataset = DMSDataset(dms_test_csv)

    # ESM input
    esm = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")

    # Can use esm_embeddings as a dictionary based on esm token_id
    esm_embeddings = esm.embeddings.word_embeddings.weight
    amino_acid = list('ACDEFGHIKLMNPQRSTUVWXY')
    esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    token_id = esm_tokenizer.convert_tokens_to_ids(amino_acid)
    print(token_id)
    amino_acid_embedding = esm_embeddings[token_id]
    #print(amino_acid_embedding.shape)

    # maybe tokenize & autotokenize, feed tokenized through bert, 
    # feed autotokenized through to convert and get the esm per token
    # replace the bert with esm embeddings per token


   

    # # Load pretrained spike model weights for BERT
    # model_pth = os.path.join(results_dir, 'ddp_runner/ddp-2023-10-06_20-16/ddp-2023-10-06_20-16_best_model_weights.pth') # 320 dim
    # saved_state = torch.load(model_pth, map_location='cuda')
    # state_dict = saved_state['model_state_dict']

    # For loading from ddp models, they have 'module.bert.' or 'module.mlm.' in keys of state_dict
    # Also need separated out for each corresponding model part
    # bert_state_dict = {key[len('module.bert.'):]: value for key, value in state_dict.items() if key.startswith('module.bert.')}
    # print(bert_state_dict)
    
    # exit()
    # mlm_state_dict = {key[len('module.mlm.'):]: value for key, value in state_dict.items() if key.startswith('module.mlm.')}

    # BERT input
    max_len = 280
    mask_prob = 0.15
    embedding_dim = 320 
    dropout = 0.1
    n_transformer_layers = 12
    n_attn_heads = 10
    tokenizer = ProteinTokenizer(max_len, mask_prob)
    bert = BERT(embedding_dim, dropout, max_len, mask_prob, n_transformer_layers, n_attn_heads)
    print(tokenizer(amino_acid))
    
    #exit()
    #bert.load_state_dict(bert_state_dict)

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
    #model.mlm.load_state_dict(mlm_state_dict)

    # Run setup
    n_epochs = 1
    batch_size = 32
    max_batch = 1
    alpha = 1
    lr = 1e-5
    device = torch.device("cuda:2")

    #count_parameters(model)
    model_result = os.path.join(run_dir, f"{result_tag}-{date_hour_minute}_train_{len(train_dataset)}_test_{len(test_dataset)}")
    metrics  = run_model(model, tokenizer, esm, esm_tokenizer, train_dataset, test_dataset, n_epochs, batch_size, max_batch, alpha, lr, device, model_result)
    calc_train_test_history(metrics, len(train_dataset), len(test_dataset), model_result)
