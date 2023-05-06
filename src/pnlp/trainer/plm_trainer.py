"""Protein Language Model Trainer"""

import os
import sys
import datetime
from os import path
from typing import Union
import sqlite3
import numpy as np
import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from pnlp.db.dataset import SeqDataset, initialize_db
from pnlp.embedding.tokenizer import ProteinTokenizer
from pnlp.embedding.nlp_embedding import NLPEmbedding, token_to_index
from pnlp.model.language import ProteinLM
from pnlp.model.bert import BERT


class ScheduledOptim():
    """A simple wrapper class for learning rate scheduling."""

    def __init__(self, optimizer, d_model: int, n_warmup_steps):
        self._optimizer=optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step(self):
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

class PLM_Trainer:

    def __init__(self,
                 save: bool,        # If model will be saved.
                 vocab_size: int,
                 embedding_dim:int,           # BERT parameters
                 dropout: float,
                 max_len: int,
                 mask_prob: float,
                 n_transformer_layers:int,
                 n_attn_heads: int,

                 batch_size: int,             # Learning parameters
                 lr: float=1e-4,
                 betas=(0.9, 0.999),
                 weight_decay: float = 0.01,
                 warmup_steps: int= 10000,
                 device: str='cpu',

    ):

        if save:
            # create the file name
            now = datetime.datetime.now()
            date = now.strftime("%Y-%m-%d")
            hour = now.strftime("%H")
            minute = now.strftime("%M")
            self.save_as = f"data_{date}_{hour}_{minute}"

        self.device = device


        # initialize model
        self.tokenizer = ProteinTokenizer(max_len, mask_prob)
        bert = BERT(embedding_dim, dropout, max_len, mask_prob, n_transformer_layers, n_attn_heads)
        self.model = ProteinLM(bert, vocab_size)
        self.model.to(self.device)

        # set optimizer, scheduler, and criterion
        self.optim = torch.optim.Adam(self.model.parameters(),
                                      lr = lr,
                                      betas = betas,
                                      weight_decay = weight_decay)

        self.optim_schedule = ScheduledOptim(self.optim, embedding_dim, warmup_steps)
        self.criterion = torch.nn.CrossEntropyLoss()

    def print_model_params(self) -> None:
        total_params = 0
        for name, param in self.model.named_parameters():
            num_params = param.numel()
            total_params += num_params
            print(f'{name}: {num_params}')
        print(f'Total number of parameters: {total_params}')



    def train(self, train_data, num_epochs: int, max_batch: Union[int, None]):
        """
        Model training

        Parameters:
        train_data: a dataloader feeding protein sequences
        num_epochs: number of toal epochs in training.
        max_batch: maximum number of batches in training. If not defined, all avaiable batches from train_data will be used.
        """
        WRITE = False
        
        if not max_batch:
            max_batch = len(train_loader)

        if hasattr(self, 'save_as'):        # TODO: check write access
            loss_history_file = ''.join([self.save_as, '_results.csv'])
            fh = open(loss_history_file, 'w')
            fh.write('epoch, loss\n')
            WRITE = True

        for epoch in range(1, num_epochs + 1):
            running_loss = self.epoch_iteration(epoch, max_batch, train_data, train=True)

            if epoch % 1 == 0:
                print(f'Epoch {epoch}: Average Loss: {running_loss}')
                if WRITE:
                    fh.write(f'{epoch}, {running_loss}\n')

            if epoch % 10 == 0:
                if hasattr(self, 'save_as'):
                    self.save_model()
        if WRITE:
            fh.close()
                    

    def test(self, test_data, num_epochs: int=10):
        self.epoch_iteration(epoch, test_data, train=False)

    def epoch_iteration(self, num_epochs: int, max_batch: int, data_loader, train: bool=True):
        """
        Loop over dataloader for training or testing

        For training mode, backpropogation is activated
        """
        mode = "train" if train else "test"

        # set the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                            desc=f'EP_{mode}: {num_epochs}',
                            total = len(data_loader),
                            bar_format='{l_bar}{r_bar}')

        running_loss = 0

        for i, batch_data in data_iter:
            if i >= max_batch:
                break
            seq_ids, seqs = batch_data
            tokenized_seqs, mask_idx = self.tokenizer(seqs)
            tokenized_seqs = tokenized_seqs.to(self.device)
            logits = self.model(tokenized_seqs)
            loss = self.criterion(logits.view(-1, logits.size(-1)), tokenized_seqs.view(-1))
            if train:
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim_schedule.step()
            running_loss += (loss.item() if running_loss is None else 0.99 * running_loss + 0.01 * loss.item())

        
        return running_loss

    def save_model(self):
        if self.save_as:
            file_path = ''.join([self.save_as, '_model_weights.pth'])
            torch.save(self.model.state_dict(), file_path)        

    # TODO: add load_model_parameter()



if __name__=="__main__":
    # Data loader
    db_file = path.abspath(path.dirname(__file__))
    db_file = path.join(db_file, '../../../data/SARS_CoV_2_spike.db')
    train_dataset = SeqDataset(db_file, "train")
    print(f'Sequence db file: {db_file}')
    print(f'Total seqs in training set: {len(train_dataset)}')


    embedding_dim = 36
    dropout=0.1
    max_len = 1500
    mask_prob = 0.15
    n_transformer_layers = 12
    attn_heads = 12

    batch_size = 10
    lr = 0.0001
    weight_decay = 0.01
    warmup_steps = 1000

    betas=(0.9, 0.999)
    tokenizer = ProteinTokenizer(max_len, mask_prob)
    embedder = NLPEmbedding(embedding_dim, max_len,dropout)

    vocab_size = len(token_to_index)
    hidden = embedding_dim

    num_epochs = 100
    num_workers = 1
    n_test_baches = 100

    USE_GPU = True
    SAVE_MODEL = True
    device = torch.device("cuda:0" if torch.cuda.is_available() and USE_GPU else "cpu")
    print(f'\nUsing device: {device}')


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


    trainer = PLM_Trainer(SAVE_MODEL, vocab_size, embedding_dim=embedding_dim, dropout=dropout, max_len=max_len,
                          mask_prob=mask_prob, n_transformer_layers=n_transformer_layers,
                          n_attn_heads=attn_heads, batch_size=batch_size, lr=lr, betas=betas,
                          weight_decay=weight_decay, warmup_steps=warmup_steps, device=device)
    # trainer.print_model_params()
    trainer.train(train_data = train_loader, num_epochs = num_epochs, max_batch=n_test_baches)
