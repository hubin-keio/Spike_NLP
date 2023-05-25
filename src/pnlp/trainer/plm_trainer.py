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
from pnlp.embedding.tokenizer import ProteinTokenizer, token_to_index, index_to_token
from pnlp.embedding.nlp_embedding import NLPEmbedding
from pnlp.model.language import ProteinLM
from pnlp.model.bert import BERT


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
                 device: str='cpu'):

        if save:
            # create the file name
            now = datetime.datetime.now()
            date = now.strftime("%Y-%m-%d")
            hour = now.strftime("%H")
            minute = now.strftime("%M")
            save_path = os.path.join(os.path.dirname(__file__),
                                     '../../../results')
            self.save_as = os.path.join(save_path, f"data_{date}_{hour}_{minute}")

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
        self.criterion = torch.nn.CrossEntropyLoss(reduction='sum')  # sum of CEL at batch level.

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
            fh.write('epoch, loss, accuracy\n')
            WRITE = True

        for epoch in range(1, num_epochs + 1):
            epoch_loss, masked_accuracy = self.epoch_iteration(epoch, max_batch, train_data, train=True)

            if epoch % 1 == 0:
                print(f'Epoch {epoch}, loss: {epoch_loss}, masked_accuracy: {masked_accuracy}')
                if WRITE:
                    fh.write(f'{epoch}, {epoch_loss}, {masked_accuracy}\n')

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

        MASK_TOKEN_IDX = token_to_index['<MASK>']

        # set the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                            desc=f'EP_{mode}: {num_epochs}',
                            total = len(data_loader),
                            bar_format='{l_bar}{r_bar}')

        total_epoch_loss = 0
        total_masked = 0
        correct_predictions = 0

        for i, batch_data in data_iter:
            if i >= max_batch:
                break

            seq_ids, seqs = batch_data
            tokenized_seqs = self.tokenizer(seqs)
            tokenized_seqs = tokenized_seqs.to(self.device)  # input tokens with masks
            predictions  = self.model(tokenized_seqs)        # model predictions
            labels = self.tokenizer._batch_pad(seqs).to(self.device)  # input tokens without masks
            loss = self.criterion(predictions.transpose(1, 2), labels)
            total_epoch_loss += loss.item()

            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim_schedule.step_and_update_lr()

            predicted_tokens  = torch.max(predictions, dim=-1)[1]
            masked_locations = torch.nonzero(torch.eq(tokenized_seqs, MASK_TOKEN_IDX), as_tuple=True)
            correct_predictions += torch.eq(predicted_tokens[masked_locations],
                                            labels[masked_locations]).sum().item()

            total_masked += masked_locations[0].numel()

        accuracy = correct_predictions / total_masked
        return total_epoch_loss, accuracy


    def save_model(self):
        if self.save_as:
            file_path = ''.join([self.save_as, '_model_weights.pth'])
            torch.save(self.model.state_dict(), file_path)

    # TODO: add load_model_parameter()

if __name__=="__main__":
    # Data loader
    db_file = path.abspath(path.dirname(__file__))
    db_file = path.join(db_file, '../../../data/SARS_CoV_2_spike_noX_RBD.db')
    train_dataset = SeqDataset(db_file, "train")
    print(f'Sequence db file: {os.path.basename(db_file)}')
    print(f'Total seqs in training set: {len(train_dataset)}')

    embedding_dim = 24
    dropout=0.1
    max_len = 280
    mask_prob = 0.15
    n_transformer_layers = 12
    attn_heads = 12

    batch_size = 50
    lr = 0.0001
    weight_decay = 0.01
    warmup_steps = 10

    betas=(0.9, 0.999)
    tokenizer = ProteinTokenizer(max_len, mask_prob)
    embedder = NLPEmbedding(embedding_dim, max_len,dropout)

    vocab_size = len(token_to_index)
    hidden = embedding_dim

    num_epochs = 100
    num_workers = 1
    n_test_baches = 50

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
    trainer.train(train_data = train_loader, num_epochs = num_epochs, max_batch = n_test_baches)
