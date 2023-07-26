#!/usr/bin/env python
"""Model Runner for running on HPC. SLURM job IDs are used as identifier for runs, unlike date & time for non HPC runs."""

import os
import sys
import time
import datetime
import random
from datetime import date
from os import path
from typing import Union
import logging
import numpy as np
import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import logging
from pnlp.db.dataset import SeqDataset, initialize_db
from pnlp.embedding.tokenizer import ProteinTokenizer, token_to_index, index_to_token
from pnlp.embedding.nlp_embedding import NLPEmbedding
from pnlp.model.language import ProteinLM
from pnlp.model.bert import BERT
from pnlp.plots import plot_run

logger = logging.getLogger(__name__)

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

class Model_Runner:

    def __init__(self,
                 save: bool,                  # If model will be saved.
                 load: bool,
                 job_id: str,
                 
                 vocab_size:int,
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
            # Create the file name
            now = datetime.datetime.now()
            date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
            save_path = os.path.join(os.path.dirname(__file__), '../../../results')
            self.save_as = os.path.join(save_path, job_id, date_hour_minute)

        self.device = device
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.n_transformer_layers = n_transformer_layers
        self.n_attn_heads = n_attn_heads
        self.batch_size = batch_size
        self.init_lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        # For saving/loading
        self.load = load
        self.epochs_ran = 1 
        self.best_acc = 0
        self.best_loss = float('inf')

        self.save_best_acc = ''.join([self.save_as, '_best_acc'])
        self.save_best_loss = ''.join([self.save_as, '_best_loss'])
        self.model_pth = ''

        # Initialize model
        self.tokenizer = ProteinTokenizer(max_len, mask_prob)
        bert = BERT(embedding_dim, dropout, max_len, mask_prob, n_transformer_layers, n_attn_heads)
        self.model = ProteinLM(bert, vocab_size)
        self.model.to(self.device)

        # Set optimizer, scheduler, and criterion
        self.optim = torch.optim.Adam(self.model.parameters(),
                                      lr = lr,
                                      betas = betas,
                                      weight_decay = weight_decay)

        self.optim_schedule = ScheduledOptim(self.optim, embedding_dim, warmup_steps)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='sum')  # sum of CEL at batch level.

    def count_parameters_with_gradidents(self) -> int:
        count = 0
        for param in self.model.parameters():
            if param.requires_grad:
                count += param.numel()
        return count

    def run(self, train_data:SeqDataset, test_data:SeqDataset, num_epochs: int, max_batch: Union[int, None]):
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
            self.run_result_csv = ''.join([self.save_as, '_results.csv'])

            with open(self.run_result_csv,'w') as fh:
                WRITE = True
                
                if not self.load:
                    fh.write('epoch, train_loss, train_accuracy, test_loss, test_accuracy\n')
                    fh.flush() # flush the buffer so lines get written to file

                else:
                    # Load the saved data into new .csv from the loaded model 
                    loaded_csv = "".join(["_".join(self.model_pth.split("_")[:-2]), '_results.csv'])
                    if os.path.exists(loaded_csv): 
                        with open(loaded_csv, 'r') as fc:
                            for line in fc:
                                fh.write(line)
                                fh.flush()  

                    else:
                        logger.warning(f"The _results.csv file '{loaded_csv}' does not exist. Ending program.")
                        exit()

                for epoch in range(self.epochs_ran, num_epochs+1):
                    logger.info(f"Epoch {epoch}")
                    start_time = time.time()
                    train_loss, train_accuracy = self.epoch_iteration(epoch, max_batch, train_data, train=True)
                    test_loss, test_accuracy = self.epoch_iteration(epoch, max_batch, test_data, train=False)

                    if epoch < 11 or epoch % 10 == 0:
                       fh.write(f'{epoch}, {train_loss:.2f}, {train_accuracy:.2f}, {test_loss:.2f}, {test_accuracy:.2f}\n')
                       fh.flush()

                    if hasattr(self, 'save_as'):
                        self.save_model(epoch, self.save_as)
                        logger.info(f"\tModel saved at {''.join([self.save_as, '_model_weights.pth'])}")

                        test_accuracy = round(test_accuracy, 2)
                        test_loss = round(test_loss, 2)

                        if test_accuracy > self.best_acc:
                            self.best_acc = test_accuracy
                            self.save_model(epoch, self.save_best_acc)
                            logger.info(f"\tNEW BEST ACCURACY: {self.best_acc}; model saved. ACC: {test_accuracy}, LOSS: {test_loss}")

                        if test_loss < self.best_loss:
                            self.best_loss = test_loss
                            self.save_model(epoch, self.save_best_loss)
                            logger.info(f"\tNEW BEST LOSS: {self.best_loss}; model saved. ACC: {test_accuracy}, LOSS: {test_loss}")
                    
                    total_epoch_time = time.time() - start_time
                    msg = f'train loss: {train_loss:.2f}, train accuracy: {train_accuracy:.2f}, test loss: {test_loss:.2f}, test accuracy: {test_accuracy:.2f}, time: {total_epoch_time}'
                    print(f'Epoch {epoch} | {msg}')
                    logger.info(f'\t{msg}')

        if WRITE:
            plot_run.plot_run(self.run_result_csv, save=True)
            logger.info(f'Run result saved to {os.path.basename(self.run_result_csv)}')

    def epoch_iteration(self, num_epochs: int, max_batch: int, data_loader, train: bool=True):
        """
        Loop over dataloader for training or testing

        For training mode, backpropogation is activated
        """
        mode = "train" if train else "test"

        MASK_TOKEN_IDX = token_to_index['<MASK>']

        # Set the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                            desc=f'EP_{mode}: {num_epochs}',
                            total = len(data_loader),
                            bar_format='{l_bar}{r_bar}')

        total_epoch_loss = 0
        total_masked = 0
        correct_predictions = 0

        for i, batch_data in data_iter:
            if max_batch > 0 and i >= max_batch:
                break

            seq_ids, seqs = batch_data
            tokenized_seqs = self.tokenizer(seqs)
            tokenized_seqs = tokenized_seqs.to(self.device)  # input tokens with masks
            labels = self.tokenizer._batch_pad(seqs).to(self.device)  # input tokens without masks

            if mode == 'train':  # train mode
                predictions  = self.model(tokenized_seqs)
                loss = self.criterion(predictions.transpose(1, 2), labels)                
                self.optim_schedule.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim_schedule.step_and_update_lr()

            else:                # test mode
                self.model.eval()
                with torch.no_grad():
                    predictions  = self.model(tokenized_seqs)
                loss = self.criterion(predictions.transpose(1, 2), labels)
                self.model.train()   # set the model back to training mode

            total_epoch_loss += loss.item()

            predicted_tokens  = torch.max(predictions, dim=-1)[1]
            masked_locations = torch.nonzero(torch.eq(tokenized_seqs, MASK_TOKEN_IDX), as_tuple=True)
            correct_predictions += torch.eq(predicted_tokens[masked_locations],
                                            labels[masked_locations]).sum().item()

            total_masked += masked_locations[0].numel()

        accuracy = correct_predictions / total_masked
        return total_epoch_loss, accuracy

    def save_model(self, epoch:int, save_path:str):
        if self.save_as:
            file_path = ''.join([save_path, '_model_weights.pth'])

            hyperparameters = {'device' : self.device, 
                               'vocab_size' : self.vocab_size,
                               'embedding_dim' : self.embedding_dim,
                               'dropout' : self.dropout,
                               'max_len' : self.max_len,
                               'mask_prob' : self.mask_prob,
                               'n_transformer_layers' : self.n_transformer_layers,
                               'n_attn_heads' : self.n_attn_heads,
                               'batch_size' : self.batch_size,
                               'init_lr' : self.init_lr,
                               'betas' : self.betas,
                               'weight_decay' : self.weight_decay,
                               'warmup_steps' : self.warmup_steps}

            state = {'hyperparameters': hyperparameters,
                     'epochs_ran': epoch+1,
                     'best_acc':self.best_acc,
                     'best_loss':self.best_loss,
                     'rng_state': torch.get_rng_state(),  # torch random number state.
                     'random_state': random.getstate(),
                     'model_state_dict': self.model.state_dict(),
                     'optim_state_dict': self.optim.state_dict()}

            torch.save(state, file_path)

    def load_model_checkpoint(self, pth:str):
        """
        Load a saved model status dictionary. 
        This should NOT be the best accuracy or best loss .pth. 
        This is for checkpointing.

        pth: saved model state dictionary file (.pth file in results directory)
        """
        if "best" in pth:
            logger.warning(f"The .pth file used should not be of a saved best acc/loss model. Ending program.")
            exit()

        map_location = self.device
        saved_state = torch.load(pth, map_location=map_location)
        saved_hyperparameters = saved_state['hyperparameters']
        if saved_hyperparameters["device"] != self.device:
            msg = f'Map saved status from {saved_hyperparameters["device"]} to {self.device}.'
            logger.warning(msg)

        assert self.vocab_size == saved_hyperparameters['vocab_size']
        assert self.embedding_dim == saved_hyperparameters['embedding_dim']
        assert self.dropout == saved_hyperparameters['dropout']
        assert self.max_len == saved_hyperparameters['max_len']
        assert self.mask_prob == saved_hyperparameters['mask_prob']
        assert self.n_transformer_layers == saved_hyperparameters['n_transformer_layers']
        assert self.n_attn_heads == saved_hyperparameters['n_attn_heads']
        assert self.batch_size == saved_hyperparameters['batch_size']
        assert self.init_lr == saved_hyperparameters['init_lr']
        assert self.betas == saved_hyperparameters['betas']
        assert self.weight_decay == saved_hyperparameters['weight_decay']
        assert self.warmup_steps == saved_hyperparameters['warmup_steps']

        self.epochs_ran = saved_state['epochs_ran']
        self.best_acc = saved_state['best_acc']
        self.best_loss = saved_state['best_loss']
        self.model.load_state_dict(saved_state['model_state_dict'])
        self.optim.load_state_dict(saved_state['optim_state_dict'])
        random.setstate(saved_state['random_state'])

    def load_model_parameters(self, pth:str):
        """
        Load a saved model status dictionary.

        pth: saved model state dictionary file (.pth file in results directory)

        # TODO: need to call dataloader.load_state_dict with saved dataloader state. But
        https://github.com/pytorch/pytorch/blob/main/torch/utils/data/dataloader.py says
        the __getstate__ is not implemented.
        """
        saved_state = torch.load(pth, map_location=self.device)
        saved_hyperparameters = saved_state['hyperparameters']
        if saved_hyperparameters["device"] != self.device:
            msg = f'Map saved status from {saved_hyperparameters["device"]} to {self.device}.'
            logger.warning(msg)

        assert self.vocab_size == saved_hyperparameters['vocab_size']
        assert self.embedding_dim == saved_hyperparameters['embedding_dim']
        assert self.dropout == saved_hyperparameters['dropout']
        assert self.max_len == saved_hyperparameters['max_len']
        assert self.mask_prob == saved_hyperparameters['mask_prob']
        assert self.n_transformer_layers == saved_hyperparameters['n_transformer_layers']
        assert self.n_attn_heads == saved_hyperparameters['n_attn_heads']
        assert self.batch_size == saved_hyperparameters['batch_size']
        assert self.init_lr == saved_hyperparameters['init_lr']
        assert self.betas == saved_hyperparameters['betas']
        assert self.weight_decay == saved_hyperparameters['weight_decay']
        assert self.warmup_steps == saved_hyperparameters['warmup_steps']

        self.model.load_state_dict(saved_state['model_state_dict'])
        random.setstate(saved_state['random_state'])
        if isinstance(saved_state['rng_state'], torch.ByteTensor):
            torch.set_rng_state(saved_state['rng_state'])  # restore random number state.

if __name__=="__main__":

    job_id = os.environ.get('SLURM_JOB_ID')
    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    log_file = os.path.join(os.path.dirname(__file__), f'../../../results/{job_id}')
    log_file = os.path.join(log_file, f'{date_hour_minute}.log')

    # Add logging configuration
    logging.basicConfig(
        filename = log_file,
        format = '%(asctime)s - %(levelname)s - %(message)s',
        datefmt = '%m/%d/%Y %I:%M:%S %p')
    logger.setLevel(logging.DEBUG)  # To disable the matplotlib font_manager logs.

    # Data loader
    db_file = path.abspath(path.dirname(__file__))
    db_file = path.join(db_file, '../../../data/SARS_CoV_2_spike_noX_RBD.db')
    train_dataset = SeqDataset(db_file, "train")
    test_dataset = SeqDataset(db_file, "test")

    # -= HYPERPARAMETERS =-
    embedding_dim = 768
    dropout=0.1
    max_len = 280
    mask_prob = 0.15
    n_transformer_layers = 12
    attn_heads = 12
    hidden = embedding_dim

    batch_size = 64
    n_test_baches = -1
    num_epochs = 100

    lr = 0.0001
    weight_decay = 0.01
    warmup_steps = 10
    betas=(0.9, 0.999)

    tokenizer = ProteinTokenizer(max_len, mask_prob)
    embedder = NLPEmbedding(embedding_dim, max_len,dropout)
    vocab_size = len(token_to_index)

    SAVE_MODEL = True
    USE_GPU = True
    LOAD_MODEL_CHECKPOINT = True
    model_checkpoint_pth="Spike_NLP/results/2023-07-25_10-25/2023-07-25_10-25_model_weights.pth"

    device = torch.device("cuda:0" if torch.cuda.is_available() and USE_GPU else "cpu")

    torch.manual_seed(0)        # Dataloader uses its own random number generator.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    runner = Model_Runner(SAVE_MODEL, LOAD_MODEL_CHECKPOINT, job_id=job_id,
                          vocab_size=vocab_size, embedding_dim=embedding_dim, dropout=dropout, 
                          max_len=max_len, mask_prob=mask_prob, n_transformer_layers=n_transformer_layers,
                          n_attn_heads=attn_heads, batch_size=batch_size, lr=lr, betas=betas,
                          weight_decay=weight_decay, warmup_steps=warmup_steps, device=device)

    logger.info(f'slurm_job_id: {job_id}')
    logger.info(f'Run results located in this directory: ../../../results/{job_id}')
    logger.info(f'Using device: {device}')
    logger.info(f'Data set: {os.path.basename(db_file)}')
    logger.info(f'Total seqs in training set: {len(train_dataset)}')
    logger.info(f'Total seqs in testing set: {len(test_dataset)}')
    logger.info(f'num_epochs: {num_epochs}')
    logger.info(f'n_test_baches: {n_test_baches}')
    logger.info(f'batch_size: {batch_size}')
    logger.info(f'embedding_dim: {embedding_dim}')
    logger.info(f'dropout: {dropout}')
    logger.info(f'max_len: {max_len}')
    logger.info(f'mask_prob: {mask_prob}')
    logger.info(f'n_transformer_layers: {n_transformer_layers}')
    logger.info(f'n_attn_heads: {attn_heads}')
    logger.info(f'lr: {lr}')
    logger.info(f'weight_decay: {weight_decay}')
    logger.info(f'warmup_steps: {warmup_steps}')
    logger.info(f'vocab_size: {vocab_size}')
    logger.info(f'hidden: {hidden}')
    logger.info(f'Number of parameters: {"{:,.0f}".format(runner.count_parameters_with_gradidents())}')

    if LOAD_MODEL_CHECKPOINT and os.path.exists(model_checkpoint_pth):
        logger.info(f'Loading model checkpoint: {model_checkpoint_pth}')
        runner.model_pth = model_checkpoint_pth
        runner.load_model_checkpoint(model_checkpoint_pth)
        runner.model.train()
        print(f"Resuming training from saved model checkpoint at Epoch {runner.epochs_ran}")
        logger.info(f"Resuming training from saved model checkpoint at Epoch {runner.epochs_ran}")
    elif LOAD_MODEL_CHECKPOINT and not os.path.exists(model_checkpoint_pth):
        logger.warning(f"The .pth file {model_checkpoint_pth} does not exist; there is no model to load. Ending program.")
        exit()

    runner.run(train_data = train_loader, test_data = test_loader,
               num_epochs = num_epochs, max_batch = n_test_baches)
