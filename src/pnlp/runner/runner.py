#!/usr/bin/env python
"""Model Runner"""

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
from collections import defaultdict, OrderedDict
from pnlp.db.dataset import SeqDataset, initialize_db
from pnlp.embedding.tokenizer import ProteinTokenizer, token_to_index, index_to_token
from pnlp.embedding.nlp_embedding import NLPEmbedding
from pnlp.model.language import ProteinLM
from pnlp.model.bert import BERT
from pnlp.plots import plot_run, plot_accuracy_stats

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
                 load_model: bool,
                 checkpoint: bool,
                 
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
            save_path = os.path.join(os.path.dirname(__file__), f'../../../results/runner/{date_hour_minute}')
            self.save_as = os.path.join(save_path, date_hour_minute)

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
        self.load_model = load_model
        self.checkpoint = checkpoint
        self.epochs_ran = 1 
        self.best_acc = 0
        self.best_loss = float('inf')
        self.save_best = ''.join([self.save_as, '_best'])
        self.model_pth = ''

        # Initialize model
        self.tokenizer = ProteinTokenizer(max_len, mask_prob)
        bert = BERT(embedding_dim, dropout, max_len, mask_prob, n_transformer_layers, n_attn_heads)
        self.model = ProteinLM(bert, vocab_size)
        self.model.to(self.device)

        # Set optimizer, scheduler, and criterion
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
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
            self.run_preds_csv = ''.join([self.save_as, '_predictions.csv'])
            self.aa_preds_tracker = {}

            with open(self.run_result_csv,'w') as fh:
                WRITE = True
                
                if not self.checkpoint:
                    fh.write('epoch, train_loss, train_accuracy, test_loss, test_accuracy\n')
                    fh.flush() # flush line buffer
                else:
                    # Load the saved data into new .csv from the loaded model 
                    loaded_csv = ''.join(['_'.join(self.model_pth.split('_')[:-2]), '_results.csv'])
                    if os.path.exists(loaded_csv): 
                        with open(loaded_csv, 'r') as fc:
                            fh.writelines(fc)
                    else:
                        logger.warning(f"The _results.csv file '{loaded_csv}' does not exist. Ending program.")
                        exit()

                for epoch in range(self.epochs_ran, num_epochs+1):
                    logger.info(f"Epoch {epoch}")
                    start_time = time.time()

                    # The defaultdict(int) assigns a default value of 0 for a key that does not yet exist
                    self.aa_pred_counter = defaultdict(int)
                                        
                    train_loss, train_accuracy = self.epoch_iteration(epoch, max_batch, train_data, mode='train')
                    test_loss, test_accuracy = self.epoch_iteration(epoch, max_batch, test_data, mode='test')

                    fh.write(f'{epoch}, {train_loss:.2f}, {train_accuracy:.2f}, {test_loss:.2f}, {test_accuracy:.2f}\n')
                    fh.flush()

                    for key in self.aa_pred_counter:
                        if key not in self.aa_preds_tracker :
                            self.aa_preds_tracker[key] = {}
                        self.aa_preds_tracker[key][epoch] = self.aa_pred_counter.get(key)

                    if hasattr(self, 'save_as'):
                        self._save_model(epoch, self.save_as)
                        logger.info(f"\tModel saved at {''.join([self.save_as, '_model_weights.pth'])}")

                        # SAVE BEST MODEL
                        test_acc = float(f"{test_accuracy:.2f}")
                        test_loss = float(f"{test_loss:.2f}")
                        if test_acc > self.best_acc or (test_acc == self.best_acc and test_loss < self.best_loss):
                            self.best_acc = test_acc
                            self.best_loss = test_loss  # Update self.best_loss when better accuracy is found
                            self._save_model(epoch, self.save_best)
                            logger.info(f"\tNEW BEST MODEL; model saved. ACC: {test_acc}, LOSS: {test_loss}")

                    total_epoch_time = time.time() - start_time
                    formatted_hms = time.strftime("%H:%M:%S", time.gmtime(total_epoch_time))
                    decimal_sec = str(total_epoch_time).split('.')[1][:2]
                    msg = f'train loss: {train_loss:.2f}, train accuracy: {train_accuracy:.2f}, test loss: {test_loss:.2f}, test accuracy: {test_accuracy:.2f}, time: {formatted_hms}.{decimal_sec}'
                    print(f'Epoch {epoch} | {msg}')
                    logger.info(f'\t{msg}')

        if WRITE:
            plot_run.plot_run(self.run_result_csv, save=True)
            logger.info(f'Run result saved to {os.path.basename(self.run_result_csv)}')

            # Write to csv, plot
            with open(self.run_preds_csv, 'w') as fg:
                header = ", ".join(f"epoch {epoch}" for epoch in range(1, num_epochs + 1))
                header = f"expected_aa->predicted_aa, {header}\n"
                fg.write(header)

                for key in self.aa_preds_tracker:
                    self.aa_preds_tracker[key] = [self.aa_preds_tracker[key].get(epoch, 0) for epoch in range(1, num_epochs + 1)]
                    data_row = ", ".join(str(val) for val in self.aa_preds_tracker[key])
                    fg.write(f"{key}, {data_row}\n")

            plot_accuracy_stats.plot_aa_perc_pred_stats_heatmap(self.aa_preds_tracker, self.run_preds_csv, save=True)
            logger.info(f'Predictions csv saved to {os.path.basename(self.run_preds_csv)}')

    def epoch_iteration(self, num_epochs: int, max_batch: int, data_loader, mode:str):
        """
        Loop over dataloader for training or testing

        For training mode, backpropogation is activated
        """
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

            # STATS
            if mode == "test":
                ALL_AAS = 'ACDEFGHIKLMNPQRSTUVWXY'
                token_to_aa = {i:aa for i, aa in enumerate(ALL_AAS)}                
                # Create a list of keys from masked_loactions in format "expected_aa -> predicted_aa" where expected_aa != predicted_aa
                aa_keys = [f"{token_to_aa.get(token.item())}->{token_to_aa.get(pred_token.item())}" for token, pred_token in zip(labels[masked_locations], predicted_tokens[masked_locations])]
                # Update the tracker as going through keys (counting occurences)
                self.aa_pred_counter.update((aa_key, self.aa_pred_counter[aa_key] + 1) for aa_key in aa_keys)                


        accuracy = correct_predictions / total_masked
        return total_epoch_loss, accuracy
    

    def _save_model(self, epoch:int, save_path:str):
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

    def _load_model(self, pth:str):
        """
        Load a saved model status dictionary. 
        checkpoint as True means loading from ".model_weights.pth"
        checkpoint as False means loading from ".best_model_weights.pth"

        pth: saved model state dictionary file (.pth file in results directory)

        # TODO: need to call dataloader.load_state_dict with saved dataloader state. But
        https://github.com/pytorch/pytorch/blob/main/torch/utils/data/dataloader.py says
        the __getstate__ is not implemented.
        """
        saved_state = torch.load(pth, map_location=self.device)
        saved_hyperparameters = saved_state['hyperparameters']
        state_dict = saved_state['model_state_dict']

        if saved_hyperparameters["device"] != self.device:
            msg = f'Map saved status from {saved_hyperparameters["device"]} to {self.device}.'
            logger.warning(msg)

        hyperparameters = ['vocab_size','embedding_dim','dropout','max_len','mask_prob',
                           'n_transformer_layers','n_attn_heads','batch_size','init_lr',
                           'betas','weight_decay','warmup_steps']

        for hp in hyperparameters:
            assert getattr(self, hp) == saved_hyperparameters[hp], f"{hp} mismatch"

        # For loading from ddp models, they have 'module' in keys of state_dict
        load_ddp = False
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k[:6] == 'module':
                load_ddp = True
                new_state_dict[k[7:]] = v   # remove 'module'
            else: break
        if load_ddp: state_dict = new_state_dict

        # Load pretrained state_dict
        self.model.load_state_dict(state_dict)
        random.setstate(saved_state['random_state'])
        if isinstance(saved_state['rng_state'], torch.ByteTensor):
            torch.set_rng_state(saved_state['rng_state'])  # restore random number state.

        if self.checkpoint:
            self.epochs_ran = saved_state['epochs_ran']
            self.best_acc = saved_state['best_acc']
            self.best_loss = saved_state['best_loss']
            self.optim.load_state_dict(saved_state['optim_state_dict'])

if __name__=="__main__":

    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    results_dir = os.path.join(os.path.join(os.path.dirname(__file__), '../../../results/runner'))
    run_dir = os.path.join(results_dir, f'{date_hour_minute}')
    os.makedirs(run_dir, exist_ok = True)

    # Add logging configuration
    log_file = os.path.join(run_dir, f'{date_hour_minute}.log')
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
    dropout = 0.1
    max_len = 280
    mask_prob = 0.15
    n_transformer_layers = 12
    attn_heads = 12
    hidden = embedding_dim

    batch_size = 64
    n_test_baches = 20
    num_epochs = 105

    lr = 1e-05
    weight_decay = 0.01
    warmup_steps = 435 # (training set size / batch size) * 0.1
    betas=(0.9, 0.999)

    tokenizer = ProteinTokenizer(max_len, mask_prob)
    embedder = NLPEmbedding(embedding_dim, max_len,dropout)
    vocab_size = len(token_to_index)

    SAVE_MODEL = True
    LOAD_MODEL = False
    CHECKPOINT = False
    model_pth = ''

    USE_GPU = True
    device = torch.device("cuda:0" if torch.cuda.is_available() and USE_GPU else "cpu")

    torch.manual_seed(0)        # Dataloader uses its own random number generator.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    runner = Model_Runner(SAVE_MODEL, LOAD_MODEL, CHECKPOINT,
                          vocab_size=vocab_size, embedding_dim=embedding_dim, dropout=dropout, 
                          max_len=max_len, mask_prob=mask_prob, n_transformer_layers=n_transformer_layers,
                          n_attn_heads=attn_heads, batch_size=batch_size, lr=lr, betas=betas,
                          weight_decay=weight_decay, warmup_steps=warmup_steps, device=device)

    logger.info(f'Run results located in this directory: {run_dir}')
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

    if LOAD_MODEL and os.path.exists(model_pth):
        logger.info(f'Loading model: {model_pth}')
        runner.model_pth = model_pth
        runner._load_model(runner.model_pth)
        runner.model.train()
        print(f"Loading from saved model at Epoch {runner.epochs_ran}")
        logger.info(f"RLoading from saved model at Epoch {runner.epochs_ran}")
    elif LOAD_MODEL and not os.path.exists(model_pth):
        logger.warning(f"The .pth file {model_pth} does not exist; there is no model to load. Ending program.")
        exit()

    runner.run(train_data = train_loader, test_data = test_loader,
               num_epochs = num_epochs, max_batch = n_test_baches)