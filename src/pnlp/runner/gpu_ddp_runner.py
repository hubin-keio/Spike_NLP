#!/usr/bin/env python
"""
Model Runner running DistributedDataParallel.
*GPU ONLY* to avoid device assignment issues.

Usage:
    torchrun
    > --standalone: utilize single node
    > --nproc_per_node: number of processes/gpus

    Example equivalent commands to run (single node, 2 gpu; top is for bio-lambda cluster, bottom is generic):
        /data/miniconda3/envs/spike_env/bin/time -v torchrun --standalone --nproc_per_node=2 gpu_ddp_runner.py
        /usr/bin/time -v torchrun --standalone --nproc_per_node=2 gpu_ddp_runner.py
"""

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
from collections import defaultdict
from pnlp.db.dataset import SeqDataset, initialize_db
from pnlp.embedding.tokenizer import ProteinTokenizer, token_to_index, index_to_token
from pnlp.embedding.nlp_embedding import NLPEmbedding
from pnlp.model.language import ProteinLM
from pnlp.model.bert import BERT
from pnlp.plots import plot_run, plot_accuracy_stats
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record

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

class Model_Runner():

    def __init__(self,
                 save: bool,                  # If model will be saved.
                 load_checkpoint: bool,       # If loading in previous model/checkpoint
                 load_best: bool,
                 device: int,                 # Device is a gpu, specifically its rank designation

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
                 warmup_steps: int= 10000):

        if save:
            # Create the file name
            now = datetime.datetime.now()
            date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
            ddp_date_hour_minute = ''.join(['ddp-', date_hour_minute])
            save_path = os.path.join(os.path.dirname(__file__), f'../../../results/{ddp_date_hour_minute}')
            self.save_as = os.path.join(save_path, ddp_date_hour_minute)
        
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
        self.load_checkpoint = load_checkpoint
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

        # Wrap model in DDP depending on device
        self.model = DDP(self.model,
                         device_ids=[device],
                         find_unused_parameters=True)
        self.world_size = int(dist.get_world_size())
        
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

    def run(self, train_sampler, test_sampler, train_data:SeqDataset, test_data:SeqDataset, num_epochs: int, max_batch: Union[int, None]):
        """
        Model training

        Parameters:
        train_data: a dataloader feeding protein sequences
        num_epochs: number of toal epochs in training.
        max_batch: maximum number of batches in training. If not defined, all avaiable batches from train_data will be used.
        """
        WRITE = False

        if not max_batch:
            max_batch = len(train_data)

        if hasattr(self, 'save_as'):        # TODO: check write access
            self.run_result_csv = ''.join([self.save_as, '_results.csv'])
            self.run_incorrect_preds_csv = ''.join([self.save_as, '_incorrect_predictions.csv'])
            self.incorrect_aa_preds_tracker = {}

            with open(self.run_result_csv,'w') as fh:
                WRITE = True
                
                if not self.load_checkpoint:
                    fh.write('epoch, train_loss, train_accuracy, test_loss, test_accuracy\n')
                    fh.flush() # flush line buffer
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

                for epoch in range(self.epochs_ran, num_epochs + 1):
                    if self.device == 0: logger.info(f"Epoch {epoch}")
                    start_time = time.time()

                    # The defaultdict(int) assigns a default value of 0 for a key that does not yet exist
                    self.incorrect_aa_pred_counter = defaultdict(int)

                    # Make sure every epoch the data distributed to processes differently
                    train_sampler.set_epoch(epoch)
                    test_sampler.set_epoch(epoch)

                    train_loss, train_accuracy = self.epoch_iteration(epoch, max_batch, train_data, train=True)
                    dist.barrier()
                    test_loss, test_accuracy = self.epoch_iteration(epoch, max_batch, test_data, train=False)

                    # Init list to store across all processes
                    train_losses = [None for _ in range(self.world_size)]
                    train_accuracies = [None for _ in range(self.world_size)]
                    test_losses = [None for _ in range(self.world_size)]
                    test_accuracies = [None for _ in range(self.world_size)]

                    # Communicate to one process
                    dist.all_gather_object(train_losses, train_loss)
                    dist.all_gather_object(train_accuracies, train_accuracy)
                    dist.all_gather_object(test_losses, test_loss)
                    dist.all_gather_object(test_accuracies, test_accuracy)

                    # Average across processes
                    train_loss_avg = sum(train_losses) / self.world_size
                    train_accuracy_avg = sum(train_accuracies) / self.world_size
                    test_loss_avg = sum(test_losses) / self.world_size
                    test_accuracy_avg = sum(test_accuracies) / self.world_size

                    # Prediction tracking across processes
                    incorrect_preds = [None for _ in range(self.world_size)] 
                    dist.all_gather_object(incorrect_preds, self.incorrect_aa_pred_counter)
                    
                    if self.device == 0:
                        # Add values to tracker dict from across processes counter dicts
                        for dictionary in incorrect_preds:
                            for key, value in dictionary.items():
                                if key not in self.incorrect_aa_preds_tracker:
                                    self.incorrect_aa_preds_tracker[key] = defaultdict(int)
                                self.incorrect_aa_preds_tracker[key][epoch] += value
 
                        fh.write(f'{epoch}, {train_loss_avg:.2f}, {train_accuracy_avg:.2f}, {test_loss_avg:.2f}, {test_accuracy_avg:.2f}\n')
                        fh.flush()

                        if hasattr(self, 'save_as'):
                            self.save_model(epoch, self.save_as)
                            logger.info(f"\tModel saved at {''.join([self.save_as, '_model_weights.pth'])}")

                            test_acc = float(f"{test_accuracy:.2f}")
                            test_loss = float(f"{test_loss:.2f}")

                            # SAVE BEST MODEL
                            if test_acc > self.best_acc or (test_acc == self.best_acc and test_loss < self.best_loss):
                                self.best_acc = test_acc
                                self.best_loss = test_loss  # Update self.best_loss when better accuracy is found
                                self.save_model(epoch, self.save_best)
                                logger.info(f"\tNEW BEST MODEL; model saved. ACC: {test_acc}, LOSS: {test_loss}")

                        total_epoch_time = time.time() - start_time
                        formatted_hms = time.strftime("%H:%M:%S", time.gmtime(total_epoch_time))
                        decimal_sec = str(total_epoch_time).split('.')[1][:2]
                        msg = f'train loss: {train_loss:.2f}, train accuracy: {train_accuracy:.2f}, test loss: {test_loss:.2f}, test accuracy: {test_accuracy:.2f}, time: {formatted_hms}.{decimal_sec}'
                        print(f'Epoch {epoch} | {msg}')
                        logger.info(f'\t{msg}')

                    # Add a barrier to prevent loading before saving is finished
                    dist.barrier()

        if WRITE:
            if self.device==0: 
                plot_run.plot_run(self.run_result_csv, save=True)
                logger.info(f'Run result saved to {os.path.basename(self.run_result_csv)}')

                # Write to csv, plot
                with open(self.run_incorrect_preds_csv, 'w') as fg:
                    header = ", ".join(f"epoch {epoch}" for epoch in range(1, num_epochs + 1))
                    header = f"expected_aa->predicted_aa, {header}\n"
                    fg.write(header)

                    for key in self.incorrect_aa_preds_tracker:
                        self.incorrect_aa_preds_tracker[key] = [self.incorrect_aa_preds_tracker[key].get(epoch, 0) for epoch in range(1, num_epochs + 1)]
                        data_row = ", ".join(str(val) for val in self.incorrect_aa_preds_tracker[key])
                        fg.write(f"{key}, {data_row}\n")

                plot_accuracy_stats.plot_top_predictions(self.incorrect_aa_preds_tracker, self.run_incorrect_preds_csv, save=True)
                plot_accuracy_stats.plot_pred_hist(self.incorrect_aa_preds_tracker, self.run_incorrect_preds_csv, save=True)
                logger.info(f'Incorrect predictions csv saved to {os.path.basename(self.run_incorrect_preds_csv)}')

    def epoch_iteration(self, num_epochs: int, max_batch: int, data_loader, train: bool=True):
        """
        Loop over dataloader for training or testing

        For training mode, backpropogation is activated
        """
        mode = "train" if train else "test"

        MASK_TOKEN_IDX = token_to_index['<MASK>']

        # set the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc=f'Rank: {self.device}, EP_{mode}: {num_epochs}',
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
                aa_keys = [f"{token_to_aa.get(token.item())}->{token_to_aa.get(pred_token.item())}" for token, pred_token in zip(labels[masked_locations], predicted_tokens[masked_locations]) if token!=pred_token]
                # Update the tracker as going through keys (counting occurences)
                self.incorrect_aa_pred_counter.update((aa_key, self.incorrect_aa_pred_counter[aa_key] + 1) for aa_key in aa_keys)                

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

        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.device}
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
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.device}
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

        self.model.load_state_dict(saved_state['model_state_dict'])
        random.setstate(saved_state['random_state'])
        if isinstance(saved_state['rng_state'], torch.ByteTensor):
            torch.set_rng_state(saved_state['rng_state'])  # restore random number state.

@record
def main():
    """
    Some notes:
        Torchrun
            For running the script, it simplifies a lot! Sets the environment variables
            MASTER_PORT, MASTER_ADDR, WORLD_SIZE, and RANK, which are required for torch.distributed.
        DistributedSampler
            Necessary for DDP. It creates disjuct subsets of batch indices then assigns each subset to a process.
            Used in the DataLoader, it will also shuffle the data.
        batch_size
            This is the batch_size ON EACH GPU/CPU process. Effective batch_size is n_processes * batch size.
            ex: 80000 total seqs = 22 batch size * 4 processes * 909 number of batches (set by train_loader)
        GPU
            GPU limited by batch size, too high will have CUDA OOM. This script updated for gpu only to avoid device assignment issues
            and avoid saving/loading between cpu and gpu devices.
    """
    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    directory = f'../../../results/ddp-{date_hour_minute}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    log_file = os.path.join(os.path.dirname(__file__), directory)
    log_file = os.path.join(log_file, f'ddp-{date_hour_minute}.log')

    # Add logging configuration
    logging.basicConfig(
        filename = log_file,
        format = '%(asctime)s - %(levelname)s - %(message)s',
        datefmt = '%m/%d/%Y %I:%M:%S %p')
    logger.setLevel(logging.DEBUG)  # To disable the matplotlib font_manager logs.

    # Check if can run on gpu
    rank = int(os.environ["LOCAL_RANK"])
    if rank == 0: logger.warning("Set to use GPU, if GPU is not available, program will terminate.")
    device = rank if torch.cuda.is_available() else exit()

    # Initialize process group
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=5400))
    torch.cuda.set_device(device) # set by rank of gpu within process

    # Data loader (database)
    db_file = path.abspath(path.dirname(__file__))
    db_file = path.join(db_file, '../../../data/SARS_CoV_2_spike_noX_RBD.db')
    train_dataset = SeqDataset(db_file, "train")
    test_dataset = SeqDataset(db_file, "test")

    ## -=HYPERPARAMETERS=- ##
    embedding_dim = 768 # 768 or 1536, 768 will be our default going forward
    dropout=0.1
    max_len = 280
    mask_prob = 0.15
    n_transformer_layers = 12
    attn_heads = 12
    hidden = embedding_dim

    batch_size = 64
    n_test_baches = -1
    num_epochs = 50

    lr = 0.0001
    weight_decay = 0.01
    warmup_steps = 435
    betas=(0.9, 0.999)

    tokenizer = ProteinTokenizer(max_len, mask_prob)
    embedder = NLPEmbedding(embedding_dim, max_len,dropout)
    vocab_size = len(token_to_index)

    SAVE_MODEL = True
    USE_GPU = True
    LOAD_MODEL_CHECKPOINT = False
    LOAD_BEST_MODEL = False
    model_checkpoint_pth=''
    best_model_pth=''

    # Distributed Samplers
    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset)

    # DataLoaders
    torch.manual_seed(0)        # Dataloader uses its own random number generator.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, sampler=test_sampler)

    # Initialize model runner
    runner = Model_Runner(SAVE_MODEL, LOAD_MODEL_CHECKPOINT, LOAD_BEST_MODEL, 
                          vocab_size=vocab_size, embedding_dim=embedding_dim, dropout=dropout, 
                          max_len=max_len, mask_prob=mask_prob, n_transformer_layers=n_transformer_layers,
                          n_attn_heads=attn_heads, batch_size=batch_size, lr=lr, betas=betas,
                          weight_decay=weight_decay, warmup_steps=warmup_steps, device=device)
    if device == 0:
        logger.info(f'Run results located in this directory: ../../../results/{directory}')
        logger.info(f'Using device: GPU')

        if dist.is_initialized(): logger.info(f"Process group has been initialized, Running DDP...")
        logger.info(f"World size: {int(dist.get_world_size())}")
        logger.info(f"Master address: {os.environ['MASTER_ADDR']}")
        logger.info(f"Master port: {os.environ['MASTER_PORT']}")

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
        if device == 0: logger.info(f'Loading model checkpoint: {model_checkpoint_pth}')
        runner.model_pth = model_checkpoint_pth
        runner.load_model_checkpoint(model_checkpoint_pth)
        runner.model.train()
        if device == 0: print(f"Resuming training from saved model checkpoint at Epoch {runner.epochs_ran}")
        if device == 0: logger.info(f"Resuming training from saved model checkpoint at Epoch {runner.epochs_ran}")
    elif LOAD_MODEL_CHECKPOINT and not os.path.exists(model_checkpoint_pth):
        if device == 0: logger.warning(f"The .pth file {model_checkpoint_pth} does not exist; there is no model to load. Ending program.")
        exit()

    if LOAD_BEST_MODEL and os.path.exists(best_model_pth):
        if device == 0: logger.info(f'Loading best model: {best_model_pth}')
        if device == 0: runner.model_pth = best_model_pth
        runner.load_model_parameters(runner.model_pth)
        if device == 0: print(f"Loading parameters from best saved model.")
        if device == 0: logger.info(f"Loading parameters from best saved model.")
    elif LOAD_BEST_MODEL and not os.path.exists(best_model_pth):
        if device == 0: logger.warning(f"The .pth file {best_model_pth} does not exist; there is no model to load. Ending program.")
        exit()

    runner.run(train_sampler=train_sampler, test_sampler=test_sampler,
               train_data = train_loader, test_data = test_loader,
               num_epochs = num_epochs, max_batch = n_test_baches)

if __name__=="__main__":
    main()
