"""Protein Language Model Trainer"""

from pnlp.model.language import ProteinLM
from pnlp.model.bert import BERT
import numpy as np
import tqdm
import torch
import sqlite3
from os import path
from torch.utils.data import DataLoader
from pnlp.db.dataset import SeqDataset, initialize_db
from pnlp.embedding.tokenizer import ProteinTokenizer
from pnlp.embedding.nlp_embedding import NLPEmbedding

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
                 embedding_dim:int,           # BERT parameters
                 dropout: float,                 
                 max_len: int,             
                 mask_prob: float,
                 n_transformer_layers:int,
                 n_attn_heads: int,

                 batch_size: int,             # Learning parameters
                 lr: float=1e-4,
                 betas=(0.9, 0.999),
                 weight_decay: float=0.01,
                 warmup_steps: int=10000
                ):
        
        bert = BERT(embedding_dim, dropout, max_len, mask_prob, n_transformer_layers, n_attn_heads)
        mlm = ProteinLM(bert, 
        
        #init parameters 
        self.embedding_dim = embedding_dim
        self.vocab_dict = vocab_dict
        self.max_len = max_len
        self.n_transformer_layers = n_transformer_layers
        self.n_attn_heads = n_attn_heads
        self.dropout = dropout
        self.mask_prob = mask_prob
        self.batch_size = batch_size
        self.hidden = self.embedding_dim
        self.lr = lr 
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.bert = BERT(self.embedding_dim,
                     self.dropout,
                     self.max_len,
                     self.mask_prob,
                     self.hidden,
                     self.n_transformer_layers,
                     self.n_attn_heads)
        #get tokenizer
        self.tokenizer = ProteinTokenizer(self.max_len, self.mask_prob)
        #initialize model
        self.model = ProteinLM(self.bert, len(vocab_dict)).to(self.device)
        #set optimizer and scheduler
        self.optim = torch.optim.Adam(self.model.parameters(),lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.hidden, n_warmup_steps=self.warmup_steps)
        
        #set criterion to evaluate accuracy 
        self.criterion = torch.nn.CrossEntropyLoss()

        #set device 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
    def train(self, train_data, epoch: int=10):
        self.epoch_iteration(epoch, train_data)

    def test(self, test_data, epoch: int=10):
        self.epoch_iteration(epoch, test_data, train=False)
    
    def epoch_iteration(self, epoch, data_loader, train: bool=True):
        
        """
        Loop over dataloader for training or testing
        
        For training mode, backpropogation is activated 
        """
        
        mode = "train" if train else "test"
        
        # set the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                            desc=f'EP_{mode}: {epoch}',
                            total = len(data_loader),
                            bar_format='{l_bar}{r_bar}')
        
        running_loss = 0
        
        for i, batch_data in data_iter:
            
            seq_ids, seqs = batch_data
            
            tokenized_seqs, mask_idx = self.tokenizer.get_token(seqs)
            
            tokenized_seqs = tokenized_seqs.to(self.device)
            
            logits = self.model(tokenized_seqs)
            
            loss = self.criterion(logits.view(-1,logits.size(-1)), tokenized_seqs.view(-1))
            
            if train:
                self.optim.zero_grad()
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optim_schedule.step()
                
            running_loss += (loss.item() if running_loss is None else 0.99 * running_loss + 0.01 * loss.item())
            
        # torch.save(self.model.state_dict(), 'model_weights.pth')
        
        return running_loss
                
                
                
                
if __name__=="__main__":
    # Data loader
    db_file = path.abspath(path.dirname(__file__))
    db_file = path.join(db_file, '../../../data/SARS_CoV_2_spike.db')
    train_dataset = SeqDataset(db_file, "train")
    print(f'Sequence db file: {db_file}')
    print(f'Total seqs in training set: {len(train_dataset)}')
    
    embedding_dim = 24
    dropout=0.1
    max_len = 1500
    mask_prob = 0.15
    lr=0.0001

    tokenizer = ProteinTokenizer(max_len, mask_prob)
    embedder = NLPEmbedding(embedding_dim, max_len,dropout)

    vocab_size = len(tokenizer.token_to_index)
    padding_idx = tokenizer.token_to_index['<PAD>']
    hidden = embedding_dim
    n_transformer_layers = 12
    attn_heads = 12

    batch_size = 32
    num_workers = 1

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    n_test_baches = 5
    
    trainer = Plm_Trainer(BERT, vocab_dict=vocab_dict, max_len=max_len,
                          embedding_dim=embedding_dim, n_transformer_layers=n_transformer_layers,
                          n_attn_heads=attn_heads, dropout=dropout,
                          mask_prob=mask_prob, batch_size=batch_size,
                          lr=lr)
    
    trainer.train(train_data = train_loader)
