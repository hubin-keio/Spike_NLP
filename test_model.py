# %% [markdown]
# # Spike NLP
# 
# We wish to utilize NLP methods to analyze the virus protein sequences. After initial experiment with the LSTM architecture, we decided to use Transformer architecture. In this notebook, we implement a BERT model. During the process, we learned from existing implementations of BERT, especially [BERT-pytorch](https://github.com/codertimo/BERT-pytorch) and [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) and [ProteinBERT](https://academic.oup.com/bioinformatics/article/38/8/2102/6502274), though we have to make changes to accomodate our own research interests. For example, we are interested in next word prediction in the pre-training of the model to generate contexualized embeddings using self-supervised learning at individual amino acid level through learning the language patterns but we are not interested in protein functional annotation, so we do not use annotation in our model as a pre-training task. In other words, we are interested in leveraing the Mask LM pre-training task to derive the embeddings for a fine-tuning model to  predict the phenotype of the virus protein sequences. Example phenotypes are bind binding kinetics of the virus protein to target receptor proteins and antibodies.

# %% [markdown]
# 
# 
# ## Embedding
# In the BERT implemetnation (bert_pytorch/model/bert.py), the masking is done after the second token (x>0) since in the original BERT paper, the first element of the input is always \[CLS\]. In our model, we will use the variant name as the \[CLS\] and the values are:
# [wt, alpha, delta, omicron, na], where "na" stands for not assigned.

# %%
import io
import os
import copy
import math
from Bio import SeqIO
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.optim import Adam
import pandas as pd
import altair as alt
import sqlite3
import tqdm
import random

# from bert_pytorch.model import BERT

# %% [markdown]
# ## Tokenization and Vocabulary
# In [ProteinBERT](https://academic.oup.com/bioinformatics/article/38/8/2102/6502274), Brandes et al used 26 unique tokens to represent the 20 standard amino acids, selenocysteine (U), and undefined amino acid (X), another amino acid (OTHER) and three speical tokens \<START\>, \<END\>, \<PAD\>.

# %%
# Based on the source code of protein_bert
# "<TRUNCATED>" is used if the longest sequence in a batch is longer than the maximum length
# used as inputs (defined as max_len in tokenize_seq). "<MASK>" is used in masked language model.
ALL_AAS = 'ACDEFGHIKLMNPQRSTUVWXY'
ADDITIONAL_TOKENS = ['<OTHER>', '<START>', '<END>', '<PAD>', '<MASK>', '<TRUNCATED>']

# Each sequence is added <START> and <END>. "<PAD>" are added to sequence shorten than max_len.
ADDED_TOKENS_PER_SEQ = 2

n_aas = len(ALL_AAS)
aa_to_token_index = {aa: i for i, aa in enumerate(ALL_AAS)}
additional_token_to_index = {token: i + n_aas for i, token in enumerate(ADDITIONAL_TOKENS)}

token_to_index = {**aa_to_token_index, **additional_token_to_index}
index_to_token = {index: token for token, index in token_to_index.items()}
n_tokens = len(token_to_index)

def tokenize_seq(seq: str, max_len:int=1500) -> torch.IntTensor:
    """
    Tokenize a sequence.

    It is the caller's responsibility to infer the maximum length of the input. In case of
    tokenizing a batch of sequences, the maximum length shall be assigned to the lenght of
    the longest sequence in the same batch. 


    seq: input insquence
    max_len: maximum number of tokens, including the special tokens such as <START>, <END>.
    
    """
    seq = seq.upper()   # All in upper case.
    other_token_index = additional_token_to_index['<OTHER>']
    token_seq = [additional_token_to_index['<START>']] + [aa_to_token_index.get(aa, other_token_index) for aa in seq]
    if len(token_seq) < max_len - 1: # -1 is for the <END> token
        n_pads = max_len -1 - len(token_seq)
        token_seq.extend(token_to_index['<PAD>'] for _ in range(n_pads))
    token_seq += [additional_token_to_index['<END>']]
    return torch.IntTensor(token_seq)
token_df = pd.DataFrame(token_to_index.items(), columns=['Token', 'Index'])
print(token_df)


# %% [markdown]
# ## Amino Acid Token Embeddings
# We will derive token embedding from the [torch.nn.Embedding class](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html). The size of the vacabulary equals the number of tokens. This approach allows the learning of the embeddings from the model intself. If we train the model with virus sepcific squences, the embeddings shall reflect the hidden properties of the amino acids in context of the trainign sequences. Note that the \<START\> and \<END\> tokens are always added at the beginning of the sequence. \<PAD\> tokens may be added before the \<END\> token if the sequence is shorter than the input sequence.
# 
# Note that using the "from_pretrained" class method of torch.nn.Embedding, we can load pre-trained weights of the embedding.

# %%
class TokenEmbedding(nn.Embedding):
    """Token embedding"""
    def __init__(self, vocab_size: int,
                 embedding_dim: int=512,
                 padding_idx=None):
        super().__init__(vocab_size, embedding_dim, padding_idx)

padding_idx = token_to_index['<PAD>']

#TODO: add support to load pre-trained embeddings.


# %% [markdown]
# ## Postional Encoding
# We will use the  sine and cosine functions of different frequencie to embed positional information as in the original BERT method.

# %%
class PositionalEncoding(nn.Module):
    """
    Impement the PE function.
    
    The PE forward function is different from the BERT-pytorch. Here we used the original method in BERT so
    PE embeddings are added to the input embeddings and no graident tracking is used.
    """

    def __init__(self,
                 d_model: int,       # model input dimension
                 dropout: float=0.1, # dropout rate
                 max_len=1500):      # maximum sequence length #TODO: need a truncation and the <truckated> token.
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # adding positional embeddings on top of original embedding
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

# %%
class SeqEncoding(nn.Module):
    """
    Encode amino acid sequence. Input sequence is represented by summing the corresponding sequence token,
    segment (e.g. question and answer or any segments separated by <SEP>), and position embeddings. In our 
    model, we only need the token and position embedding so segment embeddign is not implemented here.    
    """
    def __init__(self,
                 vocab_size: int,       # vocabulary size
                 embedding_dim: int,    # embedding dimensions
                 dropout: float=0.1,    # dropout rate
                 max_len: int=1500,     # maximum length of input sequence
                 padding_idx: int=25):  # padding token index
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim, padding_idx)
        self.add_position = PositionalEncoding(embedding_dim, dropout, max_len)
        self.embeddng_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len
        
    def forward(self, seq:str):
        x = tokenize_seq(seq, self.max_len)
        x = self.token_embedding(x)
        print(x.shape)
        x = self.add_position(x)
        return self.dropout(x)

# %% [markdown]
# ## Test Sequence and Position Embedding
# 
# Let's test the embedding of the first 28 amino acids of the test sequence. Notice that position 2 and 4 are the same amino acid (F) yet they have different emedding in every dimension due to they appear at different positions. For simplicity, we only use 6 dimensions to embed the sequence. In the actual model, we will use many more dimensions.

# %%
test_wt_seq = """>sp|P0DTC2|SPIKE_SARS2 Spike glycoprotein OS=Severe acute respiratory syndrome coronavirus 2 OX=2697049 GN=S PE=1 SV=1
MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFS
NVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIV
NNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLE
GKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQT
LLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETK
CTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISN
CVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIAD
YNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPC
NGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVN
FNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITP
GTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSY
ECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTI
SVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQE
VFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDC
LGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAM
QMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALN
TLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRA
SANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPA
ICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDP
LQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDL
QELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDD
SEPVLKGVKLHYT"""
len(test_wt_seq)

# %%
test_seqs = []
fa_parser = SeqIO.parse(io.StringIO(test_wt_seq), 'fasta')
for record in fa_parser:
    seq = record.seq
    test_seqs.append(str(seq))

# %%
def test_encoding():
    embedding_dim = 6
    dropout = 0.1
    padding_idx = 25

    max_len = 30    # test only the first 30 aas.
    seq = test_seqs[0][:max_len-2]    

    test_seq_encode = SeqEncoding(n_tokens, embedding_dim, dropout, max_len, padding_idx)
    test_pe_encode = PositionalEncoding(embedding_dim, dropout, max_len)
    y = test_pe_encode.forward(test_seq_encode(seq))
    print(f'Embedding shape: {y.shape}')
    print(f'Parameters shape in sequence embedding: {test_seq_encode.token_embedding.weight.shape}')

    y = y.detach().numpy()


    data = pd.concat([pd.DataFrame({
        "embedding": y[:, dim],
        "dimension":dim,
        "position": list(range(max_len)),
        })for dim in range(6)])
    
    aa = ['<start>']
    aa.extend(_ for _ in seq)
    aa.append('<end>')
    data['aa_token'] = aa * embedding_dim

    print(data)

    return (
        alt.Chart(data)
        .mark_line()
        .properties(width=800)
        .encode(x="position", y="embedding", color="dimension:N")
        .interactive())

test_encoding()


# %% [markdown]
# ## Attention

# %%
class Attention(nn.Module):
    """Single head scaled dot product attention"""
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)    #sqrt(d_k) is the scaling factor.

        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        
        p_attn = scores.softmax(dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
    
        return torch.matmul(p_attn, value), p_attn 

# %%
class MultiHeadedAttention(nn.Module):
    """
    Multi-head attention

    h: numer of heads
    d_model: model size
    
    """
    def __init__(self, h:int, d_model:int, n_linear: int=4, dropout=0.1):
        super().__init__()
        assert d_model % h == 0 # d_model/h is used as d_k and d_v

        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_linear)])  # n layers of linear model with the same input and output size
        self.output_linear = nn.Linear(d_model, d_model)    # Output lienar model. This implementation follows BERT-pytorch instead of using the last linear layer, which is found in the annotated transformer.
        self.attn = Attention() # The forward function in Attention class is called since no hooks are defined in Attention class. See __call__() and _call_impl() in nn.Module implementation.

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)    # same mask applied to all heads
        n_batches = query.size(0)

        # 1) Linear projections in batch from d_model => h x d_k
        query, key, value = [lin(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2)
                             for lin, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch
        x, attn = self.attn(query, key, value, mask=mask, dropout = self.dropout)   # Returned attn is not needed since x has already been weighted by attention in Attention.forward().

        # 3) "Concat using a view and apply a final linear"
        x = (x.transpose(1, 2)
             .contiguous()
             .view(n_batches, -1, self.h * self.d_k))
        
        # del query
        # del key
        # del value
        return self.output_linear(x)

# %% [markdown]
# ## Layer Normalization
# 
# Linear regression based layer normalization with parameters a_2 and b_2. An arbituary small value (epsilon or eps) is added to std to avoid the error when std is 0.

# %%
class LayerNorm(nn.Module):
    """
    Construct a layernorm module
    

    The normalization is a linear transformation of z-score. A small float
    number (eps) is added to std incase std is zero.
    
    """

    def __init__(self, features: torch.tensor, eps: float=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2

# %% [markdown]
# ## Residual Connection

# %%
class SublayerConnection(nn.Module):
    """A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size"
        return x + self.dropout(sublayer(self.norm(x)))

# %% [markdown]
# ## Positionwise Feed Forward

# %%
class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

# %% [markdown]
# ## Transformer

# %%
class TransformerBlock(nn.Module):
    """Transformer"""

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        hidden: hidden size of transformer
        attn_heads: number of attention heads
        feed_forward_hidden: feed forward layer hidden size, usually 4 * hidden_size
        dropout: dropout ratio
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)

# %% [markdown]
# # BERT
# 
# Here we define a model based on BERT. Part of the implementation is based on [BERT-pytorch](https://github.com/codertimo/BERT-pytorch)

# %%
def clones(module, n):
    """Produce N identical layers"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

# %%
class BERT(nn.Module):
    """
    BERT model
    """

    def __init__(self, 
                 vocab_size: int=27,
                 padding_idx: int=25,
                 hidden: int=768, 
                 n_transformer_layers: int=12, 
                 attn_heads: int=12,
                 dropout: float=0.1):
        """
        vacab_size: vacabulary or token size
        hidden: BERT model size (used as input size and hidden size)
        n_layers: number of Transformer layers
        attn_heads: attenion heads
        dropout: dropout ratio
        """

        super().__init__()
        self.hidden  = hidden
        self.n_transformer_layers = n_transformer_layers
        self.attn_heads = attn_heads

        # 4 * hidden_size for FFN
        self.feed_forward_hidden = hidden * 4

        # embeddings with sequence and postion
        self.embedding = SeqEncoding(vocab_size=vocab_size,
                                     embedding_dim=hidden,
                                     dropout=dropout,
                                     max_len=1500,
                                     padding_idx=padding_idx)

        self.transformer_blocks = clones(TransformerBlock(hidden, 
                                                          attn_heads, 
                                                          self.feed_forward_hidden,
                                                          dropout), n_transformer_layers)

    def forward(self, x: torch.Tensor, mask):

        # mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        
        x = self.embedding(x)   # sequence and position embedding in one step.

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask=mask)

        return x

# %%
#num_parameters_seq_encoding = sum(p.numel() for p in test_seq_encode.parameters() if p.requires_grad)
#print(f'Parameters in SeqEncoding: {num_parameters_seq_encoding}')

# %% [markdown]
# ## BERT-based Protein Language Model

# %%
class ProteinMaskedLanguageModel(nn.Module):
    """Masked language model for protein sequences"""

    def __init__(self, hidden: int, vocab_size: int):
        """
        hidden: input size of the hidden linear layers
        vocab_size: vocabulary size
        """

        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))

# %%
class ProteinLM(nn.Module):
    """"
    BERT protein language model
    """

    def __init__(self, bert: BERT, vocab_size):
        super().__init__()
        self.bert = bert
        # self.next_amino_acid = NextAminoAcidPrediction(self.bert.hidden)  # Cannot use next word prediction in a BERT model.
        self.mlm = ProteinMaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x, mask=None):
        x = self.bert(x, mask)
        return self.mlm(x)

# %% [markdown]
# ## Model Training

# %%
class ScheduledOptim():
    """A simple wrapper class for learning rate scheduling."""

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

# %% [markdown]
# # Data Loader
# Now we build a Dataloader that reads fasta files of the protein sequences. Since our purpose now is pre-training, we will not need any phenotype data. Dataloader class to read in fasta file and return encoded sequence at index

# %% [markdown]
# ## Build a database of the training and testing data set

# %%
def initialize_db(db_file_path: str, train_fasta: str, test_fasta: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_file_path)
    cur = conn.cursor()
    
    cur.execute('''CREATE TABLE train (id INTEGER PRIMARY KEY AUTOINCREMENT, header TEXT, sequence TEXT)''')
    cur.execute('''CREATE TABLE test (id INTEGER PRIMARY KEY AUTOINCREMENT, header TEXT, sequence TEXT)''')
    
    training_seqs = SeqIO.parse(open(train_fasta),'fasta')

    for i, fasta in enumerate(training_seqs):
        header, seq = fasta.id, str(fasta.seq)
        cur.execute("INSERT INTO train (header, sequence) VALUES (?,?)", (header, seq))
    conn.commit()

    test_seqs = SeqIO.parse(open(test_fasta), "fasta")

    for i, fasta in enumerate(test_seqs):
        header, seq = fasta.id, str(fasta.seq)
        cur.execute("INSERT INTO test (header, sequence) VALUES (?,?)", (header, seq))
    conn.commit()    

    print(f'Database {db_file_path} initialized.')
    return conn

# Initialize database if it does not exist already.
db_file_path = "../data/SARS_CoV_2_spike.db"
training_set_fasta = "../data/spikeprot0203.clean.uniq.training.fasta"
testing_set_fasta = "../data/spikeprot0203.clean.uniq.testing.fasta"
if os.path.isfile(db_file_path):
    conn = sqlite3.connect(db_file_path)
else:
    conn = initialize_db(db_file_path, training_set_fasta, testing_set_fasta)
                         
                         

# %%
class SeqDataset(Dataset):
    """
    Create Dataset compatible indexing of fasta file

    db_file: sqlite3 database file
    table_name: table name inside the sqlite database

    """
    def __init__(self, db_file: str, table_name: str) -> None:
        self.db_file = db_file
        self.table = table_name
        self.conn = None    # Use lazy loading
                
    def __getitem__(self, idx):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_file, isolation_level=None)  # Read only operations in sqlite connection.

        cur = self.conn.cursor()
        _, header, sequence = cur.execute(f'''SELECT * FROM {self.table} LIMIT 1 OFFSET {idx}''').fetchone()
        # print(f'idx: {idx}, header: {header}, first 30 aas: {seq[:30]}')
        
        return header, sequence

    def __len__(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_file, isolation_level=None)  # Read only operations in sqlite connection.

        cur = self.conn.cursor()
        total_seq = cur.execute(f'''SELECT COUNT(*) as total_seq FROM {self.table}''').fetchone()[0]
        return total_seq

# %%
train_dataset = SeqDataset(db_file_path, "train")
print(f'Total seqs in training set: {len(train_dataset)}')

batch_size = 5
shuffle = True
num_workers = 1

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
n = 5
for i, batch in enumerate(train_loader):
    if i >= n:
        break
    print(f'batch {i}: {batch[0]}')


# %% [markdown]
# ## Now add masks
# 
# We will mask 15% of amino acid in sequence like in the original model.

# %%
# def mask_sequence(sequence:str, max_len:int, mask_prob:float=0.15):
#     """
#     add masks to sequence with assigned probabilities.
#     """
#     token_indices = tokenize_seq(sequence, max_len)

#     len_seq = len(token_indices)
#     n_mask = max(int(mask_prob * len_seq), 1) # at least one mask
#     masked_token_idx = []
#     for _ in range(n_mask):
#         idx = int(random.random() * len_seq)
#         if idx > 0 and idx != len_seq: # avoiding mask <START> <END>
#             masked_token_idx.append(idx)
#             token_indices[idx] = token_to_index['<MASK>']

#     return token_indices, masked_token_idx

def batch_pad(seqs:tuple, max_len: int,  token_dict: dict) -> torch.Tensor:
    """
    Tokenize a sequence batch and add padding when needed.

    Returns a tuple of sequence ids and tokenized tensor of the shape [batch_size, longest_seq].
    If the longest sequence in the batch is longer then max_len, the shape is [batch_size, max_len].

    Arguments:

    seqs: a tuple or list-like sequence collection
    max_len: maximum length of the input squence
    token_dict: token to index dictionary
    """
    tokenized_seqs = []
    for a_seq in seqs:
        if len(a_seq) < max_len:
            a_seq = [aa_to_token_index.get(aa, token_dict['<OTHER>']) for aa in a_seq]
        else:                   # if more  then max_len, we will need to truncate it and mark it <TRUNCATED>
            a_seq = [aa_to_token_index.get(aa, token_dict['<OTHER>']) for aa in a_seq[:max_len-1]]
            a_seq.append(token_dict['<TRUNCATED>'])
        tokenized_seqs.append(a_seq)

    max_in_bach = max([len(a_seq) for a_seq in tokenized_seqs])  # length of longest sequence in batch
    for _, seq in enumerate(tokenized_seqs):
        n_pad = max_in_bach - len(seq)
        if n_pad > 0:
            for p in range(n_pad):
                seq.append(padding_idx)
            tokenized_seqs[_] = seq

    tokenized_seqs = torch.tensor(tokenized_seqs)

    return tokenized_seqs

def batch_mask(batch_seqs: torch.Tensor, mask_prob:float, token_dict:dict) -> torch.Tensor:
    """
    Add <MASK> token at random locations based on assigned probabilities.

    Locations of the <MASK> are the same across one batch with the exception that 
    speical tokens, e.g. <PAD> <TRUNCATED> are not masked. Returns a tuple of
    (seq_ids, masked sequence tokens, original tokens)

    Parameters:

    batch_input: tensor of tokenized sequences with the shape [batch_size, seq_lenght]
    mask_prob: masking probability.
    token_dict: token to index dictionary
    """
    batch_masked = batch_seqs.clone()
    seq_len = batch_seqs.size(1)
    
    n_mask = max(int(mask_prob * seq_len), 1) # at least one mask

    row_idx = range(batch_seqs.size(0))

    SPECIAL_TOKENS = [token_dict[st] for st in ['<PAD>', '<TRUNCATED>']]
    MASK_IDX = token_dict['<MASK>']
    
    for _ in range(n_mask):
        idx = int(random.random() * seq_len)
        for row in row_idx:
            if batch_masked[row, idx] not in SPECIAL_TOKENS:
                batch_masked[row, idx] = torch.tensor(MASK_IDX)
    return batch_masked


test_seq = 'MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFT'
max_len = 36

original_tk = [index_to_token[int(idx)] for idx in tokenize_seq(test_seq, max_len)]
masked_tokenized, masked_token_idx = mask_sequence(test_seq, max_len)

masked_tk = [index_to_token[int(idx)] for idx in masked_tokenized]

print(f'Origianl tokens: {original_tk}')
print(f'With added masks: {masked_tk}')
print(f'Masked token indices: {masked_token_idx}')
print(f'Masked tokens: {[original_tk[idx] for idx in masked_token_idx]}')

# %% [markdown]
# # Test Model Execution
# 
# Now that we have all the building blocks of the model, we can test the model execution by loading a few batch of example sequences to train the masked language model. The testing process will be very smiliar except that there is no model parameter updates.
# 
# ## Model execution steps
# 
# 1. Load training data
# We will define the training data using the SeqDataset class and sepecify we want to use the sequences from the "train" table. Example code is avaiable in the train_dataset above.
# 
# 2. Mask the batched training seqeunce
# The mask_sequence function defines how masks are added. A helper function can be added to load bached sequence from SeqDataset, add masks, and return both the masked sequence and that amino acids at masked locations.
# 
# 3. Feed the masked sequence batch to the protein language model.
# The masked sequence will be embedded as vocabularies and feed to the forward function in the protein language model, which will initialize a BERT model inside it. The hidden BERT model includes  Transformer layers with multihead self-attention and position embedding. Status of the last Transformer will used as inputs to the feed forward layers. The last feedforward layer will be used to feed the linear layer in the ProteinMaskedLanguageModel, where a softmax function is used to predict the masked tokens.
# 
# 4. The cross entropy error function is used to caculate the error of the masked token prediction per batch. The goal of the training process is to optimize the parameters so that the error is minimized. After reaching the training goal (number of maximum epochs or the average error), we will need to save the model stataus so that we can reload the model with the trained parameters.

# %%
def test_model():
    vocab_size = n_tokens
    embedding_dim = 24
    padding_idx = token_to_index['<PAD>']

    bert_hidden = embedding_dim
    n_transformer_layers = 12
    n_attn_heads = 12
    dropout = 0.1 

    bert = BERT(vocab_size, padding_idx, bert_hidden, n_transformer_layers, n_attn_heads, dropout)
    model = ProteinLM(bert, vocab_size)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters in the model: {n_params}') #TODO: This does not include the parameters in the embedding process.

    db_file_path =  "../data/SARS_CoV_2_spike.db"
    train_dataset = SeqDataset(db_file_path, "train")
    print(f'Total seqs in training set: {len(train_dataset)}')

    batch_size = 5
    num_workers = 1
    max_len = 50
    mask_prob = 0.15

    embed_tokens = SeqEncoding(vocab_size, embedding_dim, dropout, max_len, padding_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    n_test_baches = 5
    for i, batch in enumerate(train_loader):
        if i >= n_test_baches:
            break
        print(f'\nbatch {i}')
        [seq_ids, seqs] = batch
        print(seq_ids)
        for _ in seqs:
            print(_)

        #batch_embedding = torch.empty((batch_size, max_len, embedding_dim))
        #for i in range(batch_size):
            # token_indices, masked_token_idx = mask_sequence(seqs[i], max_len, mask_prob)
            # embeddings = embed_tokens(token_indices)

            #embeddings = embed_tokens(seqs[i])
            #batch_embedding[i] = embeddings
        
        batch_mask = None

        #print(f'Batch embedding shape: {batch_embedding.shape}')    # Shape: (batch_size, max_len, embedding_dim), e.g. [5, 1500, 20]
        model(batch[0][1], batch_mask)


test_model()
    

# %%
class BPTrainer:
    """
    Model trainer

    Pretrain BERT Protein model with the masked language model.
    """

    def __init__(self, 
                 bert: BERTProtein,
                 vocab_size: int,
                 train_dataloader: DataLoader,
                 test_dataloader: DataLoader = None,
                 lr: float=1e-4,
                 betas=(0.9, 0.999),
                 weight_decay: float=0.01,
                 warmup_steps: int=10000,
                 with_cuda: bool = True,
                 cuda_device = None,
                 log_freq: int = 10
                 ):
        
        # Use CUDA device if it is available and with_cuda is Truegb
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # Distributed GPU training if more than one CUDA device is detected.
        if with_cuda and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for BERT.")

        # This BERT model will be saved every epoch
        self.bert = bert
        self.model = ProteinMaskedLanguageModel(bert, vocab_size).to(self.device)

        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hpyer-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # Using negative log likelyhood loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0) #TODO: check if ignore_index should be set differently.
        
        self.log_freq = log_freq

        print(f'Total parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')

    def train(self, epoch: int=10):
        self.iteration(epoch, self.train_data)

    def test(self, epoch: int=10):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train: bool=True):
        """
        Loop over the data_loader for training or testing.

        If on train status, backward operation is activated and also auto save the model every epoch.
        """
        str_code = "train" if train else "test"

        # set the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc=f'EP_{str_code}: {epoch}',
                              total = len(data_loader),
                              bar_format='{l_bar}{r_bar}')

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}

            #TODO: get the masked sequence out.
            next_aa_predicted, mlm_predicted = self.model.forward(data['bert_input'], ...) 




        

        


