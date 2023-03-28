import io
import copy
import math
from Bio import SeqIO
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import pandas as pd
import altair as alt
import tqdm
from typing import Tuple, List, Dict
# from bert_pytorch.model import BERT

# ## Tokenization and Vocabulary
# In [ProteinBERT](https://academic.oup.com/bioinformatics/article/38/8/2102/6502274), Brandes et al used 26 unique tokens to represent the 20 standard amino acids, selenocysteine (U), and undefined amino acid (X), another amino acid (OTHER) and three speical tokens \<START\>, \<END\>, \<PAD\>.


# Based on the source code of protein_bert
# TODO: add a <TRUNCATED> token.
ALL_AAS = 'ACDEFGHIKLMNPQRSTUVWXY'
ADDITIONAL_TOKENS = ['<OTHER>', '<START>', '<END>', '<PAD>']

# Each sequence is added <START> and <END>. "<PAD>" are added to sequence shorten than max_len.
ADDED_TOKENS_PER_SEQ = 2

n_aas = len(ALL_AAS)
aa_to_token_index = {aa: i for i, aa in enumerate(ALL_AAS)}
additional_token_to_index = {token: i + n_aas for i, token in enumerate(ADDITIONAL_TOKENS)}
token_to_index = {**aa_to_token_index, **additional_token_to_index}
index_to_token = {index: token for token, index in token_to_index.items()}
n_tokens = len(token_to_index)

def tokenize_seq(seq: str, max_len: int = 1500) -> torch.Tensor:
    """
    Tokenize a sequence.

    It is the caller's responsibility to infer the maximum length of the input. In case of
    tokenizing a batch of sequences, the maximum length shall be assigned to the length of
    the longest sequence in the same batch.

    seq: input sequence
    max_len: maximum number of tokens, including the special tokens such as <START>, <END>.
    """
    seq = seq.upper()  # All in upper case.
    additional_token_to_index: dict = {"<START>": 0, "<END>": 1, "<PAD>": 2, "<OTHER>": 3}
    aa_to_token_index: dict = {}
    token_seq = [
        additional_token_to_index["<START>"]
    ] + [
        aa_to_token_index.get(aa, additional_token_to_index["<OTHER>"])
        for aa in seq
    ]
    if len(token_seq) < max_len - 1:  # -1 is for the <END> token
        n_pads = max_len - 1 - len(token_seq)
        token_seq.extend(
            [additional_token_to_index["<PAD>"]] * n_pads
        )
    token_seq += [additional_token_to_index["<END>"]]
    return torch.tensor(token_seq, dtype=torch.int64)


class TokenEmbedding(nn.Embedding):
    """Token embedding"""

    def __init__(
        self, vocab_size: int, embedding_dim: int = 512, padding_idx: int = 2
    ) -> None:
        super().__init__(vocab_size, embedding_dim, padding_idx)


class FastaDataset(Dataset):
    """Create Dataset compatible indexing of fasta file
    """
    def __init__(self, fasta_file: str, encoding_fn) -> None:
        self.sequences = list(SeqIO.parse(fasta_file, 'fasta'))
        self.encoding_fn = encoding_fn
        
    def __getitem__(self, idx):
        sequence = str(self.sequences[idx].seq)
        encoding = self.encoding_fn(sequence)
        return encoding
    
    def __len__(self):
        return len(self.sequences)


class FastaDataLoader:
    """Wrapper for fasta dataloader
    """
    def __init__(self, fasta_file: str, batch_size: int, encoding_fn, shuffle=True):
        self.dataset = FastaDataset(fasta_file, encoding_fn)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return iter(self.dataloader)


class PositionalEncoding(nn.Module):
    """
    Implement the PE function.

    The PE forward function is different from the BERT-pytorch. Here we used the original method in BERT so
    PE embeddings are added to the input embeddings and no gradient tracking is used.
    """

    def __init__(
        self, d_model: int, dropout: float = 0.1, max_len: int = 1500
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # adding positional embeddings on top of original embedding
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
    
class SeqEncoding(nn.Module):
    """
    Encode amino acid sequence. Input sequence is represented by summing the corresponding sequence token,
    segment (e.g. question and answer or any segments separated by <SEP>), and position embeddings. In our
    model, we only need the token and position embedding so segment embedding is not implemented here.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        dropout: float = 0.1,
        max_len: int = 1500,
        padding_idx: int = 25
    ) -> None:
        super().__init__()
        self.token_embedding: TokenEmbedding = TokenEmbedding(
            vocab_size, embedding_dim, padding_idx
        )
        self.add_position: PositionalEncoding = PositionalEncoding(
            embedding_dim, dropout, max_len
        )
        self.embedding_dim: int = embedding_dim
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.max_len: int = max_len

    def forward(self, seq: str) -> torch.Tensor:
        x: torch.Tensor = tokenize_seq(seq, self.max_len)
        x: torch.Tensor = self.token_embedding(x)
        x: torch.Tensor = self.add_position(x)
        return self.dropout(x)

    
class Attention(nn.Module):
    """Single head scaled dot product attention"""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
        dropout: nn.Dropout = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k: int = query.size(-1)
        scores: torch.Tensor = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # sqrt(d_k) is the scaling factor.

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn: torch.Tensor = scores.softmax(dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
    
    
class MultiHeadedAttention(nn.Module):
    """
    Multi-head attention

    h: number of heads
    d_model: model size

    """
    def __init__(
        self,
        h: int,
        d_model: int,
        n_linear: int = 4,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        assert d_model % h == 0  # d_model/h is used as d_k and d_v

        self.d_k: int = d_model // h
        self.h: int = h
        self.linears: nn.ModuleList = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(n_linear)]
        )  # n layers of linear model with the same input and output size
        self.output_linear: nn.Linear = nn.Linear(d_model, d_model)  # Output linear model. This implementation follows BERT-pytorch instead of using the last linear layer, which is found in the annotated transformer.
        self.attn: Attention = Attention()  # The forward function in Attention class is called since no hooks are defined in Attention class. See __call__() and _call_impl() in nn.Module implementation.

        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        if mask is not None:
            mask = mask.unsqueeze(1)  # same mask applied to all heads
        n_batches: int = query.size(0)

        # 1) Linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch
        x, attn = self.attn(query, key, value, mask=mask, dropout=self.dropout)  # Returned attn is not needed since x has already been weighted by attention in Attention.forward().

        # 3) "Concat using a view and apply a final linear"
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(n_batches, -1, self.h * self.d_k)
        )

        # del query
        # del key
        # del value
        return self.output_linear(x)


class LayerNorm(nn.Module):
    """
    Construct a layernorm module

    The normalization is a linear transformation of z-score. A small float
    number (eps) is added to std incase std is zero.

    """

    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.a_2: nn.Parameter = nn.Parameter(torch.ones(features))
        self.b_2: nn.Parameter = nn.Parameter(torch.zeros(features))
        self.eps: float = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean: torch.Tensor = x.mean(-1, keepdim=True)
        std: torch.Tensor = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size: int, dropout: float) -> None:
        super().__init__()
        self.norm: LayerNorm = LayerNorm(size)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        "Apply residual connection to any sublayer with the same size"
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.w_1: nn.Linear = nn.Linear(d_model, d_ff)
        self.w_2: nn.Linear = nn.Linear(d_ff, d_model)
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.activation: nn.GELU = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class TransformerBlock(nn.Module):
    """Transformer"""

    def __init__(self, hidden: int, attn_heads: int, feed_forward_hidden: int, dropout: float) -> None:
        """
        hidden: hidden size of transformer
        attn_heads: number of attention heads
        feed_forward_hidden: feed forward layer hidden size, usually 4 * hidden_size
        dropout: dropout ratio
        """

        super().__init__()
        self.attention: MultiHeadedAttention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward: PositionwiseFeedForward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer: SublayerConnection = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer: SublayerConnection = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """Produce N identical layers"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class BERT(nn.Module):
    """
    BERT model
    """

    def __init__(self, 
                 vocab_size: int=26,
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
        self.hidden: int = hidden
        self.n_transformer_layers: int = n_transformer_layers
        self.attn_heads: int = attn_heads

        # 4 * hidden_size for FFN
        self.feed_forward_hidden: int = hidden * 4

        # embeddings with sequence and postion
        self.embedding: SeqEncoding = SeqEncoding(vocab_size=vocab_size,
                                     embedding_dim=hidden,
                                     dropout=dropout,
                                     max_len=1500,
                                     padding_idx=padding_idx)

        self.transformer_blocks: nn.ModuleList = clones(TransformerBlock(hidden, 
                                                          attn_heads, 
                                                          self.feed_forward_hidden,
                                                          dropout), n_transformer_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        mask: torch.BoolTensor = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        
        x: torch.Tensor = self.embedding(x)   # sequence and position embedding in one step.

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.linear(x))


class ProteinLM(nn.Module):
    """"
    BERT protein language model
    """

    def __init__(self, bert: BERT, vocab_size: int):
        super().__init__()
        self.bert = bert
        # self.next_amino_acid = NextAminoAcidPrediction(self.bert.hidden)  # Cannot use next word prediction in a BERT model.
        self.mlm = ProteinMaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bert(x)
        return self.mlm(x)

 
# ## Model Training


class ScheduledOptim():
    """A simple wrapper class for learning rate scheduling."""

    def __init__(self, optimizer, d_model: int, n_warmup_steps: int):
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
                 betas: Tuple[float, float]=(0.9, 0.999),
                 weight_decay: float=0.01,
                 warmup_steps: int=10000,
                 with_cuda: bool = True,
                 cuda_device: Optional[Union[int, str]] = None,
                 log_freq: int = 10
                 ) -> None:
        
        # Use CUDA device if it is available and with_cuda is True
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")

        if cuda_condition:
            if cuda_device is None:
                self.device = torch.device("cuda:0")
            elif isinstance(cuda_device, int):
                self.device = torch.device(f"cuda:{cuda_device}")
            elif isinstance(cuda_device, str):
                self.device = torch.device(cuda_device)
            else:
                raise ValueError("Invalid value passed for `cuda_device` argument.")

        # Distributed GPU training if more than one CUDA device is detected.
        if with_cuda and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for BERT.")

        # This BERT model will be saved every epoch
        self.bert = bert
        self.model = ProteinMaskedLanguageModel(bert.hidden, vocab_size).to(self.device)

        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hpyer-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # Using negative log likelyhood loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0) #TODO: check if ignore_index should be set differently.
        
        self.log_freq = log_freq

        print(f'Total parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')

    def train(self, epoch: int = 10) -> None:
        self.iteration(epoch, self.train_data)

    def test(self, epoch: int = 10) -> None:
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch: int, data_loader: DataLoader, train: bool = True) -> None:
        """
        Loop over the data_loader for training or testing.

        If on train status, backward operation is activated and also auto save the model every epoch.
        """
        str_code = "train" if train else "test"

        # set the tqdm progress bar
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc=f'EP_{str_code}: {epoch}',
            total=len(data_loader),
            bar_format='{l_bar}{r_bar}'
        )

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}

            #TODO: get the masked sequence out.
            next_aa_predicted, mlm_predicted = self.model.forward(data['bert_input'], ...)