"""BERT Model"""


import copy
import torch
import torch.nn as nn

from pnlp.model.transformer import TransformerBlock

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

        self.transformer_blocks = nn.ModuleList([copy.deepcopy(
            TransformerBlock(hidden, attn_heads, self.feed_forward_hidden, dropout)) for
            _ in n_transformer_layers])

    def forward(self, x: torch.Tensor, mask):

        # mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x = self.embedding(x)   # sequence and position embedding in one step.
        # x = TokenEmbedding(self.vocab_size, self.embedding_dim, self.padding_idx)
        print(f'Embedded x with shape:: {x.shape}')

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask=mask)

        return x
