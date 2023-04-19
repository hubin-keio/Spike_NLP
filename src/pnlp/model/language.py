"""Protien Language Models"""

import torch
import torch.nn as nn

from pnlp.model.bert import BERT

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
        x = self.softmax(self.linear(x))
        return x

# %%
class ProteinLM(nn.Module):
    """"
    BERT protein language model
    """

    def __init__(self, bert: BERT, vocab_size: int):
        super().__init__()
        self.bert = bert
        self.mlm = ProteinMaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x):
        x = self.bert(x)
        return self.mlm(x)
