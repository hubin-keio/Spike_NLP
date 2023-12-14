""" BLSTM with FCN layer, MLM, and BERT. """

import torch
import torch.nn as nn
from pnlp.model.language import ProteinMaskedLanguageModel, BERT
from model.blstm import BLSTM

class BERT_BLSTM(nn.Module):
    """" BERT protein language model. """
    
    def __init__(self, bert: BERT, blstm:BLSTM, vocab_size: int):
        super().__init__()

        self.bert = bert
        self.mlm = ProteinMaskedLanguageModel(self.bert.hidden, vocab_size)
        self.blstm = blstm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bert(x)
        error_1 = self.mlm(x) # error from masked language
        error_2 = self.blstm(x) # error from regession

        return error_1, error_2