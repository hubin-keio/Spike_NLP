"""
BLSTM model with FCN layer and BERT, utilizing the DMS binding or expression datasets.
"""

from pnlp.model.language import ProteinMaskedLanguageModel, BERT
from blstm import BLSTM

class BERT_BLSTM(nn.Module):
    """"
    BERT protein language model
    """

    def __init__(self, bert: BERT, vocab_size: int):
        super().__init__()
        self.bert = bert
        # self.next_amino_acid = NextAminoAcidPrediction(self.bert.hidden)  # Cannot use next word prediction in a BERT model.
        self.mlm = ProteinMaskedLanguageModel(self.bert.hidden, vocab_size)
        self.blstm = BLSTM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bert(x)
        x = self.mlm(x)
        x = self.blstm(x)

        return x
