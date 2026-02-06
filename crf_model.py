"""
Improved NER Model with CRF Layer
"""
import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF
from config import Config


class BiLSTMCRF(nn.Module):
    """BiLSTM-CRF layer for sequence labeling"""
    def __init__(self, input_size, hidden_size, num_labels, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, input_size]
            mask: [batch, seq_len] attention mask
        Returns:
            emissions: [batch, seq_len, num_labels]
        """
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)
        return emissions

    def loss(self, emissions, tags, mask):
        """
        Compute CRF loss
        Args:
            emissions: [batch, seq_len, num_labels]
            tags: [batch, seq_len]
            mask: [batch, seq_len]
        """
        return -self.crf(emissions, tags, mask=mask, reduction='mean')

    def decode(self, emissions, mask):
        """
        Viterbi decoding
        Args:
            emissions: [batch, seq_len, num_labels]
            mask: [batch, seq_len]
        Returns:
            best_paths: List[List[int]]
        """
        return self.crf.decode(emissions, mask=mask)
