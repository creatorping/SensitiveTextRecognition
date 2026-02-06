"""
Enhanced NER Model with Multiple Improvements
"""
import torch
import torch.nn as nn
from transformers import AutoModel
from config import Config
from crf_model import BiLSTMCRF


class EnhancedNERModel(nn.Module):
    """
    Enhanced NER Model with:
    - BERT encoder
    - BiLSTM layer
    - CRF layer
    - Multi-head attention
    - Residual connections
    """
    def __init__(self, config: Config, use_crf=True):
        super().__init__()
        self.config = config
        self.use_crf = use_crf

        # BERT encoder
        self.bert = AutoModel.from_pretrained(config.model_name)

        # Freeze lower layers of BERT for faster training
        if hasattr(config, 'freeze_bert_layers') and config.freeze_bert_layers > 0:
            for layer in self.bert.encoder.layer[:config.freeze_bert_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        # Dropout
        self.dropout = nn.Dropout(0.1)

        # Multi-head self-attention for context enhancement
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Dropout(0.1)
        )

        # BiLSTM-CRF for sequence labeling
        if self.use_crf:
            self.bilstm_crf = BiLSTMCRF(
                input_size=config.hidden_size,
                hidden_size=512,
                num_labels=len(config.entity_types) * 3 + 1,  # B/I/E for each type + O
                num_layers=2,
                dropout=0.1
            )
        else:
            # Simple classifier without CRF
            self.classifier = nn.Linear(config.hidden_size, len(config.entity_types) * 3 + 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch, seq_len] (optional, for training)
        Returns:
            if training: loss
            else: logits or decoded tags
        """
        # BERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]
        sequence_output = self.dropout(sequence_output)

        # Self-attention with residual connection
        attn_output, _ = self.self_attention(
            sequence_output, sequence_output, sequence_output,
            key_padding_mask=~attention_mask.bool()
        )
        sequence_output = self.layer_norm1(sequence_output + attn_output)

        # Feed-forward with residual connection
        ffn_output = self.ffn(sequence_output)
        sequence_output = self.layer_norm2(sequence_output + ffn_output)

        if self.use_crf:
            # BiLSTM-CRF
            emissions = self.bilstm_crf(sequence_output, mask=attention_mask.bool())

            if labels is not None:
                # Training: compute CRF loss
                loss = self.bilstm_crf.loss(emissions, labels, mask=attention_mask.bool())
                return loss
            else:
                # Inference: Viterbi decoding
                predictions = self.bilstm_crf.decode(emissions, mask=attention_mask.bool())
                return predictions
        else:
            # Simple classifier
            logits = self.classifier(sequence_output)

            if labels is not None:
                # Training: compute cross entropy loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                return loss
            else:
                return logits

    def get_embeddings(self):
        """Get word embeddings for adversarial training"""
        return self.bert.embeddings.word_embeddings.weight
