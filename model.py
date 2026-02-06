"""
Nested Privacy Entity Recognition Model with Biaffine Attention
"""
import torch
import torch.nn as nn
from transformers import AutoModel
from config import Config


class BiaffineAttention(nn.Module):
    """Biaffine attention for span classification"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Biaffine transformation: U = x1^T W x2 + b
        self.W = nn.Parameter(torch.Tensor(out_features, in_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.bias)

    def forward(self, x1, x2):
        """
        Args:
            x1: [batch, seq_len, hidden] - start representations
            x2: [batch, seq_len, hidden] - end representations
        Returns:
            scores: [batch, seq_len, seq_len, out_features]
        """
        batch_size, seq_len, hidden = x1.size()

        # x1: [batch, seq_len, 1, hidden]
        # x2: [batch, 1, seq_len, hidden]
        x1 = x1.unsqueeze(2)
        x2 = x2.unsqueeze(1)

        # Biaffine transformation
        # [batch, seq_len, seq_len, out_features]
        scores = torch.einsum('bnih,oij,bnjh->bnmo', x1, self.W, x2)
        scores = scores + self.bias

        return scores


class SpanRepresentation(nn.Module):
    """Generate span representations from token embeddings"""
    def __init__(self, hidden_size, span_hidden_size):
        super().__init__()
        self.start_proj = nn.Linear(hidden_size, span_hidden_size)
        self.end_proj = nn.Linear(hidden_size, span_hidden_size)
        self.span_proj = nn.Linear(hidden_size * 2, span_hidden_size)
        self.dropout = nn.Dropout(0.3)  # Increased for better regularization

    def forward(self, sequence_output, span_positions):
        """
        Args:
            sequence_output: [batch, seq_len, hidden]
            span_positions: [batch, num_spans, 2] - (start, end) positions
        Returns:
            span_reps: [batch, num_spans, span_hidden_size]
        """
        batch_size, num_spans, _ = span_positions.size()

        # Extract start and end token representations
        start_indices = span_positions[:, :, 0]  # [batch, num_spans]
        end_indices = span_positions[:, :, 1]    # [batch, num_spans]

        # Gather representations
        start_reps = torch.gather(
            sequence_output,
            1,
            start_indices.unsqueeze(-1).expand(-1, -1, sequence_output.size(-1))
        )  # [batch, num_spans, hidden]

        end_reps = torch.gather(
            sequence_output,
            1,
            end_indices.unsqueeze(-1).expand(-1, -1, sequence_output.size(-1))
        )  # [batch, num_spans, hidden]

        # Concatenate and project
        span_reps = torch.cat([start_reps, end_reps], dim=-1)
        span_reps = self.span_proj(span_reps)
        span_reps = self.dropout(torch.relu(span_reps))

        return span_reps


class NestedPrivacyNER(nn.Module):
    """Nested Privacy Entity Recognition Model"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # BERT encoder
        self.bert = AutoModel.from_pretrained(
            config.model_name,
            local_files_only=True,
            trust_remote_code=True
        )

        # Enable gradient checkpointing to save memory
        if config.use_gradient_checkpointing:
            self.bert.gradient_checkpointing_enable()

        self.dropout = nn.Dropout(0.3)  # Increased for better regularization

        # Span representation
        self.span_repr = SpanRepresentation(config.hidden_size, config.span_hidden_size)

        # Biaffine attention for nested entity detection
        self.start_proj = nn.Linear(config.hidden_size, config.biaffine_size)
        self.end_proj = nn.Linear(config.hidden_size, config.biaffine_size)
        self.biaffine = BiaffineAttention(config.biaffine_size, len(config.entity_types) + 1)

        # Span classifier with stronger regularization
        self.span_classifier = nn.Sequential(
            nn.Linear(config.span_hidden_size, config.span_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),  # Increased for better regularization
            nn.Linear(config.span_hidden_size, len(config.entity_types) + 1)
        )

        # Multi-scale CNN for local context
        self.cnn_layers = nn.ModuleList([
            nn.Conv1d(config.hidden_size, config.hidden_size // 4, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]
        ])
        self.cnn_proj = nn.Linear(config.hidden_size // 4 * 3, config.hidden_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights to prevent NaN"""
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask, span_positions=None):
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            span_positions: [batch, num_spans, 2]
        Returns:
            logits: [batch, num_spans, num_classes] or [batch, seq_len, seq_len, num_classes]
        """
        # BERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]
        sequence_output = self.dropout(sequence_output)

        # Multi-scale CNN for local context
        cnn_input = sequence_output.transpose(1, 2)  # [batch, hidden, seq_len]
        cnn_outputs = [torch.relu(conv(cnn_input)) for conv in self.cnn_layers]
        cnn_output = torch.cat(cnn_outputs, dim=1)  # [batch, hidden//4*3, seq_len]
        cnn_output = cnn_output.transpose(1, 2)  # [batch, seq_len, hidden//4*3]
        cnn_output = self.cnn_proj(cnn_output)  # [batch, seq_len, hidden]

        # Combine BERT and CNN features
        sequence_output = sequence_output + cnn_output

        if span_positions is not None:
            # Span-based classification
            span_reps = self.span_repr(sequence_output, span_positions)
            logits = self.span_classifier(span_reps)
            return logits
        else:
            # Biaffine attention for all possible spans
            start_reps = self.start_proj(sequence_output)
            end_reps = self.end_proj(sequence_output)
            logits = self.biaffine(start_reps, end_reps)
            return logits

    def get_embeddings(self):
        """Get word embeddings for adversarial training"""
        return self.bert.embeddings.word_embeddings.weight
