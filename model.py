"""
Nested Privacy Entity Recognition Model with Biaffine Attention
针对H800优化，支持大模型和高级特征融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from config import Config


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.out_proj(out)

        return out


class LayerFusion(nn.Module):
    """层次化特征融合 - 融合BERT多层输出"""
    def __init__(self, hidden_size, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        # 可学习的层权重
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states_list):
        """
        Args:
            hidden_states_list: list of [batch, seq_len, hidden] tensors
        """
        # 归一化权重
        weights = F.softmax(self.layer_weights, dim=0)

        # 加权融合
        fused = sum(w * h for w, h in zip(weights, hidden_states_list))
        fused = self.layer_norm(fused)

        return fused


class BiaffineAttention(nn.Module):
    """Biaffine attention for span classification"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.Tensor(out_features, in_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.bias)

    def forward(self, x1, x2):
        batch_size, seq_len, hidden = x1.size()

        x1 = x1.unsqueeze(2)
        x2 = x2.unsqueeze(1)

        scores = torch.einsum('bnih,oij,bnjh->bnmo', x1, self.W, x2)
        scores = scores + self.bias

        return scores


class SpanRepresentation(nn.Module):
    """Generate span representations from token embeddings"""
    def __init__(self, hidden_size, span_hidden_size, dropout=0.3):
        super().__init__()
        self.start_proj = nn.Linear(hidden_size, span_hidden_size)
        self.end_proj = nn.Linear(hidden_size, span_hidden_size)
        self.span_proj = nn.Linear(hidden_size * 2, span_hidden_size)

        # 增加span宽度嵌入
        self.width_embedding = nn.Embedding(50, span_hidden_size)
        self.combine_proj = nn.Linear(span_hidden_size * 2, span_hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(span_hidden_size)

    def forward(self, sequence_output, span_positions):
        batch_size, num_spans, _ = span_positions.size()

        start_indices = span_positions[:, :, 0]
        end_indices = span_positions[:, :, 1]

        start_reps = torch.gather(
            sequence_output,
            1,
            start_indices.unsqueeze(-1).expand(-1, -1, sequence_output.size(-1))
        )

        end_reps = torch.gather(
            sequence_output,
            1,
            end_indices.unsqueeze(-1).expand(-1, -1, sequence_output.size(-1))
        )

        # 基础span表示
        span_reps = torch.cat([start_reps, end_reps], dim=-1)
        span_reps = self.span_proj(span_reps)

        # 添加宽度信息
        widths = (end_indices - start_indices).clamp(0, 49)
        width_reps = self.width_embedding(widths)

        # 融合
        combined = torch.cat([span_reps, width_reps], dim=-1)
        combined = self.combine_proj(combined)
        combined = self.layer_norm(combined)
        combined = self.dropout(torch.relu(combined))

        return combined


class NestedPrivacyNER(nn.Module):
    """Nested Privacy Entity Recognition Model - H800优化版"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # 加载预训练模型配置
        try:
            model_config = AutoConfig.from_pretrained(
                config.model_name,
                trust_remote_code=True
            )
            # 更新hidden_size以匹配实际模型
            actual_hidden_size = getattr(model_config, 'hidden_size', config.hidden_size)
            if actual_hidden_size != config.hidden_size:
                print(f"注意: 模型实际hidden_size={actual_hidden_size}, 配置中为{config.hidden_size}")
                config.hidden_size = actual_hidden_size
        except Exception as e:
            print(f"无法加载模型配置: {e}")

        # BERT encoder
        self.bert = AutoModel.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            output_hidden_states=getattr(config, 'use_layer_fusion', False)
        )

        # Gradient checkpointing
        if config.use_gradient_checkpointing:
            self.bert.gradient_checkpointing_enable()

        self.dropout = nn.Dropout(getattr(config, 'hidden_dropout', 0.1))

        # 层次化特征融合
        self.use_layer_fusion = getattr(config, 'use_layer_fusion', False)
        if self.use_layer_fusion:
            fusion_layers = getattr(config, 'fusion_layers', [-1, -2, -3, -4])
            self.layer_fusion = LayerFusion(config.hidden_size, len(fusion_layers))
            self.fusion_layer_indices = fusion_layers

        # 多头注意力增强
        self.use_multi_head_attention = getattr(config, 'use_multi_head_attention', False)
        if self.use_multi_head_attention:
            num_heads = getattr(config, 'num_attention_heads', 8)
            self.extra_attention = MultiHeadAttention(
                config.hidden_size,
                num_heads,
                getattr(config, 'attention_dropout', 0.1)
            )
            self.attention_layer_norm = nn.LayerNorm(config.hidden_size)

        # Span representation
        self.span_repr = SpanRepresentation(
            config.hidden_size,
            config.span_hidden_size,
            getattr(config, 'classifier_dropout', 0.3)
        )

        # Biaffine attention
        self.start_proj = nn.Linear(config.hidden_size, config.biaffine_size)
        self.end_proj = nn.Linear(config.hidden_size, config.biaffine_size)
        self.biaffine = BiaffineAttention(config.biaffine_size, len(config.entity_types) + 1)

        # Span classifier - 增强版
        classifier_dropout = getattr(config, 'classifier_dropout', 0.3)
        self.span_classifier = nn.Sequential(
            nn.Linear(config.span_hidden_size, config.span_hidden_size * 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(config.span_hidden_size * 2, config.span_hidden_size),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(config.span_hidden_size, len(config.entity_types) + 1)
        )

        # Multi-scale CNN for local context
        self.cnn_layers = nn.ModuleList([
            nn.Conv1d(config.hidden_size, config.hidden_size // 4, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]
        ])
        self.cnn_proj = nn.Linear(config.hidden_size // 4 * 3, config.hidden_size)
        self.cnn_layer_norm = nn.LayerNorm(config.hidden_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
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
        # BERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # 层次化特征融合
        if self.use_layer_fusion and hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = [outputs.hidden_states[i] for i in self.fusion_layer_indices]
            sequence_output = self.layer_fusion(hidden_states)
        else:
            sequence_output = outputs.last_hidden_state

        sequence_output = self.dropout(sequence_output)

        # 多头注意力增强
        if self.use_multi_head_attention:
            attn_output = self.extra_attention(sequence_output, attention_mask)
            sequence_output = self.attention_layer_norm(sequence_output + attn_output)

        # Multi-scale CNN
        cnn_input = sequence_output.transpose(1, 2)
        cnn_outputs = [torch.relu(conv(cnn_input)) for conv in self.cnn_layers]
        cnn_output = torch.cat(cnn_outputs, dim=1)
        cnn_output = cnn_output.transpose(1, 2)
        cnn_output = self.cnn_proj(cnn_output)
        cnn_output = self.cnn_layer_norm(cnn_output)

        # Residual connection
        sequence_output = sequence_output + cnn_output

        if span_positions is not None:
            span_reps = self.span_repr(sequence_output, span_positions)
            logits = self.span_classifier(span_reps)
            return logits
        else:
            start_reps = self.start_proj(sequence_output)
            end_reps = self.end_proj(sequence_output)
            logits = self.biaffine(start_reps, end_reps)
            return logits

    def get_embeddings(self):
        """Get word embeddings for adversarial training"""
        return self.bert.embeddings.word_embeddings.weight
