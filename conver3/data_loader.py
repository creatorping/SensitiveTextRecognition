"""
Data loader for nested privacy entity recognition
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Tuple, Dict
import numpy as np
from config import Config


class PrivacyDataset(Dataset):
    def __init__(self, data_path: str, label_path: str, tokenizer, config: Config, is_train=True):
        self.tokenizer = tokenizer
        self.config = config
        self.is_train = is_train
        self.max_length = config.max_length

        # Load data
        self.texts, self.labels = self._load_data(data_path, label_path)

    def _load_data(self, data_path: str, label_path: str) -> Tuple[List[str], List[List[str]]]:
        """Load text and labels from files"""
        texts = []
        labels = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('→')[-1] if '→' in line else line.strip()
                texts.append(text)

        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                label = line.strip().split('→')[-1] if '→' in line else line.strip()
                label_list = label.split()
                labels.append(label_list)

        assert len(texts) == len(labels), "Text and label counts don't match"

        return texts, labels

    def _extract_spans(self, labels: List[str], text: str) -> List[Tuple[int, int, str]]:
        """Extract entity spans from BIO labels
        Returns: List of (start, end, entity_type)
        """
        spans = []
        current_entity = None
        start_idx = -1

        for idx, label in enumerate(labels):
            if label.startswith('B_'):
                # Save previous entity if exists
                if current_entity is not None:
                    spans.append((start_idx, idx - 1, current_entity))
                # Start new entity
                current_entity = label[2:]  # Remove 'B_' prefix
                start_idx = idx
            elif label.startswith('E_'):
                # End current entity
                if current_entity is not None:
                    entity_type = label[2:]  # Remove 'E_' prefix
                    if entity_type == current_entity:
                        spans.append((start_idx, idx, current_entity))
                current_entity = None
                start_idx = -1
            elif label.startswith('I_'):
                # Continue current entity
                continue
            else:  # 'O'
                # Save previous entity if exists
                if current_entity is not None:
                    spans.append((start_idx, idx - 1, current_entity))
                current_entity = None
                start_idx = -1

        # Handle entity at end of sequence
        if current_entity is not None:
            spans.append((start_idx, len(labels) - 1, current_entity))

        return spans

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Align labels with tokens
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # Create character to token mapping
        char_to_token = []
        for i in range(len(text)):
            token_idx = encoding.char_to_token(i)
            char_to_token.append(token_idx if token_idx is not None else -1)

        # Extract spans
        spans = self._extract_spans(labels, text)

        # Create span labels matrix (for all possible spans)
        max_span_len = min(self.config.max_span_length, len(text))
        span_labels = []
        span_positions = []

        for start in range(len(text)):
            for end in range(start, min(start + max_span_len, len(text))):
                # Map character positions to token positions
                start_token = char_to_token[start] if start < len(char_to_token) else -1
                end_token = char_to_token[end] if end < len(char_to_token) else -1

                if start_token == -1 or end_token == -1:
                    continue

                # Check if this span matches any entity
                label = 'O'
                for span_start, span_end, entity_type in spans:
                    if span_start == start and span_end == end:
                        label = entity_type
                        break

                span_positions.append((start_token, end_token))
                span_labels.append(self.config.entity_types.index(label) if label != 'O' else -1)

        # Pad or truncate spans
        max_spans = 200  # Maximum number of spans to consider
        if len(span_positions) > max_spans:
            span_positions = span_positions[:max_spans]
            span_labels = span_labels[:max_spans]

        # Pad
        while len(span_positions) < max_spans:
            span_positions.append((0, 0))
            span_labels.append(-1)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'span_positions': torch.tensor(span_positions, dtype=torch.long),
            'span_labels': torch.tensor(span_labels, dtype=torch.long),
            'text': text,
            'original_labels': labels
        }


def collate_fn(batch):
    """Custom collate function to handle non-tensor fields"""
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'span_positions': torch.stack([item['span_positions'] for item in batch]),
        'span_labels': torch.stack([item['span_labels'] for item in batch]),
        'text': [item['text'] for item in batch],
        'original_labels': [item['original_labels'] for item in batch]
    }


def create_dataloader(data_path: str, label_path: str, config: Config, is_train=True):
    """Create dataloader"""
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        local_files_only=True,
        trust_remote_code=True
    )

    dataset = PrivacyDataset(data_path, label_path, tokenizer, config, is_train)

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=is_train,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return dataloader, tokenizer
