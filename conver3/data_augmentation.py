"""
Data augmentation strategies for NER
"""
import random
import numpy as np
from typing import List, Tuple


class NERDataAugmenter:
    """Data augmentation for NER tasks"""

    def __init__(self, aug_prob=0.15):
        self.aug_prob = aug_prob

    def random_mask(self, text: str, labels: List[str]) -> Tuple[str, List[str]]:
        """Randomly mask some characters (not in entities)"""
        chars = list(text)
        new_labels = labels.copy()

        for i in range(len(chars)):
            if labels[i] == 'O' and random.random() < self.aug_prob:
                chars[i] = '[MASK]'

        return ''.join(chars), new_labels

    def random_swap(self, text: str, labels: List[str]) -> Tuple[str, List[str]]:
        """Randomly swap adjacent characters (not in entities)"""
        chars = list(text)
        new_labels = labels.copy()

        for i in range(len(chars) - 1):
            if (labels[i] == 'O' and labels[i+1] == 'O' and
                random.random() < self.aug_prob):
                chars[i], chars[i+1] = chars[i+1], chars[i]
                new_labels[i], new_labels[i+1] = new_labels[i+1], new_labels[i]

        return ''.join(chars), new_labels

    def entity_replacement(self, text: str, labels: List[str],
                          entity_dict: dict) -> Tuple[str, List[str]]:
        """Replace entities with similar entities"""
        # Extract entities
        entities = []
        i = 0
        while i < len(labels):
            if labels[i].startswith('B_'):
                entity_type = labels[i][2:]
                start = i
                i += 1
                while i < len(labels) and labels[i].startswith('I_'):
                    i += 1
                if i < len(labels) and labels[i].startswith('E_'):
                    end = i
                    entities.append((start, end, entity_type))
                    i += 1
                else:
                    i += 1
            else:
                i += 1

        # Replace entities
        chars = list(text)
        new_labels = labels.copy()

        for start, end, entity_type in entities:
            if (entity_type in entity_dict and
                random.random() < self.aug_prob):
                replacements = entity_dict[entity_type]
                if replacements:
                    new_entity = random.choice(replacements)
                    # Replace text
                    for j in range(start, end + 1):
                        if j - start < len(new_entity):
                            chars[j] = new_entity[j - start]
                        else:
                            chars[j] = ''

        return ''.join(chars), new_labels

    def back_translation(self, text: str, labels: List[str]) -> Tuple[str, List[str]]:
        """
        Simulate back translation by random synonym replacement
        (In practice, you would use a translation API)
        """
        # Simplified version: just return original
        # In production, use translation API
        return text, labels

    def augment(self, text: str, labels: List[str],
                entity_dict: dict = None) -> List[Tuple[str, List[str]]]:
        """
        Apply multiple augmentation strategies
        Returns list of (augmented_text, augmented_labels)
        """
        augmented_samples = [(text, labels)]  # Original

        # Random mask
        if random.random() < 0.3:
            aug_text, aug_labels = self.random_mask(text, labels)
            augmented_samples.append((aug_text, aug_labels))

        # Random swap
        if random.random() < 0.3:
            aug_text, aug_labels = self.random_swap(text, labels)
            augmented_samples.append((aug_text, aug_labels))

        # Entity replacement
        if entity_dict and random.random() < 0.3:
            aug_text, aug_labels = self.entity_replacement(text, labels, entity_dict)
            augmented_samples.append((aug_text, aug_labels))

        return augmented_samples


class MixupAugmenter:
    """
    Mixup augmentation for NER
    Reference: https://arxiv.org/abs/1710.09412
    """
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def mixup(self, x1, x2, y1, y2):
        """
        Mixup two samples
        Args:
            x1, x2: input features
            y1, y2: labels
        """
        lam = np.random.beta(self.alpha, self.alpha)
        mixed_x = lam * x1 + (1 - lam) * x2
        return mixed_x, y1, y2, lam


def create_entity_dict_from_data(data_path: str, label_path: str) -> dict:
    """
    Create entity dictionary from training data for augmentation
    """
    entity_dict = {}

    with open(data_path, 'r', encoding='utf-8') as f_data, \
         open(label_path, 'r', encoding='utf-8') as f_label:

        for text_line, label_line in zip(f_data, f_label):
            text = text_line.strip().split('→')[-1] if '→' in text_line else text_line.strip()
            labels = label_line.strip().split('→')[-1] if '→' in label_line else label_line.strip()
            labels = labels.split()

            # Extract entities
            i = 0
            while i < len(labels):
                if labels[i].startswith('B_'):
                    entity_type = labels[i][2:]
                    start = i
                    i += 1
                    while i < len(labels) and labels[i].startswith('I_'):
                        i += 1
                    if i < len(labels) and labels[i].startswith('E_'):
                        end = i
                        entity_text = text[start:end+1]

                        if entity_type not in entity_dict:
                            entity_dict[entity_type] = []
                        if entity_text not in entity_dict[entity_type]:
                            entity_dict[entity_type].append(entity_text)
                        i += 1
                    else:
                        i += 1
                else:
                    i += 1

    return entity_dict
