"""
Inference and evaluation script
"""
import torch
from tqdm import tqdm
import json
from typing import List, Tuple

from config import Config
from model import NestedPrivacyNER
from data_loader import create_dataloader
from transformers import AutoTokenizer


class PrivacyEntityExtractor:
    """Extract privacy entities from text"""
    def __init__(self, model_path: str, config: Config):
        self.config = config
        self.device = config.device

        # Load model
        self.model = NestedPrivacyNER(config).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def extract_entities(self, text: str, threshold=0.5) -> List[Tuple[int, int, str, float]]:
        """
        Extract entities from text
        Returns: List of (start, end, entity_type, confidence)
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Generate all possible spans
        max_span_len = min(self.config.max_span_length, len(text))
        span_positions = []
        char_to_token = []

        for i in range(len(text)):
            token_idx = encoding.char_to_token(0, i)
            char_to_token.append(token_idx if token_idx is not None else -1)

        for start in range(len(text)):
            for end in range(start, min(start + max_span_len, len(text))):
                start_token = char_to_token[start] if start < len(char_to_token) else -1
                end_token = char_to_token[end] if end < len(char_to_token) else -1

                if start_token != -1 and end_token != -1:
                    span_positions.append((start_token, end_token))

        if len(span_positions) == 0:
            return []

        # Batch inference
        span_positions_tensor = torch.tensor(span_positions, dtype=torch.long).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask, span_positions_tensor)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            confidences = torch.max(probs, dim=-1)[0]

        # Extract entities
        entities = []
        for idx, (start, end) in enumerate(span_positions):
            pred = preds[0, idx].item()
            conf = confidences[0, idx].item()

            if pred > 0 and conf >= threshold:  # Not 'O' class
                entity_type = self.config.entity_types[pred - 1]
                entities.append((start, end, entity_type, conf))

        # Non-maximum suppression for nested entities
        entities = self._nms(entities)

        return entities

    def _nms(self, entities: List[Tuple[int, int, str, float]], iou_threshold=0.5) -> List[Tuple[int, int, str, float]]:
        """Non-maximum suppression for overlapping entities"""
        if len(entities) == 0:
            return []

        # Sort by confidence
        entities = sorted(entities, key=lambda x: x[3], reverse=True)

        keep = []
        while len(entities) > 0:
            current = entities[0]
            keep.append(current)
            entities = entities[1:]

            # Remove overlapping entities with lower confidence
            filtered = []
            for entity in entities:
                iou = self._compute_iou(current, entity)
                if iou < iou_threshold:
                    filtered.append(entity)
            entities = filtered

        return keep

    def _compute_iou(self, entity1, entity2):
        """Compute IoU between two entities"""
        start1, end1 = entity1[0], entity1[1]
        start2, end2 = entity2[0], entity2[1]

        intersection = max(0, min(end1, end2) - max(start1, start2))
        union = max(end1, end2) - min(start1, start2)

        return intersection / union if union > 0 else 0


def detailed_evaluation(model_path: str, config: Config):
    """Detailed evaluation with entity-level metrics"""
    print("Loading model and data...")
    extractor = PrivacyEntityExtractor(model_path, config)
    test_loader, _ = create_dataloader(
        config.test_data_path,
        config.test_label_path,
        config,
        is_train=False
    )

    # Entity-level evaluation
    entity_tp = {entity_type: 0 for entity_type in config.entity_types}
    entity_fp = {entity_type: 0 for entity_type in config.entity_types}
    entity_fn = {entity_type: 0 for entity_type in config.entity_types}

    print("\nEvaluating...")
    for batch in tqdm(test_loader):
        texts = batch['text']
        original_labels = batch['original_labels']

        for text, labels in zip(texts, original_labels):
            # Extract ground truth entities
            gt_entities = extract_gt_entities(labels)

            # Predict entities
            pred_entities = extractor.extract_entities(text)

            # Match entities
            matched_gt = set()
            for pred_start, pred_end, pred_type, _ in pred_entities:
                matched = False
                for idx, (gt_start, gt_end, gt_type) in enumerate(gt_entities):
                    if pred_start == gt_start and pred_end == gt_end and pred_type == gt_type:
                        entity_tp[pred_type] += 1
                        matched_gt.add(idx)
                        matched = True
                        break

                if not matched:
                    entity_fp[pred_type] += 1

            # Count false negatives
            for idx, (gt_start, gt_end, gt_type) in enumerate(gt_entities):
                if idx not in matched_gt:
                    entity_fn[gt_type] += 1

    # Compute metrics per entity type
    print("\n" + "="*60)
    print("Entity-level Evaluation Results")
    print("="*60)

    overall_tp = 0
    overall_fp = 0
    overall_fn = 0

    for entity_type in config.entity_types:
        tp = entity_tp[entity_type]
        fp = entity_fp[entity_type]
        fn = entity_fn[entity_type]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n{entity_type}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}")

        overall_tp += tp
        overall_fp += fp
        overall_fn += fn

    # Overall metrics
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    print("\n" + "="*60)
    print("Overall:")
    print(f"  Precision: {overall_precision:.4f}")
    print(f"  Recall: {overall_recall:.4f}")
    print(f"  F1: {overall_f1:.4f}")
    print("="*60)

    return overall_f1


def extract_gt_entities(labels: List[str]) -> List[Tuple[int, int, str]]:
    """Extract ground truth entities from BIO labels"""
    entities = []
    current_entity = None
    start_idx = -1

    for idx, label in enumerate(labels):
        if label.startswith('B_'):
            if current_entity is not None:
                entities.append((start_idx, idx - 1, current_entity))
            current_entity = label[2:]
            start_idx = idx
        elif label.startswith('E_'):
            if current_entity is not None:
                entity_type = label[2:]
                if entity_type == current_entity:
                    entities.append((start_idx, idx, current_entity))
            current_entity = None
            start_idx = -1
        elif label == 'O':
            if current_entity is not None:
                entities.append((start_idx, idx - 1, current_entity))
            current_entity = None
            start_idx = -1

    if current_entity is not None:
        entities.append((start_idx, len(labels) - 1, current_entity))

    return entities


if __name__ == "__main__":
    config = Config()
    detailed_evaluation('best_model.pt', config)
