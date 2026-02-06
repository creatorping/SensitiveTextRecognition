"""
Utility functions for data analysis and visualization
"""
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np


def analyze_dataset(data_path, label_path):
    """Analyze dataset statistics"""
    print("="*80)
    print("Dataset Analysis")
    print("="*80)

    # Load data
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

    # Basic statistics
    print(f"\nTotal samples: {len(texts)}")
    print(f"Average text length: {np.mean([len(t) for t in texts]):.2f} characters")
    print(f"Max text length: {max([len(t) for t in texts])} characters")
    print(f"Min text length: {min([len(t) for t in texts])} characters")

    # Entity statistics
    entity_counts = Counter()
    nested_count = 0
    total_entities = 0

    for label_seq in labels:
        entities = []
        current_entity = None
        start_idx = -1

        for idx, label in enumerate(label_seq):
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
                        entity_counts[entity_type] += 1
                        total_entities += 1
                current_entity = None
                start_idx = -1
            elif label == 'O':
                if current_entity is not None:
                    entities.append((start_idx, idx - 1, current_entity))
                    entity_counts[current_entity] += 1
                    total_entities += 1
                current_entity = None
                start_idx = -1

        if current_entity is not None:
            entities.append((start_idx, len(label_seq) - 1, current_entity))
            entity_counts[current_entity] += 1
            total_entities += 1

        # Check for nested entities
        for i in range(len(entities)):
            for j in range(len(entities)):
                if i != j:
                    start1, end1, _ = entities[i]
                    start2, end2, _ = entities[j]
                    # Check if entity i contains entity j
                    if start1 <= start2 and end1 >= end2 and not (start1 == start2 and end1 == end2):
                        nested_count += 1
                        break

    print(f"\nTotal entities: {total_entities}")
    print(f"Nested entities: {nested_count} ({nested_count/total_entities*100:.2f}%)")

    print("\nEntity type distribution:")
    for entity_type, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {entity_type}: {count} ({count/total_entities*100:.2f}%)")

    # Entity length statistics
    entity_lengths = []
    for label_seq in labels:
        current_length = 0
        in_entity = False

        for label in label_seq:
            if label.startswith('B_'):
                in_entity = True
                current_length = 1
            elif label.startswith('I_'):
                if in_entity:
                    current_length += 1
            elif label.startswith('E_'):
                if in_entity:
                    current_length += 1
                    entity_lengths.append(current_length)
                in_entity = False
                current_length = 0
            else:
                if in_entity:
                    entity_lengths.append(current_length)
                in_entity = False
                current_length = 0

    if entity_lengths:
        print(f"\nEntity length statistics:")
        print(f"  Average: {np.mean(entity_lengths):.2f} characters")
        print(f"  Max: {max(entity_lengths)} characters")
        print(f"  Min: {min(entity_lengths)} characters")
        print(f"  Median: {np.median(entity_lengths):.2f} characters")

    print("="*80)


def compare_models_performance(results_dict):
    """
    Compare performance of different model configurations
    results_dict: {model_name: {'f1': f1, 'precision': p, 'recall': r}}
    """
    models = list(results_dict.keys())
    f1_scores = [results_dict[m]['f1'] for m in models]
    precision_scores = [results_dict[m]['precision'] for m in models]
    recall_scores = [results_dict[m]['recall'] for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision_scores, width, label='Precision')
    ax.bar(x, recall_scores, width, label='Recall')
    ax.bar(x + width, f1_scores, width, label='F1')

    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    print("Saved comparison plot to model_comparison.png")


if __name__ == "__main__":
    from config import Config

    config = Config()

    print("\nTraining Set Analysis:")
    analyze_dataset(config.train_data_path, config.train_label_path)

    print("\n\nTest Set Analysis:")
    analyze_dataset(config.test_data_path, config.test_label_path)
