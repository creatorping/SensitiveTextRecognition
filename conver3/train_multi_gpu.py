"""
Multi-GPU Training Script for Dual RTX 5090D
支持DataParallel和DistributedDataParallel
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn.parallel import DataParallel
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import random
import os
from sklearn.metrics import f1_score, precision_score, recall_score

from config import Config
from model import NestedPrivacyNER
from data_loader import create_dataloader
from adversarial import FGM, PGD, RDrop, EMA


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_loss(logits, labels, label_smoothing=0.0, class_weights=None):
    """Compute cross entropy loss with optional class weights"""
    mask = labels != -1
    active_logits = logits[mask]
    active_labels = labels[mask]

    if len(active_labels) == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Use class weights if provided
    if class_weights is not None:
        loss = nn.functional.cross_entropy(
            active_logits,
            active_labels,
            weight=class_weights,
            reduction='mean'
        )
    else:
        loss = nn.functional.cross_entropy(active_logits, active_labels)

    # Add small value to prevent NaN
    loss = loss + 1e-8

    return loss


def train_epoch(model, dataloader, optimizer, scheduler, config, fgm=None, pgd=None, ema=None, device_ids=None):
    """Train for one epoch with multi-GPU support"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(progress_bar):
        # Move data to GPU
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        span_positions = batch['span_positions'].to(config.device)
        span_labels = batch['span_labels'].to(config.device)

        # Get class weights
        class_weights = getattr(config, 'class_weights', None)

        # Forward pass
        if config.use_rdrop:
            logits1 = model(input_ids, attention_mask, span_positions)
            logits2 = model(input_ids, attention_mask, span_positions)

            ce_loss1 = compute_loss(logits1, span_labels, config.label_smoothing, class_weights)
            ce_loss2 = compute_loss(logits2, span_labels, config.label_smoothing, class_weights)
            ce_loss = (ce_loss1 + ce_loss2) / 2

            mask = span_labels != -1
            kl_loss = RDrop.compute_kl_loss(logits1, logits2, mask.float())

            loss = ce_loss + config.rdrop_alpha * kl_loss
        else:
            logits = model(input_ids, attention_mask, span_positions)
            loss = compute_loss(logits, span_labels, config.label_smoothing, class_weights)

        loss = loss / config.gradient_accumulation_steps

        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: Skipping batch {batch_idx} due to NaN/Inf loss")
            optimizer.zero_grad()
            continue

        loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            # Check for NaN gradients
            nan_grads = False
            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    print(f"WARNING: NaN/Inf gradient in {name}")
                    nan_grads = True
                    break

            if nan_grads:
                optimizer.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if ema is not None:
                ema.update()

        total_loss += loss.item() * config.gradient_accumulation_steps
        progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, config, debug=False):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_labels = []
    debug_samples = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            span_positions = batch['span_positions'].to(config.device)
            span_labels = batch['span_labels'].to(config.device)

            logits = model(input_ids, attention_mask, span_positions)
            preds = torch.argmax(logits, dim=-1)

            mask = span_labels != -1
            active_preds = preds[mask].cpu().numpy()
            active_labels = span_labels[mask].cpu().numpy()

            all_preds.extend(active_preds)
            all_labels.extend(active_labels)

            if debug and batch_idx < 3:
                debug_samples.append({
                    'batch_idx': batch_idx,
                    'labels': active_labels[:20],
                    'preds': active_preds[:20],
                })

    if debug:
        print("\n" + "="*80)
        print("DEBUG: Prediction Analysis")
        print("="*80)

        from collections import Counter
        label_dist = Counter(all_labels)
        pred_dist = Counter(all_preds)

        print("\nLabel Distribution (True):")
        for label_id, count in sorted(label_dist.items()):
            label_name = config.id2label.get(label_id, f"ID_{label_id}")
            print(f"  {label_name} (ID={label_id}): {count} ({count/len(all_labels)*100:.2f}%)")

        print("\nPrediction Distribution:")
        for label_id, count in sorted(pred_dist.items()):
            label_name = config.id2label.get(label_id, f"ID_{label_id}")
            print(f"  {label_name} (ID={label_id}): {count} ({count/len(all_preds)*100:.2f}%)")

        print("\nSample Predictions (First 3 batches):")
        for sample in debug_samples:
            print(f"\nBatch {sample['batch_idx']}:")
            print(f"  True labels: {sample['labels']}")
            print(f"  Predictions: {sample['preds']}")
        print("="*80 + "\n")

    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def train(config):
    """Main training function with multi-GPU support"""
    set_seed(config.seed)

    # Check available GPUs
    n_gpus = torch.cuda.device_count()
    print(f"\n{'='*80}")
    print(f"🚀 Multi-GPU Training Configuration")
    print(f"{'='*80}")
    print(f"Available GPUs: {n_gpus}")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"{'='*80}\n")

    # Create dataloaders
    print("Loading data...")
    train_loader, tokenizer = create_dataloader(
        config.train_data_path,
        config.train_label_path,
        config,
        is_train=True
    )

    test_loader, _ = create_dataloader(
        config.test_data_path,
        config.test_label_path,
        config,
        is_train=False
    )

    # Create model
    print("Creating model...")
    model = NestedPrivacyNER(config)

    # Wrap model with DataParallel for multi-GPU
    if n_gpus > 1:
        print(f"Using DataParallel with {n_gpus} GPUs")
        device_ids = list(range(n_gpus))
        model = DataParallel(model, device_ids=device_ids)
        # Update batch size for multi-GPU
        effective_batch_size = config.batch_size * n_gpus
        print(f"Effective batch size: {effective_batch_size}")
    else:
        device_ids = None

    model = model.to(config.device)

    # Load checkpoint if resuming
    start_epoch = 0
    best_f1 = 0

    # Get the actual model (unwrap from DataParallel if needed)
    actual_model = model.module if isinstance(model, DataParallel) else model

    if config.resume_from_checkpoint and os.path.exists(config.resume_from_checkpoint):
        print(f"\nLoading checkpoint from {config.resume_from_checkpoint}...")
        checkpoint = torch.load(config.resume_from_checkpoint, map_location=config.device)

        # Load model state dict
        actual_model.load_state_dict(checkpoint['model_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint['f1']
        print(f"Resumed from epoch {checkpoint['epoch']}, best F1: {best_f1:.4f}")
        print(f"Will continue training from epoch {start_epoch}")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Load optimizer and scheduler state if resuming
    if config.resume_from_checkpoint and os.path.exists(config.resume_from_checkpoint):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # EMA
    ema = EMA(actual_model, decay=0.999)

    if config.resume_from_checkpoint and os.path.exists(config.resume_from_checkpoint):
        ema.shadow = checkpoint['ema_state_dict']

    # Compute class weights
    print("\nComputing class weights...")
    class_counts = torch.zeros(len(config.entity_types) + 1)
    total_samples = 0

    for batch in train_loader:
        span_labels = batch['span_labels']
        mask = span_labels != -1
        active_labels = span_labels[mask]
        for label in active_labels:
            class_counts[label] += 1
            total_samples += 1

    # Compute inverse frequency weights
    class_weights = total_samples / (class_counts + 1e-6)

    # Reduce weight for 'O' class
    if config.reduce_o_weight:
        class_weights[0] = class_weights[0] * 0.5

    # Normalize weights
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = class_weights.to(config.device)

    print("\nClass weights computed:")
    for i, weight in enumerate(class_weights):
        label_name = config.id2label.get(i, f"ID_{i}")
        print(f"  {label_name} (ID={i}): weight={weight:.4f}, count={int(class_counts[i])}")

    config.class_weights = class_weights

    # Training loop
    print("\nStarting training...")

    for epoch in range(start_epoch, config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, config, None, None, ema, device_ids)
        print(f"Train Loss: {train_loss:.4f}")

        # Evaluate with EMA
        ema.apply_shadow()
        debug_mode = (epoch < 3)
        metrics = evaluate(model, test_loader, config, debug=debug_mode)
        ema.restore()

        print(f"F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")

        # Save best model only
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            # Save the unwrapped model
            torch.save({
                'epoch': epoch,
                'model_state_dict': actual_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ema_state_dict': ema.shadow,
                'f1': best_f1,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
            }, 'best_model.pt')
            print(f"✓ Saved best model with F1: {best_f1:.4f}")

    print(f"\nTraining completed! Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    config = Config()
    train(config)
