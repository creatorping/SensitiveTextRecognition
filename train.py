"""
Training script with adversarial training and model smoothing
"""
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import random
from sklearn.metrics import f1_score, precision_score, recall_score

from config import Config
from model import NestedPrivacyNER
from data_loader import create_dataloader
from adversarial import FGM, PGD, RDrop, EMA


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_loss(logits, labels, label_smoothing=0.0, class_weights=None):
    """
    Compute cross entropy loss with label smoothing and class weights
    Args:
        logits: [batch, num_spans, num_classes]
        labels: [batch, num_spans]
        class_weights: tensor of shape [num_classes] for class weighting
    """
    # Filter out padding (-1 labels)
    mask = labels != -1
    active_logits = logits[mask]
    active_labels = labels[mask]

    if len(active_labels) == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Use weighted cross entropy
    loss = nn.functional.cross_entropy(
        active_logits,
        active_labels,
        weight=class_weights,
        reduction='mean'
    )

    # Check for NaN
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"WARNING: NaN/Inf detected in loss!")
        print(f"  Logits stats: min={active_logits.min():.4f}, max={active_logits.max():.4f}")
        print(f"  Labels stats: min={active_labels.min()}, max={active_labels.max()}")
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    return loss


def train_epoch(model, dataloader, optimizer, scheduler, config, fgm=None, pgd=None, ema=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")

    # Initialize gradients
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        span_positions = batch['span_positions'].to(config.device)
        span_labels = batch['span_labels'].to(config.device)

        # Forward pass
        class_weights = getattr(config, 'class_weights', None)

        if config.use_rdrop:
            # R-Drop: forward twice with different dropout
            logits1 = model(input_ids, attention_mask, span_positions)
            logits2 = model(input_ids, attention_mask, span_positions)

            # Compute CE loss with class weights
            ce_loss1 = compute_loss(logits1, span_labels, config.label_smoothing, class_weights)
            ce_loss2 = compute_loss(logits2, span_labels, config.label_smoothing, class_weights)
            ce_loss = (ce_loss1 + ce_loss2) / 2

            # Compute KL loss
            mask = span_labels != -1
            kl_loss = RDrop.compute_kl_loss(logits1, logits2, mask.float())

            # Total loss
            loss = ce_loss + config.rdrop_alpha * kl_loss
        else:
            logits = model(input_ids, attention_mask, span_positions)
            loss = compute_loss(logits, span_labels, config.label_smoothing, class_weights)

        # Backward
        loss = loss / config.gradient_accumulation_steps

        # Check for NaN before backward
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: Skipping batch {batch_idx} due to NaN/Inf loss")
            optimizer.zero_grad()
            continue

        loss.backward()

        # Adversarial training - FGM
        if config.use_fgm and fgm is not None:
            fgm.attack(emb_name='word_embeddings')
            if config.use_rdrop:
                logits_adv1 = model(input_ids, attention_mask, span_positions)
                logits_adv2 = model(input_ids, attention_mask, span_positions)
                ce_loss_adv1 = compute_loss(logits_adv1, span_labels, config.label_smoothing, class_weights)
                ce_loss_adv2 = compute_loss(logits_adv2, span_labels, config.label_smoothing, class_weights)
                loss_adv = (ce_loss_adv1 + ce_loss_adv2) / 2
            else:
                logits_adv = model(input_ids, attention_mask, span_positions)
                loss_adv = compute_loss(logits_adv, span_labels, config.label_smoothing, class_weights)
            loss_adv = loss_adv / config.gradient_accumulation_steps

            if not (torch.isnan(loss_adv) or torch.isinf(loss_adv)):
                loss_adv.backward()
            fgm.restore(emb_name='word_embeddings')

        # Adversarial training - PGD
        if config.use_pgd and pgd is not None:
            pgd.backup_grad()
            for t in range(config.adv_k):
                pgd.attack(emb_name='word_embeddings', is_first_attack=(t == 0))
                if t != config.adv_k - 1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                if config.use_rdrop:
                    logits_adv1 = model(input_ids, attention_mask, span_positions)
                    logits_adv2 = model(input_ids, attention_mask, span_positions)
                    ce_loss_adv1 = compute_loss(logits_adv1, span_labels, config.label_smoothing, class_weights)
                    ce_loss_adv2 = compute_loss(logits_adv2, span_labels, config.label_smoothing, class_weights)
                    loss_adv = (ce_loss_adv1 + ce_loss_adv2) / 2
                else:
                    logits_adv = model(input_ids, attention_mask, span_positions)
                    loss_adv = compute_loss(logits_adv, span_labels, config.label_smoothing, class_weights)
                loss_adv = loss_adv / config.gradient_accumulation_steps

                if not (torch.isnan(loss_adv) or torch.isinf(loss_adv)):
                    loss_adv.backward()
            pgd.restore(emb_name='word_embeddings')

        # Gradient accumulation
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            # Check for NaN gradients
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    print(f"WARNING: NaN/Inf gradient in {name}")
                    has_nan_grad = True
                    break

            if has_nan_grad:
                print("Skipping optimizer step due to NaN gradients")
                optimizer.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # EMA update
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

    # For debugging
    debug_samples = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            span_positions = batch['span_positions'].to(config.device)
            span_labels = batch['span_labels'].to(config.device)

            logits = model(input_ids, attention_mask, span_positions)
            preds = torch.argmax(logits, dim=-1)

            # Filter out padding
            mask = span_labels != -1
            active_preds = preds[mask].cpu().numpy()
            active_labels = span_labels[mask].cpu().numpy()

            all_preds.extend(active_preds)
            all_labels.extend(active_labels)

            # Collect debug samples (first 3 batches)
            if debug and batch_idx < 3:
                debug_samples.append({
                    'batch_idx': batch_idx,
                    'labels': active_labels[:20],  # First 20 spans
                    'preds': active_preds[:20],
                    'logits_sample': logits[0, :5].cpu().numpy()  # First 5 spans of first sample
                })

    # Print debug information
    if debug:
        print("\n" + "="*80)
        print("DEBUG: Prediction Analysis")
        print("="*80)

        # Class distribution
        from collections import Counter
        label_dist = Counter(all_labels)
        pred_dist = Counter(all_preds)

        print(f"\nLabel Distribution (True):")
        for label_id in sorted(label_dist.keys()):
            label_name = config.id2label.get(label_id, f"ID_{label_id}")
            print(f"  {label_name} (ID={label_id}): {label_dist[label_id]} ({100*label_dist[label_id]/len(all_labels):.2f}%)")

        print(f"\nPrediction Distribution:")
        for pred_id in sorted(pred_dist.keys()):
            pred_name = config.id2label.get(pred_id, f"ID_{pred_id}")
            print(f"  {pred_name} (ID={pred_id}): {pred_dist[pred_id]} ({100*pred_dist[pred_id]/len(all_preds):.2f}%)")

        print(f"\nSample Predictions (first 3 batches):")
        for sample in debug_samples:
            print(f"\n  Batch {sample['batch_idx']}:")
            print(f"    True labels: {sample['labels']}")
            print(f"    Predictions: {sample['preds']}")
            print(f"    Logits (first 5 spans, first sample):")
            for i, logit in enumerate(sample['logits_sample']):
                print(f"      Span {i}: {logit}")

        print("="*80 + "\n")

    # Compute metrics
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def train(config):
    """Main training function"""
    # Set seed
    set_seed(config.seed)

    # 打印GPU信息
    print("="*60)
    print("GPU Configuration")
    print("="*60)
    if torch.cuda.is_available():
        print(f"CUDA Available: True")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        print(f"Number of GPUs visible: {torch.cuda.device_count()}")
        print(f"Current device: {config.device}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA Available: False, using CPU")
    print("="*60)

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
    model = NestedPrivacyNER(config).to(config.device)

    # Load checkpoint if resuming
    start_epoch = 0
    best_f1 = 0
    patience_counter = 0
    if config.resume_from_checkpoint and os.path.exists(config.resume_from_checkpoint):
        print(f"\nLoading checkpoint from {config.resume_from_checkpoint}...")
        checkpoint = torch.load(config.resume_from_checkpoint, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch
        best_f1 = checkpoint.get('f1', 0)
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"Resumed from epoch {checkpoint.get('epoch', 0)}, best F1: {best_f1:.4f}")
        print(f"Will continue training from epoch {start_epoch + 1}")
    else:
        print("Starting training from scratch...")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Load optimizer state if resuming
    if config.resume_from_checkpoint and os.path.exists(config.resume_from_checkpoint):
        checkpoint = torch.load(config.resume_from_checkpoint, map_location=config.device)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state restored")

    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Load scheduler state if resuming
    if config.resume_from_checkpoint and os.path.exists(config.resume_from_checkpoint):
        checkpoint = torch.load(config.resume_from_checkpoint, map_location=config.device)
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state restored")

    # Add ReduceLROnPlateau for adaptive learning rate
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    plateau_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )

    # Adversarial training
    fgm = FGM(model, epsilon=config.adv_epsilon) if config.use_fgm else None
    pgd = PGD(model, epsilon=config.adv_epsilon, alpha=config.adv_alpha) if config.use_pgd else None

    # EMA
    ema = EMA(model, decay=0.999)

    # Compute class weights to handle imbalance
    print("\nComputing class weights...")
    class_counts = torch.zeros(len(config.entity_types) + 1)
    total_samples = 0

    for batch in train_loader:
        span_labels = batch['span_labels']
        mask = span_labels != -1
        active_labels = span_labels[mask]
        for label in active_labels:
            class_counts[label] += 1
        total_samples += len(active_labels)

    # Compute inverse frequency weights
    class_weights = torch.zeros(len(config.entity_types) + 1)
    for i in range(len(class_weights)):
        if class_counts[i] > 0:
            class_weights[i] = total_samples / (len(class_weights) * class_counts[i])
        else:
            class_weights[i] = 1.0

    # Normalize weights
    class_weights = class_weights / class_weights.sum() * len(class_weights)

    # Reduce weight for 'O' class (index 0) to focus on entities
    if config.reduce_o_weight:
        class_weights[0] = class_weights[0] * 0.5

    class_weights = class_weights.to(config.device)

    print(f"Class weights computed:")
    for i, weight in enumerate(class_weights):
        label_name = config.id2label.get(i, f"ID_{i}")
        print(f"  {label_name} (ID={i}): weight={weight:.4f}, count={int(class_counts[i])}")

    # Store class weights in config for use in training
    config.class_weights = class_weights

    # Training loop
    print("\nStarting training...")
    print(f"Early stopping patience: {config.early_stopping_patience} epochs")
    print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.2e}")

    for epoch in range(start_epoch, config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, config, fgm, pgd, ema)
        print(f"Train Loss: {train_loss:.4f}")

        # Evaluate with EMA
        ema.apply_shadow()
        # Enable debug for first 3 epochs and when performance degrades
        debug_mode = (epoch < 3) or (patience_counter >= 2)
        metrics = evaluate(model, test_loader, config, debug=debug_mode)
        ema.restore()

        print(f"F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Update plateau scheduler based on validation F1
        plateau_scheduler.step(metrics['f1'])

        # Early stopping and model saving
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ema_state_dict': ema.shadow,
                'f1': best_f1,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'patience_counter': patience_counter,
            }, 'best_model.pt')
            print(f"✓ Saved best model with F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            print(f"⚠ No improvement for {patience_counter}/{config.early_stopping_patience} epochs")

            # Check for severe degradation
            if best_f1 - metrics['f1'] > 0.15:
                print(f"⚠ WARNING: Severe performance drop detected! (Best: {best_f1:.4f} -> Current: {metrics['f1']:.4f})")
                print(f"⚠ Consider stopping training or reducing learning rate")

        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            print(f"\n⚠ Early stopping triggered after {patience_counter} epochs without improvement")
            print(f"Best F1: {best_f1:.4f} at epoch {epoch - patience_counter + 1}")
            break

    print(f"\nTraining completed! Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    config = Config()
    train(config)
