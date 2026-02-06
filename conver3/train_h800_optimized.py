"""
Optimized Multi-GPU Training Script for 7x H800 GPUs
针对7个H800 GPU优化的分布式训练脚本
使用DistributedDataParallel实现最佳性能
"""
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import random
import os
from sklearn.metrics import f1_score, precision_score, recall_score
import argparse

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
    torch.backends.cudnn.benchmark = True  # Enable for better performance


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def compute_loss(logits, labels, label_smoothing=0.0, class_weights=None):
    """Compute cross entropy loss with optional class weights"""
    mask = labels != -1
    active_logits = logits[mask]
    active_labels = labels[mask]

    if len(active_labels) == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    if class_weights is not None:
        loss = nn.functional.cross_entropy(
            active_logits,
            active_labels,
            weight=class_weights,
            reduction='mean'
        )
    else:
        loss = nn.functional.cross_entropy(active_logits, active_labels)

    loss = loss + 1e-8
    return loss


def train_epoch(model, dataloader, optimizer, scheduler, config, scaler, fgm=None, pgd=None, ema=None, rank=0):
    """Train for one epoch with mixed precision"""
    model.train()
    total_loss = 0

    if rank == 0:
        progress_bar = tqdm(dataloader, desc="Training")
    else:
        progress_bar = dataloader

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].cuda(non_blocking=True)
        attention_mask = batch['attention_mask'].cuda(non_blocking=True)
        span_positions = batch['span_positions'].cuda(non_blocking=True)
        span_labels = batch['span_labels'].cuda(non_blocking=True)

        class_weights = getattr(config, 'class_weights', None)

        # Mixed precision training
        with autocast(enabled=config.use_amp):
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

        if torch.isnan(loss) or torch.isinf(loss):
            if rank == 0:
                print(f"WARNING: Skipping batch {batch_idx} due to NaN/Inf loss")
            optimizer.zero_grad()
            continue

        # Backward with gradient scaling
        scaler.scale(loss).backward()

        # Adversarial training - FGM
        if config.use_fgm and fgm is not None:
            fgm.attack(emb_name='word_embeddings')
            with autocast(enabled=config.use_amp):
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
                scaler.scale(loss_adv).backward()
            fgm.restore(emb_name='word_embeddings')

        # Gradient accumulation
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            # Unscale gradients and clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            if ema is not None:
                ema.update()

        total_loss += loss.item() * config.gradient_accumulation_steps

        if rank == 0:
            progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, config, rank=0, debug=False):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_labels = []
    debug_samples = []

    with torch.no_grad():
        if rank == 0:
            progress_bar = tqdm(dataloader, desc="Evaluating")
        else:
            progress_bar = dataloader

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].cuda(non_blocking=True)
            attention_mask = batch['attention_mask'].cuda(non_blocking=True)
            span_positions = batch['span_positions'].cuda(non_blocking=True)
            span_labels = batch['span_labels'].cuda(non_blocking=True)

            with autocast(enabled=config.use_amp):
                logits = model(input_ids, attention_mask, span_positions)

            preds = torch.argmax(logits, dim=-1)

            mask = span_labels != -1
            active_preds = preds[mask].cpu().numpy()
            active_labels = span_labels[mask].cpu().numpy()

            all_preds.extend(active_preds)
            all_labels.extend(active_labels)

            if debug and batch_idx < 3 and rank == 0:
                debug_samples.append({
                    'batch_idx': batch_idx,
                    'labels': active_labels[:20],
                    'preds': active_preds[:20],
                })

    if debug and rank == 0:
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
    """Main training function with DDP"""
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        print(f"\n{'='*80}")
        print(f"🚀 Distributed Training Configuration for H800 GPUs")
        print(f"{'='*80}")
        print(f"World Size (Total GPUs): {world_size}")
        print(f"Rank: {rank}")
        print(f"Local Rank: {local_rank}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(local_rank)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1024**3:.2f} GB")
        print(f"Batch Size per GPU: {config.batch_size}")
        print(f"Effective Batch Size: {config.batch_size * world_size * config.gradient_accumulation_steps}")
        print(f"Mixed Precision: {config.use_amp}")
        print(f"{'='*80}\n")

    set_seed(config.seed + rank)

    # Create dataloaders with DistributedSampler
    if rank == 0:
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

    # Wrap with DistributedSampler
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_loader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        train_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=config.num_workers,
            pin_memory=True,
            prefetch_factor=config.prefetch_factor,
            persistent_workers=True
        )

    # Create model
    if rank == 0:
        print("Creating model...")

    model = NestedPrivacyNER(config)
    model = model.cuda()

    # Wrap with DDP
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,  # Set to True if you have unused parameters
            broadcast_buffers=False,  # Reduce communication overhead
            gradient_as_bucket_view=True  # Memory optimization
        )

    # Get actual model for EMA
    actual_model = model.module if isinstance(model, DDP) else model

    # Load checkpoint if resuming
    start_epoch = 0
    best_f1 = 0

    if config.resume_from_checkpoint and os.path.exists(config.resume_from_checkpoint):
        if rank == 0:
            print(f"\nLoading checkpoint from {config.resume_from_checkpoint}...")

        map_location = {'cuda:0': f'cuda:{local_rank}'}
        checkpoint = torch.load(config.resume_from_checkpoint, map_location=map_location)

        actual_model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint['f1']

        if rank == 0:
            print(f"Resumed from epoch {checkpoint['epoch']}, best F1: {best_f1:.4f}")
            print(f"Will continue training from epoch {start_epoch}")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
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

    # Mixed precision scaler
    scaler = GradScaler(enabled=config.use_amp)

    # EMA
    ema = EMA(actual_model, decay=0.999)

    if config.resume_from_checkpoint and os.path.exists(config.resume_from_checkpoint):
        ema.shadow = checkpoint['ema_state_dict']

    # Adversarial training
    fgm = FGM(actual_model, epsilon=config.adv_epsilon) if config.use_fgm else None
    pgd = PGD(actual_model, epsilon=config.adv_epsilon, alpha=config.adv_alpha) if config.use_pgd else None

    # Compute class weights
    if rank == 0:
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

    class_weights = total_samples / (class_counts + 1e-6)

    if config.reduce_o_weight:
        class_weights[0] = class_weights[0] * 0.5

    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = class_weights.cuda()

    if rank == 0:
        print("\nClass weights computed:")
        for i, weight in enumerate(class_weights):
            label_name = config.id2label.get(i, f"ID_{i}")
            print(f"  {label_name} (ID={i}): weight={weight:.4f}, count={int(class_counts[i])}")

    config.class_weights = class_weights

    # Training loop
    if rank == 0:
        print("\nStarting training...")

    for epoch in range(start_epoch, config.num_epochs):
        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        # Set epoch for DistributedSampler
        if world_size > 1 and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, config, scaler, fgm, pgd, ema, rank)

        if rank == 0:
            print(f"Train Loss: {train_loss:.4f}")

        # Evaluate with EMA (only on rank 0)
        if rank == 0:
            ema.apply_shadow()
            debug_mode = (epoch < 3)
            metrics = evaluate(model, test_loader, config, rank, debug=debug_mode)
            ema.restore()

            print(f"F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")

            # Save best model
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': actual_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'ema_state_dict': ema.shadow,
                    'f1': best_f1,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                }, 'best_model_h800.pt')
                print(f"✓ Saved best model with F1: {best_f1:.4f}")

        # Synchronize all processes
        if world_size > 1:
            dist.barrier()

    if rank == 0:
        print(f"\nTraining completed! Best F1: {best_f1:.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    config = Config()
    train(config)
