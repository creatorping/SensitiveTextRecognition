"""
Distributed Data Parallel (DDP) Training Script for Multi-GPU
Optimized for 8x H800 GPUs
"""
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import random
import os
from sklearn.metrics import f1_score, precision_score, recall_score

from config import Config
from model import NestedPrivacyNER
from data_loader import create_dataloader, collate_fn
from adversarial import FGM, PGD, RDrop, EMA


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group with timeout
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=torch.distributed.timedelta(seconds=1800)  # 30 min timeout
    )
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


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


def train_epoch(model, dataloader, optimizer, scheduler, config, rank, fgm=None, pgd=None, ema=None, scaler=None):
    """Train for one epoch with DDP"""
    model.train()
    total_loss = 0

    # Only show progress bar on rank 0
    if rank == 0:
        progress_bar = tqdm(dataloader, desc="Training")
    else:
        progress_bar = dataloader

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Move data to GPU
            input_ids = batch['input_ids'].to(rank, non_blocking=True)
            attention_mask = batch['attention_mask'].to(rank, non_blocking=True)
            span_positions = batch['span_positions'].to(rank, non_blocking=True)
            span_labels = batch['span_labels'].to(rank, non_blocking=True)

        # Get class weights
        class_weights = getattr(config, 'class_weights', None)

        # Forward pass with AMP
        if config.use_amp and scaler is not None:
            with torch.amp.autocast('cuda'):
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
        else:
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
            if rank == 0:
                print(f"WARNING: Skipping batch {batch_idx} due to NaN/Inf loss")
            optimizer.zero_grad()
            continue

        # Backward pass with AMP
        if config.use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            # Check for NaN gradients
            nan_grads = False
            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    if rank == 0:
                        print(f"WARNING: NaN/Inf gradient in {name}")
                    nan_grads = True
                    break

            if nan_grads:
                optimizer.zero_grad()
                continue

            # Optimizer step with AMP
            if config.use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

            if ema is not None:
                ema.update()

        total_loss += loss.item() * config.gradient_accumulation_steps

        if rank == 0:
            progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})

        except Exception as e:
            if rank == 0:
                print(f"\nError in batch {batch_idx}: {e}")
            optimizer.zero_grad()
            continue

    return total_loss / len(dataloader)


def evaluate(model, dataloader, config, rank, debug=False):
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
            input_ids = batch['input_ids'].to(rank)
            attention_mask = batch['attention_mask'].to(rank)
            span_positions = batch['span_positions'].to(rank)
            span_labels = batch['span_labels'].to(rank)

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

    # Convert to tensors
    all_preds_tensor = torch.tensor(all_preds, dtype=torch.long).to(rank)
    all_labels_tensor = torch.tensor(all_labels, dtype=torch.long).to(rank)

    # Get the size of each rank's tensor
    local_size = torch.tensor([all_preds_tensor.size(0)], dtype=torch.long, device=rank)
    size_list = [torch.zeros(1, dtype=torch.long, device=rank) for _ in range(dist.get_world_size())]
    dist.all_gather(size_list, local_size)

    # Get max size for padding
    max_size = max([s.item() for s in size_list])

    # Pad tensors to max_size
    if all_preds_tensor.size(0) < max_size:
        padding = torch.zeros(max_size - all_preds_tensor.size(0), dtype=torch.long, device=rank)
        all_preds_tensor = torch.cat([all_preds_tensor, padding])
        all_labels_tensor = torch.cat([all_labels_tensor, padding])

    # Gather predictions from all GPUs
    gathered_preds = [torch.zeros(max_size, dtype=torch.long, device=rank) for _ in range(dist.get_world_size())]
    gathered_labels = [torch.zeros(max_size, dtype=torch.long, device=rank) for _ in range(dist.get_world_size())]

    dist.all_gather(gathered_preds, all_preds_tensor)
    dist.all_gather(gathered_labels, all_labels_tensor)

    # Only compute metrics on rank 0
    if rank == 0:
        # Trim padded values based on actual sizes
        all_preds_list = []
        all_labels_list = []
        for i, (preds, labels) in enumerate(zip(gathered_preds, gathered_labels)):
            actual_size = size_list[i].item()
            all_preds_list.append(preds[:actual_size].cpu().numpy())
            all_labels_list.append(labels[:actual_size].cpu().numpy())

        all_preds = np.concatenate(all_preds_list)
        all_labels = np.concatenate(all_labels_list)

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
    else:
        return {'f1': 0, 'precision': 0, 'recall': 0}


def train_worker(rank, world_size, config):
    """Training function for each GPU process"""
    try:
        # Setup distributed training
        setup(rank, world_size)
        set_seed(config.seed + rank)  # Different seed per process

    if rank == 0:
        print(f"\n{'='*80}")
        print(f"🚀 Distributed Data Parallel Training")
        print(f"{'='*80}")
        print(f"World Size (GPUs): {world_size}")
        for i in range(world_size):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"{'='*80}\n")

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

    # Wrap dataloaders with DistributedSampler
    train_sampler = DistributedSampler(
        train_loader.dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    test_sampler = DistributedSampler(
        test_loader.dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    # Recreate dataloaders with distributed samplers
    train_loader = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=8,  # Increased for better data loading performance
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=4,  # Prefetch batches for faster loading
        persistent_workers=True  # Keep workers alive between epochs
    )

    test_loader = torch.utils.data.DataLoader(
        test_loader.dataset,
        batch_size=config.batch_size,
        sampler=test_sampler,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=4,
        persistent_workers=True
    )

    # Create model
    if rank == 0:
        print("Creating model...")

    model = NestedPrivacyNER(config).to(rank)

    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # Enable static graph for gradient checkpointing compatibility
    if config.use_gradient_checkpointing:
        model._set_static_graph()

    # Load checkpoint if resuming
    start_epoch = 0
    best_f1 = 0

    if config.resume_from_checkpoint and os.path.exists(config.resume_from_checkpoint):
        if rank == 0:
            print(f"\nLoading checkpoint from {config.resume_from_checkpoint}...")

        # Map to current device
        checkpoint = torch.load(config.resume_from_checkpoint, map_location=f'cuda:{rank}')

        # Load model state dict
        model.module.load_state_dict(checkpoint['model_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint['f1']

        if rank == 0:
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
    ema = EMA(model.module, decay=0.999)

    if config.resume_from_checkpoint and os.path.exists(config.resume_from_checkpoint):
        ema.shadow = checkpoint['ema_state_dict']

    # AMP Scaler
    scaler = torch.amp.GradScaler('cuda') if config.use_amp else None

    # Compute class weights efficiently
    if rank == 0:
        print("\nComputing class weights...")

    class_counts = torch.zeros(len(config.entity_types) + 1, dtype=torch.long).to(rank)

    # Sample only first 1000 batches for class weight computation (faster)
    max_batches = min(1000, len(train_loader))
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= max_batches:
            break
        span_labels = batch['span_labels'].to(rank)
        mask = span_labels != -1
        active_labels = span_labels[mask]
        # Use bincount for faster counting
        if len(active_labels) > 0:
            counts = torch.bincount(active_labels, minlength=len(config.entity_types) + 1)
            class_counts += counts

    # Sum class counts across all GPUs
    dist.all_reduce(class_counts, op=dist.ReduceOp.SUM)

    total_samples = class_counts.sum().item()

    # Compute inverse frequency weights
    class_weights = total_samples / (class_counts + 1e-6)

    # Reduce weight for 'O' class
    if config.reduce_o_weight:
        class_weights[0] = class_weights[0] * 0.5

    # Normalize weights
    class_weights = class_weights / class_weights.sum() * len(class_weights)

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
        # Set epoch for sampler (important for shuffling)
        train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, config, rank, None, None, ema, scaler)

        # Synchronize before evaluation
        dist.barrier()

        if rank == 0:
            print(f"Train Loss: {train_loss:.4f}")

        # Evaluate with EMA
        ema.apply_shadow()
        debug_mode = (epoch < 3)
        metrics = evaluate(model, test_loader, config, rank, debug=debug_mode)
        ema.restore()

        # Synchronize after evaluation
        dist.barrier()

        if rank == 0:
            print(f"F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")

            # Save best model only on rank 0
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']

                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'ema_state_dict': ema.shadow,
                    'f1': best_f1,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                }

                # Save with error handling
                try:
                    torch.save(checkpoint, 'best_model.pt')
                    print(f"✓ Saved best model with F1: {best_f1:.4f}")
                except Exception as e:
                    print(f"Warning: Failed to save checkpoint: {e}")

        # Synchronize all processes before next epoch
        dist.barrier()

    if rank == 0:
        print(f"\nTraining completed! Best F1: {best_f1:.4f}")

    except Exception as e:
        if rank == 0:
            print(f"\nError in training worker {rank}: {e}")
            import traceback
            traceback.print_exc()
        raise
    finally:
        # Always cleanup
        cleanup()


def main():
    """Main entry point for distributed training"""
    config = Config()

    # Use only the first 2 GPUs (GPU 0 and GPU 1)
    world_size = 2

    # Set visible devices to only use GPU 0 and GPU 1
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    if torch.cuda.device_count() < 2:
        print("Warning: Less than 2 GPUs available. Consider using train.py instead.")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        return

    print(f"Using {world_size} GPUs: GPU 0 and GPU 1")

    # Spawn processes for each GPU
    mp.spawn(
        train_worker,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()
