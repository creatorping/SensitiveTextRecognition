"""
Multi-GPU Training Script Optimized for Dual RTX 5090D
使用DistributedDataParallel (DDP) 和 Automatic Mixed Precision (AMP)
"""
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import random
import os
from sklearn.metrics import f1_score, precision_score, recall_score

from config import Config
from model import NestedPrivacyNER
from data_loader import create_dataloader
from adversarial import EMA


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_distributed():
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # 单机多卡环境
        rank = 0
        world_size = torch.cuda.device_count()
        local_rank = 0

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://',
                           world_size=world_size, rank=rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()


def compute_loss(logits, labels, label_smoothing=0.0, class_weights=None):
    """计算交叉熵损失"""
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


def train_epoch(model, dataloader, optimizer, scheduler, scaler, config, ema=None, rank=0):
    """训练一个epoch - 支持混合精度"""
    model.train()
    total_loss = 0

    if rank == 0:
        progress_bar = tqdm(dataloader, desc="Training")
    else:
        progress_bar = dataloader

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(progress_bar):
        # 将数据移到GPU
        input_ids = batch['input_ids'].cuda(non_blocking=True)
        attention_mask = batch['attention_mask'].cuda(non_blocking=True)
        span_positions = batch['span_positions'].cuda(non_blocking=True)
        span_labels = batch['span_labels'].cuda(non_blocking=True)

        class_weights = getattr(config, 'class_weights', None)

        # 混合精度前向传播
        with autocast(enabled=config.use_amp):
            logits = model(input_ids, attention_mask, span_positions)
            loss = compute_loss(logits, span_labels, config.label_smoothing, class_weights)
            loss = loss / config.gradient_accumulation_steps

        # 检查NaN
        if torch.isnan(loss) or torch.isinf(loss):
            if rank == 0:
                print(f"WARNING: Skipping batch {batch_idx} due to NaN/Inf loss")
            optimizer.zero_grad()
            continue

        # 混合精度反向传播
        scaler.scale(loss).backward()

        # 梯度累积
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            # 检查NaN梯度
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

            # 梯度裁剪和优化器步骤
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            if ema is not None:
                ema.update()

        total_loss += loss.item() * config.gradient_accumulation_steps

        if rank == 0 and isinstance(progress_bar, tqdm):
            progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, config, rank=0, debug=False):
    """评估模型"""
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

            # 混合精度推理
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
    """主训练函数 - DDP模式"""
    # 设置分布式环境
    rank, world_size, local_rank = setup_distributed()

    set_seed(config.seed + rank)  # 每个进程使用不同的种子

    if rank == 0:
        print(f"\n{'='*80}")
        print(f"🚀 Distributed Training Configuration (Optimized for RTX 5090D)")
        print(f"{'='*80}")
        print(f"World Size: {world_size}")
        print(f"Rank: {rank}")
        print(f"Local Rank: {local_rank}")
        for i in range(world_size):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Mixed Precision (AMP): {'Enabled' if config.use_amp else 'Disabled'}")
        print(f"Batch Size per GPU: {config.batch_size}")
        print(f"Effective Batch Size: {config.batch_size * world_size * config.gradient_accumulation_steps}")
        print(f"{'='*80}\n")

    # 创建数据加载器
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

    # 创建模型
    if rank == 0:
        print("Creating model...")

    model = NestedPrivacyNER(config).cuda()

    # 使用DDP包装模型
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False)

    # 加载检查点
    start_epoch = 0
    best_f1 = 0

    if config.resume_from_checkpoint and os.path.exists(config.resume_from_checkpoint):
        if rank == 0:
            print(f"\nLoading checkpoint from {config.resume_from_checkpoint}...")

        checkpoint = torch.load(config.resume_from_checkpoint, map_location=f'cuda:{local_rank}')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint['f1']

        if rank == 0:
            print(f"Resumed from epoch {checkpoint['epoch']}, best F1: {best_f1:.4f}")
            print(f"Will continue training from epoch {start_epoch}")

    # 优化器和调度器
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-8  # 防止数值不稳定
    )

    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 加载优化器和调度器状态
    if config.resume_from_checkpoint and os.path.exists(config.resume_from_checkpoint):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # 混合精度训练的GradScaler
    scaler = GradScaler(enabled=config.use_amp)

    if config.resume_from_checkpoint and os.path.exists(config.resume_from_checkpoint) and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # EMA
    ema = EMA(model.module, decay=0.999)

    if config.resume_from_checkpoint and os.path.exists(config.resume_from_checkpoint):
        ema.shadow = checkpoint['ema_state_dict']

    # 计算类权重（仅在rank 0）
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

    # 训练循环
    if rank == 0:
        print("\nStarting training...")

    for epoch in range(start_epoch, config.num_epochs):
        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, config, ema, rank)

        if rank == 0:
            print(f"Train Loss: {train_loss:.4f}")

        # 评估（仅在rank 0）
        if rank == 0:
            ema.apply_shadow()
            debug_mode = (epoch < 3)
            metrics = evaluate(model, test_loader, config, rank, debug=debug_mode)
            ema.restore()

            print(f"F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")

            # 保存最佳模型
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'ema_state_dict': ema.shadow,
                    'f1': best_f1,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                }, 'best_model_5090d.pt')
                print(f"✓ Saved best model with F1: {best_f1:.4f}")

        # 同步所有进程
        dist.barrier()

    if rank == 0:
        print(f"\nTraining completed! Best F1: {best_f1:.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    config = Config()
    train(config)
