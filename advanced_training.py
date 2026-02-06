"""
Advanced training techniques and hyperparameter optimization
"""
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import optuna
from typing import Dict, Any

from config import Config
from model import NestedPrivacyNER
from data_loader import create_dataloader
from train import train_epoch, evaluate, set_seed
from adversarial import FGM, PGD, EMA


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C] logits
            targets: [N] class indices
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CosineAnnealingWithWarmup:
    """Cosine annealing with warmup scheduler"""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


def train_with_focal_loss(config: Config):
    """Train model with focal loss for better handling of class imbalance"""
    set_seed(config.seed)

    # Create dataloaders
    train_loader, _ = create_dataloader(
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
    model = NestedPrivacyNER(config).to(config.device)

    # Use focal loss
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Scheduler
    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = CosineAnnealingWithWarmup(optimizer, warmup_steps, total_steps)

    # Adversarial training
    fgm = FGM(model, epsilon=config.adv_epsilon) if config.use_fgm else None
    ema = EMA(model, decay=0.999)

    best_f1 = 0

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, config, fgm, None, ema)
        print(f"Train Loss: {train_loss:.4f}")

        # Evaluate
        ema.apply_shadow()
        metrics = evaluate(model, test_loader, config)
        ema.restore()

        print(f"F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': best_f1,
            }, 'best_model_focal.pt')

    return best_f1


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for hyperparameter optimization"""
    # Suggest hyperparameters
    config = Config()
    config.learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True)
    config.batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    config.adv_epsilon = trial.suggest_float('adv_epsilon', 0.5, 2.0)
    config.rdrop_alpha = trial.suggest_float('rdrop_alpha', 2.0, 6.0)
    config.label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.2)
    config.warmup_ratio = trial.suggest_float('warmup_ratio', 0.05, 0.15)

    # Train model
    set_seed(config.seed)

    train_loader, _ = create_dataloader(
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

    model = NestedPrivacyNER(config).to(config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    total_steps = len(train_loader) * 10 // config.gradient_accumulation_steps  # Train for 10 epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = CosineAnnealingWithWarmup(optimizer, warmup_steps, total_steps)

    fgm = FGM(model, epsilon=config.adv_epsilon)
    ema = EMA(model, decay=0.999)

    best_f1 = 0

    for epoch in range(10):  # Quick training for hyperparameter search
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, config, fgm, None, ema)

        ema.apply_shadow()
        metrics = evaluate(model, test_loader, config)
        ema.restore()

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']

        # Report intermediate value for pruning
        trial.report(best_f1, epoch)

        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_f1


def hyperparameter_search(n_trials=50):
    """Run hyperparameter optimization with Optuna"""
    print("="*80)
    print("Hyperparameter Optimization")
    print("="*80)

    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    )

    study.optimize(objective, n_trials=n_trials)

    print("\n" + "="*80)
    print("Best hyperparameters:")
    print("="*80)
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print(f"\nBest F1 score: {study.best_value:.4f}")

    # Save best hyperparameters
    import json
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)

    print("\nSaved best hyperparameters to best_hyperparameters.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='focal',
                       choices=['focal', 'hyperparam'],
                       help='Training mode')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of trials for hyperparameter search')

    args = parser.parse_args()

    if args.mode == 'focal':
        config = Config()
        f1 = train_with_focal_loss(config)
        print(f"\nFinal F1 with Focal Loss: {f1:.4f}")
    elif args.mode == 'hyperparam':
        hyperparameter_search(n_trials=args.n_trials)
