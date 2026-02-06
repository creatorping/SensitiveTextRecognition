"""
Adversarial Training Methods: FGM and PGD
"""
import torch
import torch.nn as nn


class FGM:
    """Fast Gradient Method for adversarial training"""
    def __init__(self, model, epsilon=1.0):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self, emb_name='word_embeddings'):
        """Generate adversarial examples"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                # Check if gradient exists
                if param.grad is not None:
                    norm = torch.norm(param.grad)
                    if norm != 0 and not torch.isnan(norm):
                        r_at = self.epsilon * param.grad / norm
                        param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        """Restore original embeddings"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD:
    """Projected Gradient Descent for adversarial training"""
    def __init__(self, model, epsilon=1.0, alpha=0.3):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, emb_name='word_embeddings', is_first_attack=False):
        """Generate adversarial examples with PGD"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                # Check if gradient exists
                if param.grad is not None:
                    norm = torch.norm(param.grad)
                    if norm != 0 and not torch.isnan(norm):
                        r_at = self.alpha * param.grad / norm
                        param.data.add_(r_at)
                        param.data = self.project(name, param.data)

    def restore(self, emb_name='word_embeddings'):
        """Restore original embeddings"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data):
        """Project perturbation to epsilon ball"""
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.epsilon:
            r = self.epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        """Backup gradients"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        """Restore gradients"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class RDrop:
    """R-Drop: Regularized Dropout for model smoothing"""
    @staticmethod
    def compute_kl_loss(p, q, pad_mask=None):
        """
        Compute KL divergence between two distributions
        Args:
            p, q: [batch, seq_len, num_classes]
            pad_mask: [batch, seq_len]
        """
        # Add numerical stability
        p_log_softmax = nn.functional.log_softmax(p, dim=-1)
        q_log_softmax = nn.functional.log_softmax(q, dim=-1)
        p_softmax = nn.functional.softmax(p, dim=-1)
        q_softmax = nn.functional.softmax(q, dim=-1)

        # Clamp to prevent log(0)
        p_softmax = torch.clamp(p_softmax, min=1e-8, max=1.0)
        q_softmax = torch.clamp(q_softmax, min=1e-8, max=1.0)

        p_loss = nn.functional.kl_div(
            p_log_softmax,
            q_softmax,
            reduction='none'
        )
        q_loss = nn.functional.kl_div(
            q_log_softmax,
            p_softmax,
            reduction='none'
        )

        # Sum over classes
        p_loss = p_loss.sum(dim=-1)
        q_loss = q_loss.sum(dim=-1)

        # Apply mask if provided
        if pad_mask is not None:
            p_loss = p_loss * pad_mask
            q_loss = q_loss * pad_mask
            # Avoid division by zero
            mask_sum = pad_mask.sum()
            if mask_sum > 0:
                loss = (p_loss.sum() + q_loss.sum()) / (mask_sum * 2)
            else:
                loss = torch.tensor(0.0, device=p.device, requires_grad=True)
        else:
            loss = (p_loss.mean() + q_loss.mean()) / 2

        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print("WARNING: NaN/Inf in KL loss, returning 0")
            return torch.tensor(0.0, device=p.device, requires_grad=True)

        return loss


class EMA:
    """Exponential Moving Average for model smoothing"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        """Register model parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply shadow parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
