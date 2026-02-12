"""
trainer_epidemiology_v2.py

V2 Trainer for GraphPDE Epidemiology model.
Uses techniques from DO_NOT_USE but WITHOUT CMS, SMM, M3.

Key features:
1. Warmup scheduler with cosine annealing and warm restarts
2. Gradient spike protection
3. Parameter groups with different learning rates
4. MSE + MAE loss with CCC optimization
5. CCC (Concordance Correlation Coefficient) as primary metric

Metrics focus: MSE ↓ | MAE ↓ | CCC ↑
"""

import os
import json
import csv
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm
import numpy as np
from pathlib import Path
from datetime import datetime


# =============================================================================
# WARMUP SCHEDULER
# =============================================================================

class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int = 200,
        total_steps: int = 50000,
        min_lr_ratio: float = 0.01,
        restart_period: int = 5000,
        restart_mult: float = 1.5,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.restart_period = restart_period
        self.restart_mult = restart_mult

        self.target_lrs = []
        for g in optimizer.param_groups:
            target_lr = g['lr'] * g.get('lr_scale', 1.0)
            self.target_lrs.append(target_lr)

        self.current_step = 0
        self.current_restart = 0
        self.steps_since_restart = 0
        self.current_period = restart_period

        self._set_lr(0.0)

    def _set_lr(self, factor: float):
        for group, target_lr in zip(self.optimizer.param_groups, self.target_lrs):
            lr_scale = group.get('lr_scale', 1.0)
            if lr_scale > 0:
                group['lr'] = target_lr * factor / lr_scale
            else:
                group['lr'] = 0.0

    def step(self):
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            factor = self.current_step / self.warmup_steps
        else:
            self.steps_since_restart += 1

            if self.steps_since_restart >= self.current_period:
                self.current_restart += 1
                self.steps_since_restart = 0
                self.current_period = int(self.current_period * self.restart_mult)

            progress = self.steps_since_restart / self.current_period
            factor = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        self._set_lr(factor)

    def get_last_lr(self) -> List[float]:
        return [g['lr'] * g.get('lr_scale', 1.0) for g in self.optimizer.param_groups]

    def state_dict(self):
        return {
            'current_step': self.current_step,
            'current_restart': self.current_restart,
            'steps_since_restart': self.steps_since_restart,
            'current_period': self.current_period,
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']
        self.current_restart = state_dict['current_restart']
        self.steps_since_restart = state_dict['steps_since_restart']
        self.current_period = state_dict['current_period']


# =============================================================================
# GRADIENT SPIKE PROTECTION
# =============================================================================

class GradientSpikeProtector:
    """Protects against gradient spikes."""

    def __init__(
        self,
        model: nn.Module,
        max_grad_norm: float = 1000.0,
        spike_threshold: float = 3.0,
        history_size: int = 20,
        emergency_clip: float = 0.1,
        verbose: bool = False,
    ):
        self.model = model
        self.max_grad_norm = max_grad_norm
        self.spike_threshold = spike_threshold
        self.history_size = history_size
        self.emergency_clip = emergency_clip
        self.verbose = verbose

        self.grad_history: List[float] = []
        self.spike_count = 0

    def _compute_grad_norm(self) -> float:
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm().item() ** 2
        return math.sqrt(total_norm)

    def protect(self, step: int = 0) -> Dict[str, Any]:
        total_norm = self._compute_grad_norm()

        result = {
            'grad_norm': total_norm,
            'was_spike': False,
            'was_clipped': False,
            'clip_factor': 1.0
        }

        if len(self.grad_history) >= 5:
            mean_grad = sum(self.grad_history[-self.history_size:]) / min(
                len(self.grad_history), self.history_size
            )

            is_spike = total_norm > self.spike_threshold * mean_grad
            is_severe = total_norm > self.max_grad_norm

            if is_spike or is_severe:
                result['was_spike'] = True
                self.spike_count += 1

                if is_severe:
                    clip_factor = self.emergency_clip
                else:
                    clip_factor = mean_grad / total_norm

                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad.mul_(clip_factor)

                result['was_clipped'] = True
                result['clip_factor'] = clip_factor

        effective_norm = total_norm if not result['was_clipped'] else total_norm * result['clip_factor']
        self.grad_history.append(effective_norm)
        if len(self.grad_history) > self.history_size * 2:
            self.grad_history.pop(0)

        return result


# =============================================================================
# PARAMETER GROUPS
# =============================================================================

def create_parameter_groups(
    model: nn.Module,
    base_lr: float = 1e-3,
    weight_decay: float = 0.01,
    verbose: bool = True,
) -> List[Dict]:
    """Create parameter groups with different learning rates."""

    configs = [
        {
            'name': 'physics',
            'patterns': ['growth_rate', 'decay_rate', 'log_capacity', 'competition_scale',
                        'diffusion_coef', 'diffusion_weight', 'physics_scale'],
            'lr_scale': 0.1,
            'weight_decay': 0.0,
        },
        {
            'name': 'combination_weights',
            'patterns': ['physics_weight', 'neural_weight'],
            'lr_scale': 0.2,
            'weight_decay': 0.0,
        },
        {
            'name': 'encoder',
            'patterns': ['encoder', 'decoder_encoder'],
            'lr_scale': 1.0,
            'weight_decay': weight_decay,
        },
        {
            'name': 'residual_mlp',
            'patterns': ['residual_mlp'],
            'lr_scale': 1.0,
            'weight_decay': weight_decay,
        },
        {
            'name': 'decoder',
            'patterns': ['horizon_decoder', 'decoder'],
            'lr_scale': 1.0,
            'weight_decay': weight_decay,
        },
        {
            'name': 'normalization',
            'patterns': ['norm', 'ln', 'bn'],
            'lr_scale': 0.5,
            'weight_decay': 0.0,
        },
    ]

    param_groups = []
    assigned_params = set()

    for config in configs:
        group_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in assigned_params:
                continue

            if any(pattern.lower() in name.lower() for pattern in config['patterns']):
                group_params.append(param)
                assigned_params.add(id(param))

        if group_params:
            param_groups.append({
                'params': group_params,
                'lr': base_lr,
                'lr_scale': config['lr_scale'],
                'weight_decay': config['weight_decay'],
                'name': config['name'],
            })

            if verbose:
                effective_lr = base_lr * config['lr_scale']
                print(f"  {config['name']:20s}: {len(group_params):4d} params, lr={effective_lr:.6f}")

    # Remaining parameters
    remaining = []
    for name, param in model.named_parameters():
        if param.requires_grad and id(param) not in assigned_params:
            remaining.append(param)
            assigned_params.add(id(param))
            if verbose:
                print(f"    [unassigned] {name}")

    if remaining:
        param_groups.append({
            'params': remaining,
            'lr': base_lr,
            'lr_scale': 1.0,
            'weight_decay': weight_decay,
            'name': 'other',
        })
        if verbose:
            print(f"  {'other':20s}: {len(remaining):4d} params (default settings)")

    return param_groups


# =============================================================================
# METRICS - Focus on MSE, MAE, CCC
# =============================================================================

def compute_ccc(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Concordance Correlation Coefficient (CCC).

    CCC measures agreement between predictions and targets, combining
    precision (Pearson correlation) and accuracy (bias).

    CCC = (2 * ρ * σ_pred * σ_target) / (σ_pred² + σ_target² + (μ_pred - μ_target)²)

    where ρ is Pearson correlation, σ is standard deviation, μ is mean.

    Range: [-1, 1], where 1 is perfect agreement.

    Note: CCC is scale-invariant, so normalized or raw values give same result.
    """
    pred_flat = predictions.flatten()
    targ_flat = targets.flatten()

    # Means
    pred_mean = pred_flat.mean()
    targ_mean = targ_flat.mean()

    # Variances
    pred_var = pred_flat.var()
    targ_var = targ_flat.var()

    # Covariance
    covariance = ((pred_flat - pred_mean) * (targ_flat - targ_mean)).mean()

    # CCC
    numerator = 2 * covariance
    denominator = pred_var + targ_var + (pred_mean - targ_mean) ** 2

    ccc = (numerator / (denominator + 1e-8)).item()

    return ccc


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    data_mean: float = 0.0,
    data_std: float = 1.0,
) -> Dict[str, float]:
    """
    Compute epidemiology prediction metrics.

    Primary metrics: MSE ↓ | MAE ↓ | CCC ↑

    Args:
        predictions: Model predictions (normalized)
        targets: Ground truth targets (normalized)
        data_mean: Mean used for normalization (to denormalize for MSE/MAE)
        data_std: Std used for normalization (to denormalize for MSE/MAE)

    Returns:
        Dictionary with mse, mae (in raw scale), and ccc
    """
    with torch.no_grad():
        # Denormalize predictions and targets for MSE/MAE
        # raw = normalized * std + mean
        pred_raw = predictions * data_std + data_mean
        targ_raw = targets * data_std + data_mean

        pred_flat = pred_raw.flatten()
        targ_flat = targ_raw.flatten()

        # MSE (lower is better) - in raw scale
        mse = F.mse_loss(pred_flat, targ_flat).item()

        # MAE (lower is better) - in raw scale
        mae = F.l1_loss(pred_flat, targ_flat).item()

        # CCC (higher is better) - PRIMARY METRIC
        # CCC is scale-invariant, but we use raw for consistency
        ccc = compute_ccc(pred_raw, targ_raw)

        metrics = {
            'mse': mse,
            'mae': mae,
            'ccc': ccc,
        }

        return metrics


# =============================================================================
# LOSS FUNCTION WITH CCC
# =============================================================================

def ccc_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute CCC loss (1 - CCC) for optimization.

    Minimizing this loss maximizes CCC.
    """
    pred_flat = predictions.flatten()
    targ_flat = targets.flatten()

    # Means
    pred_mean = pred_flat.mean()
    targ_mean = targ_flat.mean()

    # Variances
    pred_var = pred_flat.var()
    targ_var = targ_flat.var()

    # Covariance
    covariance = ((pred_flat - pred_mean) * (targ_flat - targ_mean)).mean()

    # CCC
    numerator = 2 * covariance
    denominator = pred_var + targ_var + (pred_mean - targ_mean) ** 2

    ccc = numerator / (denominator + 1e-8)

    # Loss is 1 - CCC (so minimizing loss maximizes CCC)
    return 1 - ccc


# =============================================================================
# TRAINER
# =============================================================================

class EpidemiologyTrainerV2:
    """
    V2 Trainer for epidemiology with warmup, spike protection, and param groups.

    Optimizes for: MSE ↓ | MAE ↓ | CCC ↑
    Model selection based on: CCC (higher is better)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_steps: int = 200,
        total_steps: int = None,
        checkpoint_dir: str = './checkpoints',
        use_spike_protection: bool = True,
        max_grad_norm: float = 1000.0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Parameter groups
        print("\nCreating parameter groups:")
        param_groups = create_parameter_groups(model, base_lr=lr, weight_decay=weight_decay)

        # Optimizer
        self.optimizer = AdamW(param_groups)

        # Scheduler
        if total_steps is None:
            total_steps = len(train_loader) * 200

        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=0.01,
            restart_period=len(train_loader) * 30,
            restart_mult=1.5,
        )

        # Spike protection
        self.spike_protector = None
        if use_spike_protection:
            self.spike_protector = GradientSpikeProtector(
                model,
                max_grad_norm=max_grad_norm,
                spike_threshold=3.0,
                verbose=False,
            )

        # Tracking - CCC is primary metric (higher is better)
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_ccc = float('-inf')  # CCC: higher is better
        self.metrics_history = []

        # Store normalization parameters for denormalization during metrics
        if hasattr(model, 'data_mean') and hasattr(model, 'data_std'):
            self.data_mean = model.data_mean.item()
            self.data_std = model.data_std.item()
        else:
            self.data_mean = 0.0
            self.data_std = 1.0

        # CSV logging
        self.csv_path = self.checkpoint_dir / 'training_metrics.csv'
        self._csv_initialized = False

    def _init_csv(self, append: bool = False):
        if append and self.csv_path.exists():
            self._csv_initialized = True
            return

        headers = [
            'epoch', 'train_loss', 'train_mse', 'train_mae', 'train_ccc',
            'val_loss', 'val_mse', 'val_mae', 'val_ccc',
            'lr', 'growth_rate', 'decay_rate', 'diffusion_coef', 'spike_count'
        ]
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        self._csv_initialized = True

    def _log_to_csv(self, epoch: int, train_metrics: Dict, val_metrics: Dict,
                    lr: float, physics_params: Dict, spike_count: int):
        row = [
            epoch,
            train_metrics.get('loss', 0),
            train_metrics.get('mse', 0),
            train_metrics.get('mae', 0),
            train_metrics.get('ccc', 0),
            val_metrics.get('loss', 0),
            val_metrics.get('mse', 0),
            val_metrics.get('mae', 0),
            val_metrics.get('ccc', 0),
            lr,
            physics_params.get('growth_rate', 0),
            physics_params.get('decay_rate', 0),
            physics_params.get('diffusion_coef', 0),
            spike_count,
        ]
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Combined loss: MSE + MAE + CCC_loss

        - MSE: penalizes large errors
        - MAE: robust to outliers
        - CCC_loss: optimizes for agreement/correlation
        """
        mse_loss = F.mse_loss(predictions, targets)
        mae_loss = F.l1_loss(predictions, targets)
        ccc_loss_val = ccc_loss(predictions, targets)

        # Combined: MSE for scale, MAE for robustness, CCC for agreement
        # Weight CCC more heavily since it's our primary metric
        total_loss = 0.3 * mse_loss + 0.2 * mae_loss + 0.5 * ccc_loss_val

        return total_loss

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_loss = 0.0
        all_predictions = []
        all_targets = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch in pbar:
            if 'input' in batch:
                inputs = batch['input'].to(self.device)
            else:
                inputs = batch['features'].to(self.device)
            targets = batch['target'].to(self.device)

            self.optimizer.zero_grad()

            output = self.model({'input': inputs})
            predictions = output['predictions']

            loss = self._compute_loss(predictions, targets)

            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: NaN/Inf loss, skipping")
                continue

            loss.backward()

            if self.spike_protector is not None:
                self.spike_protector.protect(self.global_step)

            self.optimizer.step()
            self.scheduler.step()

            if hasattr(self.model, 'clip_parameters'):
                self.model.clip_parameters()

            epoch_loss += loss.item()
            all_predictions.append(predictions.detach())
            all_targets.append(targets.detach())

            self.global_step += 1

            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.2e}'})

        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        metrics = compute_metrics(
            all_predictions, all_targets,
            data_mean=self.data_mean, data_std=self.data_std
        )
        metrics['loss'] = epoch_loss / len(self.train_loader)

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()

        epoch_loss = 0.0
        all_predictions = []
        all_targets = []

        for batch in tqdm(self.val_loader, desc="Validation"):
            if 'input' in batch:
                inputs = batch['input'].to(self.device)
            else:
                inputs = batch['features'].to(self.device)
            targets = batch['target'].to(self.device)

            output = self.model({'input': inputs})
            predictions = output['predictions']

            loss = self._compute_loss(predictions, targets)

            epoch_loss += loss.item()
            all_predictions.append(predictions)
            all_targets.append(targets)

        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        metrics = compute_metrics(
            all_predictions, all_targets,
            data_mean=self.data_mean, data_std=self.data_std
        )
        metrics['loss'] = epoch_loss / len(self.val_loader)

        return metrics

    def train(
        self,
        n_epochs: int = 200,
        val_interval: int = 1,
        early_stopping_patience: int = 30,
        save_best: bool = True,
        resume: bool = False,
    ) -> Dict[str, Any]:
        """Train the model."""

        start_epoch = self.current_epoch if resume else 0
        self._init_csv(append=resume)

        patience_counter = 0
        best_epoch = start_epoch if resume else 0

        print("\n" + "=" * 70)
        print("TRAINING V2 - Metrics: MSE ↓ | MAE ↓ | CCC ↑" + (" (RESUMED)" if resume else ""))
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Epochs: {start_epoch} -> {n_epochs}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"Primary metric for model selection: CCC (higher is better)")
        print(f"Warmup steps: {self.scheduler.warmup_steps}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")

        for epoch in range(start_epoch, n_epochs):
            self.current_epoch = epoch

            train_metrics = self.train_epoch()

            if epoch % val_interval == 0:
                val_metrics = self.validate()

                improved = False

                # Track best loss
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']

                # Primary metric: CCC (higher is better)
                if val_metrics['ccc'] > self.best_val_ccc:
                    self.best_val_ccc = val_metrics['ccc']
                    best_epoch = epoch
                    improved = True

                    if save_best:
                        self._save_checkpoint('best_model.pt', val_metrics)

                if improved:
                    patience_counter = 0
                else:
                    patience_counter += 1

                lr = self.scheduler.get_last_lr()[0]
                spike_count = self.spike_protector.spike_count if self.spike_protector else 0

                print(f"\nEpoch {epoch}:")
                print(f"  Train: Loss={train_metrics['loss']:.4f}, MSE={train_metrics['mse']:.4f}, MAE={train_metrics['mae']:.4f}, CCC={train_metrics['ccc']:.4f}")
                print(f"  Val:   Loss={val_metrics['loss']:.4f}, MSE={val_metrics['mse']:.4f}, MAE={val_metrics['mae']:.4f}, CCC={val_metrics['ccc']:.4f}")
                print(f"  LR: {lr:.2e}, Best Val CCC: {self.best_val_ccc:.4f} (epoch {best_epoch})")

                physics_params = {}
                if hasattr(self.model, 'get_parameters_dict'):
                    physics_params = self.model.get_parameters_dict()
                    print(f"  Physics: growth={physics_params.get('growth_rate', 0):.4f}, decay={physics_params.get('decay_rate', 0):.4f}")

                self.metrics_history.append({
                    'epoch': epoch,
                    'train': train_metrics,
                    'val': val_metrics,
                    'lr': lr,
                })

                self._log_to_csv(epoch, train_metrics, val_metrics, lr, physics_params, spike_count)
                self._save_checkpoint('last_checkpoint.pt', val_metrics)

                if epoch > 0 and epoch % 10 == 0:
                    self._save_checkpoint(f'checkpoint_epoch_{epoch}.pt', val_metrics)

                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        self._save_checkpoint('final_model.pt', val_metrics)

        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

        return {
            'best_val_loss': self.best_val_loss,
            'best_val_ccc': self.best_val_ccc,
            'best_epoch': best_epoch,
            'final_epoch': self.current_epoch,
            'history': self.metrics_history,
        }

    def _save_checkpoint(self, filename: str, metrics: Dict[str, float]):
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'best_val_loss': self.best_val_loss,
            'best_val_ccc': self.best_val_ccc,
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, filepath: str, resume_training: bool = False):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if resume_training:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.best_val_ccc = checkpoint.get('best_val_ccc', float('-inf'))
            self.global_step = checkpoint.get('global_step', 0)
            print(f"Resuming training from epoch {self.current_epoch}")
        else:
            self.current_epoch = checkpoint['epoch']
            print(f"Loaded checkpoint from epoch {self.current_epoch}")


def create_trainer_v2(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = 'cuda',
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    warmup_steps: int = 200,
    checkpoint_dir: str = './checkpoints',
) -> EpidemiologyTrainerV2:
    """Factory function to create trainer."""
    return EpidemiologyTrainerV2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        checkpoint_dir=checkpoint_dir,
    )
