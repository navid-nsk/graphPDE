"""
trainer_traffic_v2.py

Improved trainer for GraphPDE Traffic model.
Uses techniques from DO_NOT_USE but WITHOUT CMS, SMM, M3.

Key improvements over clean version:
1. Warmup scheduler with cosine annealing and warm restarts
2. Gradient spike protection
3. Multi-scale momentum (simplified)
4. Parameter groups with different learning rates
5. Better loss function with horizon weighting
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
from typing import Dict, Any, Optional, Tuple, List, Set
from tqdm import tqdm
import numpy as np
from pathlib import Path
from datetime import datetime


# =============================================================================
# WARMUP SCHEDULER (from DO_NOT_USE)
# =============================================================================

class WarmupCosineScheduler:
    """
    Learning rate scheduler with linear warmup and cosine annealing with warm restarts.

    Critical for stable training - gradients are huge at initialization,
    so we need to start with very small learning rates and ramp up.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int = 500,
        total_steps: int = 100000,
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

        # Store target learning rates for each group
        self.target_lrs = []
        for g in optimizer.param_groups:
            target_lr = g['lr'] * g.get('lr_scale', 1.0)
            self.target_lrs.append(target_lr)

        # Warmup state
        self.current_step = 0
        self.current_restart = 0
        self.steps_since_restart = 0
        self.current_period = restart_period

        # Initialize to near-zero LR
        self._set_lr(0.0)

    def _set_lr(self, factor: float):
        """Set learning rates with given factor."""
        for group, target_lr in zip(self.optimizer.param_groups, self.target_lrs):
            lr_scale = group.get('lr_scale', 1.0)
            if lr_scale > 0:
                group['lr'] = target_lr * factor / lr_scale
            else:
                group['lr'] = 0.0

    def step(self):
        """Update learning rates. Call after optimizer.step()."""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Linear warmup
            factor = self.current_step / self.warmup_steps
        else:
            # Cosine annealing with warm restarts
            self.steps_since_restart += 1

            # Check for restart
            if self.steps_since_restart >= self.current_period:
                self.current_restart += 1
                self.steps_since_restart = 0
                self.current_period = int(self.current_period * self.restart_mult)

            # Cosine factor
            progress = self.steps_since_restart / self.current_period
            factor = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        self._set_lr(factor)

    def get_last_lr(self) -> List[float]:
        """Get current effective learning rates."""
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
# GRADIENT SPIKE PROTECTION (from DO_NOT_USE)
# =============================================================================

class GradientSpikeProtector:
    """
    Protects against gradient spikes that cause model collapse.
    """

    def __init__(
        self,
        model: nn.Module,
        max_grad_norm: float = 2000.0,
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
        """Compute total gradient norm."""
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm().item() ** 2
        return math.sqrt(total_norm)

    def protect(self, step: int = 0) -> Dict[str, Any]:
        """Check for gradient spike and clip if necessary."""
        total_norm = self._compute_grad_norm()

        result = {
            'grad_norm': total_norm,
            'was_spike': False,
            'was_clipped': False,
            'clip_factor': 1.0
        }

        # Check for spike (need some history first)
        if len(self.grad_history) >= 5:
            mean_grad = sum(self.grad_history[-self.history_size:]) / min(
                len(self.grad_history), self.history_size
            )

            is_spike = total_norm > self.spike_threshold * mean_grad
            is_severe = total_norm > self.max_grad_norm

            if is_spike or is_severe:
                result['was_spike'] = True
                self.spike_count += 1

                # Determine clip factor
                if is_severe:
                    clip_factor = self.emergency_clip
                else:
                    clip_factor = mean_grad / total_norm

                # Apply clipping
                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad.mul_(clip_factor)

                result['was_clipped'] = True
                result['clip_factor'] = clip_factor

                if self.verbose:
                    print(f"  [SpikeProtector] Step {step}: grad_norm {total_norm:.0f} -> {total_norm * clip_factor:.0f}")

        # Update history
        effective_norm = total_norm if not result['was_clipped'] else total_norm * result['clip_factor']
        self.grad_history.append(effective_norm)
        if len(self.grad_history) > self.history_size * 2:
            self.grad_history.pop(0)

        return result


# =============================================================================
# PARAMETER GROUPS (from DO_NOT_USE, simplified)
# =============================================================================

def create_parameter_groups(
    model: nn.Module,
    base_lr: float = 1e-3,
    weight_decay: float = 0.01,
    verbose: bool = True,
) -> List[Dict]:
    """
    Create parameter groups with different learning rates.

    Groups:
    - Physics parameters: very low LR, no weight decay
    - Attention: medium LR
    - ODE/Encoder: standard LR
    - Decoder: standard LR
    - Combination weights: low LR
    """

    configs = [
        # Physics parameters - very conservative
        {
            'name': 'physics',
            'patterns': ['free_flow_speed', 'jam_density', 'relaxation_rate', 'diffusion'],
            'lr_scale': 0.1,
            'weight_decay': 0.0,
        },
        # Combination weights - slow to prevent oscillation
        {
            'name': 'combination_weights',
            'patterns': ['physics_weight', 'attention_weight', 'neural_weight',
                        'alpha', 'beta', 'term_weights'],
            'lr_scale': 0.2,
            'weight_decay': 0.0,
        },
        # Attention
        {
            'name': 'attention',
            'patterns': ['attention', 'attn', 'q_proj', 'k_proj', 'v_proj', 'o_proj'],
            'lr_scale': 0.5,
            'weight_decay': weight_decay,
        },
        # Encoder/Input projection
        {
            'name': 'encoder',
            'patterns': ['encoder', 'input_proj', 'history_encoder'],
            'lr_scale': 1.0,
            'weight_decay': weight_decay,
        },
        # ODE components
        {
            'name': 'ode',
            'patterns': ['reaction', 'laplacian', 'ode', 'graph'],
            'lr_scale': 0.8,
            'weight_decay': weight_decay,
        },
        # Decoder
        {
            'name': 'decoder',
            'patterns': ['decoder', 'output', 'horizon'],
            'lr_scale': 1.0,
            'weight_decay': weight_decay,
        },
        # Normalization - no weight decay
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

    # Catch-all for remaining parameters
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
# METRICS COMPUTATION
# =============================================================================

def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Compute traffic prediction metrics."""
    with torch.no_grad():
        pred_flat = predictions.flatten()
        targ_flat = targets.flatten()

        mse = F.mse_loss(pred_flat, targ_flat).item()
        mae = F.l1_loss(pred_flat, targ_flat).item()
        rmse = np.sqrt(mse)

        # R-squared
        ss_res = ((targ_flat - pred_flat) ** 2).sum()
        ss_tot = ((targ_flat - targ_flat.mean()) ** 2).sum()
        r2 = (1 - (ss_res / (ss_tot + 1e-8))).item()

        # MAPE (1 mph minimum)
        epsilon = 1.0
        mape = (torch.abs(targ_flat - pred_flat) / torch.clamp(torch.abs(targ_flat), min=epsilon)).mean().item() * 100

        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
        }

        # Per-horizon metrics
        if predictions.dim() == 3 and predictions.shape[-1] >= 12:
            for h_idx, h_name in [(2, 'h3'), (5, 'h6'), (11, 'h12')]:
                h_pred = predictions[:, :, h_idx].flatten()
                h_targ = targets[:, :, h_idx].flatten()
                metrics[f'mae_{h_name}'] = F.l1_loss(h_pred, h_targ).item()
                metrics[f'rmse_{h_name}'] = np.sqrt(F.mse_loss(h_pred, h_targ).item())
                metrics[f'mape_{h_name}'] = (torch.abs(h_targ - h_pred) / torch.clamp(torch.abs(h_targ), min=epsilon)).mean().item() * 100

        return metrics


# =============================================================================
# TRAINER
# =============================================================================

class TrafficTrainerV2:
    """
    Improved trainer with:
    - Warmup scheduler with cosine annealing
    - Gradient spike protection
    - Parameter groups with different LR scales
    - Better loss function
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        total_steps: int = None,
        checkpoint_dir: str = './checkpoints',
        use_spike_protection: bool = True,
        max_grad_norm: float = 2000.0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create parameter groups
        print("\nCreating parameter groups:")
        param_groups = create_parameter_groups(model, base_lr=lr, weight_decay=weight_decay)

        # Optimizer
        self.optimizer = AdamW(param_groups)

        # Scheduler
        if total_steps is None:
            total_steps = len(train_loader) * 100  # Assume 100 epochs

        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=0.01,
            restart_period=len(train_loader) * 20,  # Restart every 20 epochs
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

        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_mape = float('inf')
        self.metrics_history = []

        # CSV logging
        self.csv_path = self.checkpoint_dir / 'training_metrics.csv'
        self._csv_initialized = False

    def _init_csv(self, append: bool = False):
        """Initialize CSV file with headers."""
        if append and self.csv_path.exists():
            self._csv_initialized = True
            return

        headers = [
            'epoch', 'train_loss', 'train_mape', 'train_mae', 'train_rmse',
            'val_loss', 'val_mape', 'val_mae', 'val_rmse', 'val_r2',
            'val_mape_h3', 'val_mape_h6', 'val_mape_h12',
            'val_mae_h3', 'val_mae_h6', 'val_mae_h12',
            'val_rmse_h3', 'val_rmse_h6', 'val_rmse_h12',
            'lr', 'free_flow_speed', 'jam_density', 'relaxation_rate',
            'spike_count'
        ]
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        self._csv_initialized = True

    def _log_to_csv(self, epoch: int, train_metrics: Dict, val_metrics: Dict,
                    lr: float, physics_params: Dict, spike_count: int):
        """Log metrics to CSV."""
        row = [
            epoch,
            train_metrics.get('loss', 0),
            train_metrics.get('mape', 0),
            train_metrics.get('mae', 0),
            train_metrics.get('rmse', 0),
            val_metrics.get('loss', 0),
            val_metrics.get('mape', 0),
            val_metrics.get('mae', 0),
            val_metrics.get('rmse', 0),
            val_metrics.get('r2', 0),
            val_metrics.get('mape_h3', 0),
            val_metrics.get('mape_h6', 0),
            val_metrics.get('mape_h12', 0),
            val_metrics.get('mae_h3', 0),
            val_metrics.get('mae_h6', 0),
            val_metrics.get('mae_h12', 0),
            val_metrics.get('rmse_h3', 0),
            val_metrics.get('rmse_h6', 0),
            val_metrics.get('rmse_h12', 0),
            lr,
            physics_params.get('free_flow_speed', 0),
            physics_params.get('jam_density', 0),
            physics_params.get('relaxation_rate', 0),
            spike_count,
        ]
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Improved loss function with:
        - Horizon-weighted MAPE
        - Smooth L1 component for stability
        """
        # Extract benchmark horizons
        pred_h3 = predictions[:, :, 2:3]
        pred_h6 = predictions[:, :, 5:6]
        pred_h12 = predictions[:, :, 11:12]

        targ_h3 = targets[:, :, 2:3]
        targ_h6 = targets[:, :, 5:6]
        targ_h12 = targets[:, :, 11:12]

        # MAPE per horizon
        epsilon = 1.0
        mape_h3 = (torch.abs(pred_h3 - targ_h3) / torch.clamp(torch.abs(targ_h3), min=epsilon)).mean()
        mape_h6 = (torch.abs(pred_h6 - targ_h6) / torch.clamp(torch.abs(targ_h6), min=epsilon)).mean()
        mape_h12 = (torch.abs(pred_h12 - targ_h12) / torch.clamp(torch.abs(targ_h12), min=epsilon)).mean()

        # Weighted MAPE: H3=40%, H6=35%, H12=25% (prioritize short-term)
        horizon_loss = 0.4 * mape_h3 + 0.35 * mape_h6 + 0.25 * mape_h12

        # All horizons MAPE
        all_mape = (torch.abs(predictions - targets) / torch.clamp(torch.abs(targets), min=epsilon)).mean()

        # Smooth L1 for stability (normalized)
        smooth_l1 = F.smooth_l1_loss(predictions / 100, targets / 100)

        # Combined loss
        total_loss = 0.7 * horizon_loss + 0.2 * all_mape + 0.1 * smooth_l1

        return total_loss * 100.0

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

            # Forward
            output = self.model({'input': inputs})
            predictions = output['predictions']

            # Loss
            loss = self._compute_loss(predictions, targets)

            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: NaN/Inf loss, skipping")
                continue

            # Backward
            loss.backward()

            # Spike protection
            if self.spike_protector is not None:
                self.spike_protector.protect(self.global_step)

            # Update
            self.optimizer.step()
            self.scheduler.step()

            # Clip physics parameters
            if hasattr(self.model, 'clip_parameters'):
                self.model.clip_parameters()

            epoch_loss += loss.item()
            all_predictions.append(predictions.detach())
            all_targets.append(targets.detach())

            self.global_step += 1

            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.2e}'})

        # Compute metrics
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        metrics = compute_metrics(all_predictions, all_targets)
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
        metrics = compute_metrics(all_predictions, all_targets)
        metrics['loss'] = epoch_loss / len(self.val_loader)

        return metrics

    def train(
        self,
        n_epochs: int = 100,
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
        print("TRAINING V2" + (" (RESUMED)" if resume else ""))
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Epochs: {start_epoch} -> {n_epochs}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"Warmup steps: {self.scheduler.warmup_steps}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print(f"CSV metrics file: {self.csv_path}")
        if resume:
            print(f"Resuming from epoch {start_epoch}, best MAPE: {self.best_val_mape:.2f}%")

        for epoch in range(start_epoch, n_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            if epoch % val_interval == 0:
                val_metrics = self.validate()

                # Track best
                improved = False
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    improved = True

                if val_metrics['mape'] < self.best_val_mape:
                    self.best_val_mape = val_metrics['mape']
                    best_epoch = epoch
                    improved = True

                    if save_best:
                        self._save_checkpoint('best_model.pt', val_metrics)

                # Early stopping
                if improved:
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Log
                lr = self.scheduler.get_last_lr()[0]
                spike_count = self.spike_protector.spike_count if self.spike_protector else 0

                print(f"\nEpoch {epoch}:")
                print(f"  Train: Loss={train_metrics['loss']:.4f}, MAPE={train_metrics['mape']:.2f}%")
                print(f"  Val:   Loss={val_metrics['loss']:.4f}, MAPE={val_metrics['mape']:.2f}%")
                print(f"         MAE_H3={val_metrics.get('mae_h3', 0):.2f}, MAE_H6={val_metrics.get('mae_h6', 0):.2f}, MAE_H12={val_metrics.get('mae_h12', 0):.2f}")
                print(f"         MAPE_H3={val_metrics.get('mape_h3', 0):.2f}%, MAPE_H6={val_metrics.get('mape_h6', 0):.2f}%, MAPE_H12={val_metrics.get('mape_h12', 0):.2f}%")
                print(f"  LR: {lr:.2e}, Best Val MAPE: {self.best_val_mape:.2f}% (epoch {best_epoch})")

                # Physics parameters
                physics_params = {}
                if hasattr(self.model, 'get_parameters_dict'):
                    physics_params = self.model.get_parameters_dict()
                    print(f"  Physics: v_f={physics_params.get('free_flow_speed', 0):.1f}, jam_dens={physics_params.get('jam_density', 0):.3f}")

                if spike_count > 0:
                    print(f"  Gradient spikes protected: {spike_count}")

                # Store history
                self.metrics_history.append({
                    'epoch': epoch,
                    'train': train_metrics,
                    'val': val_metrics,
                    'lr': lr,
                })

                # Log to CSV
                self._log_to_csv(epoch, train_metrics, val_metrics, lr, physics_params, spike_count)

                # Save last checkpoint
                self._save_checkpoint('last_checkpoint.pt', val_metrics)

                # Save periodic checkpoint
                if epoch > 0 and epoch % 10 == 0:
                    self._save_checkpoint(f'checkpoint_epoch_{epoch}.pt', val_metrics)

                # Early stopping check
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)")
                    break

        # Save final checkpoint
        self._save_checkpoint('final_model.pt', val_metrics)

        # Save history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

        return {
            'best_val_loss': self.best_val_loss,
            'best_val_mape': self.best_val_mape,
            'best_epoch': best_epoch,
            'final_epoch': self.current_epoch,
            'history': self.metrics_history,
        }

    def _save_checkpoint(self, filename: str, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'best_val_loss': self.best_val_loss,
            'best_val_mape': self.best_val_mape,
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, filepath: str, resume_training: bool = False):
        """Load model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if resume_training:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.best_val_mape = checkpoint.get('best_val_mape', float('inf'))
            self.global_step = checkpoint.get('global_step', 0)
            print(f"Resuming training from epoch {self.current_epoch}")
            print(f"  Best val MAPE so far: {self.best_val_mape:.2f}%")
        else:
            self.current_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.best_val_mape = checkpoint.get('best_val_mape', float('inf'))
            print(f"Loaded checkpoint from epoch {self.current_epoch}")


def create_trainer_v2(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = 'cuda',
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    warmup_steps: int = 500,
    checkpoint_dir: str = './checkpoints',
) -> TrafficTrainerV2:
    """Factory function to create trainer."""
    return TrafficTrainerV2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        checkpoint_dir=checkpoint_dir,
    )
