"""
trainer_graphpde_multistep.py

Advanced Trainer for GraphPDE with rolling multi-step prediction.

Implements NL Framework training techniques:
1. Multi-frequency parameter group updates (different modules update at different rates)
2. Gradient Spike Protection (adaptive to gradient history)
3. Warmup + Cosine Annealing with Restarts scheduler
4. Learnable Learning Rate Controller (hypergradient descent)
5. Per-module learning rate scaling
6. Multi-scale momentum

Rolling Windows:
  From 2001: predicts 2006 -> 2011 -> 2016 -> 2021 (4 steps)
  From 2006: predicts 2011 -> 2016 -> 2021 (3 steps)
  From 2011: predicts 2016 -> 2021 (2 steps)
  From 2016: predicts 2021 (1 step)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime
import csv
import json

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_metrics(predictions, targets):
    """Compute comprehensive evaluation metrics."""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    non_zero_mask = targets > 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((predictions[non_zero_mask] - targets[non_zero_mask]) /
                              targets[non_zero_mask])) * 100
    else:
        mape = 0.0

    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    median_ae = np.median(np.abs(predictions - targets))
    relative_errors = np.abs(predictions - targets) / (targets + 1.0)
    mean_relative_error = np.mean(relative_errors)

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'median_ae': median_ae,
        'mean_relative_error': mean_relative_error
    }


def label_smoothed_mse_loss(predictions, targets, smoothing=0.1):
    """MSE Loss with Label Smoothing for regression."""
    mse_loss = F.mse_loss(predictions, targets)
    smoothed_targets = (1.0 - smoothing) * targets + smoothing * predictions.detach()
    smooth_loss = F.mse_loss(predictions, smoothed_targets)
    total_loss = (1.0 - smoothing) * mse_loss + smoothing * smooth_loss
    return total_loss


# ============================================================================
# GRADIENT SPIKE PROTECTOR
# ============================================================================

class GradientSpikeProtector:
    """
    Protects against gradient spikes that cause model collapse.

    Features:
    - Tracks gradient norm history
    - Detects spikes relative to moving average
    - Applies proportional or emergency clipping
    - Can exclude specific parameter patterns from clipping
    """

    def __init__(
        self,
        model: nn.Module,
        max_grad_norm: float = 2000.0,
        spike_threshold: float = 3.0,
        history_size: int = 20,
        emergency_clip: float = 0.1,
        excluded_patterns: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.max_grad_norm = max_grad_norm
        self.spike_threshold = spike_threshold
        self.history_size = history_size
        self.emergency_clip = emergency_clip
        self.verbose = verbose

        self.grad_history: List[float] = []
        self.spike_count = 0
        self.protected_count = 0

        # Pre-compute which parameters to exclude
        self.excluded_params: Set[int] = set()
        if excluded_patterns:
            for name, param in model.named_parameters():
                if any(pattern in name for pattern in excluded_patterns):
                    self.excluded_params.add(id(param))

    def _compute_grad_norm(self) -> float:
        """Compute total gradient norm (excluding excluded params)."""
        total_norm = 0.0
        for name, param in self.model.named_parameters():
            if param.grad is not None and id(param) not in self.excluded_params:
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

                # Apply clipping (excluding excluded parameters)
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and id(param) not in self.excluded_params:
                            param.grad.mul_(clip_factor)

                result['was_clipped'] = True
                result['clip_factor'] = clip_factor
                self.protected_count += 1

                if self.verbose:
                    print(f"  [SpikeProtector] Step {step}: grad_norm {total_norm:.0f} "
                          f"-> {total_norm * clip_factor:.0f} (spike #{self.spike_count})")

        # Update history
        effective_norm = total_norm if not result['was_clipped'] else total_norm * result['clip_factor']
        self.grad_history.append(effective_norm)
        if len(self.grad_history) > self.history_size * 2:
            self.grad_history.pop(0)

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get protection statistics."""
        return {
            'spike_count': self.spike_count,
            'protected_count': self.protected_count,
            'avg_grad_norm': sum(self.grad_history) / len(self.grad_history) if self.grad_history else 0,
            'max_grad_norm': max(self.grad_history) if self.grad_history else 0,
        }


# ============================================================================
# WARMUP SCHEDULER WITH COSINE ANNEALING
# ============================================================================

class WarmupCosineScheduler:
    """
    Learning rate scheduler with linear warmup and cosine annealing with restarts.

    Schedule:
    1. Linear warmup: LR goes from 0 to base_lr over warmup_steps
    2. Cosine annealing: LR decays following cosine schedule with warm restarts
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int = 500,
        total_steps: int = 100000,
        min_lr_ratio: float = 0.01,
        restart_period: int = 20000,
        restart_mult: float = 1.5,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.restart_period = restart_period
        self.restart_mult = restart_mult

        # Store the target learning rates for each group
        self.target_lrs = []
        for g in optimizer.param_groups:
            target_lr = g['lr'] * g.get('lr_scale', 1.0)
            self.target_lrs.append(target_lr)

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
        """Update learning rates."""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Linear warmup
            factor = self.current_step / self.warmup_steps
        else:
            # Cosine annealing with warm restarts
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
        """Get current effective learning rates."""
        return [g['lr'] * g.get('lr_scale', 1.0) for g in self.optimizer.param_groups]

    def get_lr_factor(self) -> float:
        """Get current LR multiplier."""
        if self.current_step <= self.warmup_steps:
            return self.current_step / self.warmup_steps
        else:
            progress = self.steps_since_restart / self.current_period
            return self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

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


# ============================================================================
# LEARNABLE LR CONTROLLER (Hypergradient Descent)
# ============================================================================

class LearnableLRController:
    """
    Adaptive learning rate controller using hypergradient descent.

    Adjusts LR based on gradient alignment between consecutive steps:
    - If gradients align (same direction) -> increase LR
    - If gradients oppose (oscillating) -> decrease LR

    Features:
    - Dead zone to prevent over-correction
    - Plateau detection for reduction on stagnation
    - Per-group LR scaling
    """

    def __init__(
        self,
        optimizer,
        model: Optional[nn.Module] = None,
        update_interval: int = 50,
        warmup_steps: int = 200,
        meta_lr: float = 0.01,
        momentum: float = 0.9,
        lr_scale_min: float = 0.1,
        lr_scale_max: float = 10.0,
        dead_zone_low: float = -0.2,
        dead_zone_high: float = 0.1,
        plateau_patience: int = 25,
        plateau_threshold: float = 0.001,
        plateau_factor: float = 0.9,
        plateau_cooldown: int = 15,
        plateau_min_scale: float = 0.05,
        verbose: bool = False,
    ):
        self.optimizer = optimizer
        self.model = model
        self.update_interval = update_interval
        self.warmup_steps = warmup_steps
        self.meta_lr = meta_lr
        self.momentum = momentum
        self.lr_scale_min = lr_scale_min
        self.lr_scale_max = lr_scale_max
        self.dead_zone_low = dead_zone_low
        self.dead_zone_high = dead_zone_high
        self.plateau_patience = plateau_patience
        self.plateau_threshold = plateau_threshold
        self.plateau_factor = plateau_factor
        self.plateau_cooldown = plateau_cooldown
        self.plateau_min_scale = plateau_min_scale
        self.verbose = verbose

        self.step_count = 0

        # Initialize per-group state
        self.group_names = [
            g.get('name', f'group_{i}')
            for i, g in enumerate(optimizer.param_groups)
        ]

        self.base_lrs = {}
        for name, g in zip(self.group_names, optimizer.param_groups):
            self.base_lrs[name] = g['lr'] * g.get('lr_scale', 1.0)

        self.log_scales = {name: 0.0 for name in self.group_names}
        self.hypergradient_ema = {name: 0.0 for name in self.group_names}
        self.prev_grads: Dict[str, Optional[torch.Tensor]] = {name: None for name in self.group_names}

        # Loss history
        self.loss_history: List[float] = []
        self.loss_window = 50

        # Plateau tracking
        self.val_loss_history: List[float] = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.epochs_since_plateau_reduction = float('inf')
        self.plateau_reductions = 0

    def _compute_loss_trend(self) -> float:
        """Compute loss trend over recent window."""
        if len(self.loss_history) < 10:
            return 0.0

        recent = self.loss_history[-10:]
        older = self.loss_history[-min(len(self.loss_history), self.loss_window):-10]

        if not older:
            return 0.0

        recent_mean = sum(recent) / len(recent)
        older_mean = sum(older) / len(older)

        return (recent_mean - older_mean) / (older_mean + 1e-8)

    def _compute_hypergradient(self, current_grad, prev_grad) -> float:
        """Compute hypergradient (cosine similarity between gradients)."""
        if prev_grad is None:
            return 0.0
        if current_grad.shape != prev_grad.shape:
            return 0.0

        g_curr = current_grad.flatten()
        g_prev = prev_grad.flatten()

        dot = torch.dot(g_curr, g_prev).item()
        norm_product = (g_curr.norm() * g_prev.norm()).item() + 1e-8

        return dot / norm_product

    def _get_group_gradient(self, group_name: str) -> Optional[torch.Tensor]:
        """Get flattened gradient for a parameter group."""
        for i, g in enumerate(self.optimizer.param_groups):
            if g.get('name', f'group_{i}') == group_name:
                grads = []
                for p in g['params']:
                    if p.grad is not None:
                        grads.append(p.grad.flatten())
                if grads:
                    return torch.cat(grads)
        return None

    def _update_lrs(self):
        """Update learning rates using hypergradient descent."""
        loss_trend = self._compute_loss_trend()

        for name in self.group_names:
            current_grad = self._get_group_gradient(name)
            if current_grad is None:
                continue

            prev_grad = self.prev_grads.get(name)
            hypergradient = self._compute_hypergradient(current_grad, prev_grad)

            self.prev_grads[name] = current_grad.detach().clone()

            # EMA smoothing
            self.hypergradient_ema[name] = (
                self.momentum * self.hypergradient_ema[name] +
                (1 - self.momentum) * hypergradient
            )

            smoothed_hg = self.hypergradient_ema[name]

            # Dead zone
            if self.dead_zone_low < smoothed_hg < self.dead_zone_high:
                continue

            # Adjustment based on loss trend
            adjustment_factor = 1.0
            if loss_trend > 0.05:
                adjustment_factor = 0.5
            elif loss_trend < -0.05:
                adjustment_factor = 1.5

            # Update log-scale
            if smoothed_hg >= self.dead_zone_high:
                effective_hg = smoothed_hg - self.dead_zone_high
            else:
                effective_hg = smoothed_hg - self.dead_zone_low

            self.log_scales[name] += self.meta_lr * adjustment_factor * effective_hg

            # Clamp
            log_min = math.log(self.lr_scale_min)
            log_max = math.log(self.lr_scale_max)
            self.log_scales[name] = max(log_min, min(log_max, self.log_scales[name]))

        self._apply_lrs()

    def _apply_lrs(self):
        """Apply learned LR scales to optimizer."""
        for i, g in enumerate(self.optimizer.param_groups):
            name = g.get('name', f'group_{i}')
            if name in self.log_scales:
                scale = math.exp(self.log_scales[name])
                base_lr = self.base_lrs[name]
                lr_scale = g.get('lr_scale', 1.0)
                if lr_scale > 0:
                    g['lr'] = base_lr * scale / lr_scale

    def step(self, loss: float):
        """Update learning rates based on current state."""
        self.step_count += 1
        self.loss_history.append(loss)

        if len(self.loss_history) > self.loss_window * 2:
            self.loss_history = self.loss_history[-self.loss_window:]

        if self.step_count < self.warmup_steps:
            return

        if self.step_count % self.update_interval != 0:
            return

        self._update_lrs()

    def check_plateau(self, val_loss: float) -> bool:
        """Check for plateau and reduce LRs if detected."""
        self.val_loss_history.append(val_loss)
        self.epochs_since_plateau_reduction += 1

        if val_loss < self.best_val_loss * (1 - self.plateau_threshold):
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            return False
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.plateau_patience:
            if self.epochs_since_plateau_reduction >= self.plateau_cooldown:
                self._reduce_all_lrs()
                self.epochs_since_plateau_reduction = 0
                self.epochs_without_improvement = 0
                self.plateau_reductions += 1
                return True

        return False

    def _reduce_all_lrs(self):
        """Reduce all LR scales by plateau_factor."""
        log_factor = math.log(self.plateau_factor)
        log_min = math.log(self.plateau_min_scale)

        for name in self.log_scales:
            self.log_scales[name] = max(log_min, self.log_scales[name] + log_factor)

        self._apply_lrs()

        if self.verbose:
            print(f"\n[LR Controller] PLATEAU DETECTED! Reducing all LRs by {self.plateau_factor}x")

    def get_effective_lrs(self) -> Dict[str, float]:
        """Get all effective learning rates."""
        lrs = {}
        for name in self.group_names:
            scale = math.exp(self.log_scales.get(name, 0.0))
            base = self.base_lrs.get(name, 1e-3)
            lrs[name] = base * scale
        return lrs

    def state_dict(self):
        return {
            'step_count': self.step_count,
            'log_scales': self.log_scales.copy(),
            'hypergradient_ema': self.hypergradient_ema.copy(),
            'loss_history': self.loss_history.copy(),
            'val_loss_history': self.val_loss_history.copy(),
            'best_val_loss': self.best_val_loss,
            'epochs_without_improvement': self.epochs_without_improvement,
            'plateau_reductions': self.plateau_reductions,
        }

    def load_state_dict(self, state_dict):
        self.step_count = state_dict['step_count']
        self.log_scales = state_dict['log_scales'].copy()
        self.hypergradient_ema = state_dict['hypergradient_ema'].copy()
        self.loss_history = state_dict.get('loss_history', []).copy()
        self.val_loss_history = state_dict.get('val_loss_history', []).copy()
        self.best_val_loss = state_dict.get('best_val_loss', float('inf'))
        self.epochs_without_improvement = state_dict.get('epochs_without_improvement', 0)
        self.plateau_reductions = state_dict.get('plateau_reductions', 0)
        self._apply_lrs()


# ============================================================================
# PARAMETER GROUP CONFIGURATION FOR GRAPHPDE
# ============================================================================

# Configuration for each parameter group
# Format: name, patterns, lr_scale, update_freq, weight_decay, grad_clip
GRAPHPDE_PARAM_GROUPS = [
    # Physics parameters - very conservative updates
    {
        'name': 'diffusion',
        'patterns': ['diffusion_coef', 'diffusion_weight'],
        'lr_scale': 0.01,
        'update_freq': 10,
        'weight_decay': 0.0,
        'grad_clip': 0.1,
    },
    {
        'name': 'physics_rates',
        'patterns': ['growth_rates', 'emigration_rates', 'immigration_rates',
                     'carrying_capacity', 'individual_capacity'],
        'lr_scale': 0.05,
        'update_freq': 5,
        'weight_decay': 0.0,
        'grad_clip': 0.5,
    },
    # Combination weights - slow to prevent oscillation
    {
        'name': 'combination_weights',
        'patterns': ['residual_weight'],
        'lr_scale': 0.1,
        'update_freq': 10,
        'weight_decay': 0.0,
        'grad_clip': 0.5,
    },
    # Embeddings - no weight decay
    {
        'name': 'embeddings',
        'patterns': ['city_embeddings', 'ethnicity_embeddings', 'period_embeddings'],
        'lr_scale': 0.5,
        'update_freq': 1,
        'weight_decay': 0.0,
        'grad_clip': 1.0,
    },
    # Attention module
    {
        'name': 'attention',
        'patterns': ['ethnicity_attention'],
        'lr_scale': 0.3,
        'update_freq': 1,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
    },
    # Census modulator
    {
        'name': 'census_modulator',
        'patterns': ['census_modulator'],
        'lr_scale': 0.5,
        'update_freq': 1,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
    },
    # Main neural network (residual MLP)
    {
        'name': 'residual_mlp',
        'patterns': ['residual_mlp'],
        'lr_scale': 1.0,
        'update_freq': 1,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
    },
]


def create_parameter_groups(
    model: nn.Module,
    base_lr: float = 1e-3,
    default_weight_decay: float = 0.01,
    verbose: bool = True
) -> List[Dict]:
    """
    Create parameter groups for GraphPDE optimizer with different LR scales.

    Args:
        model: GraphPDE model
        base_lr: Base learning rate
        default_weight_decay: Default weight decay
        verbose: Print group information

    Returns:
        List of parameter group dicts for optimizer
    """
    param_groups = []
    assigned_params = set()

    if verbose:
        print("\n" + "=" * 70)
        print("CREATING PARAMETER GROUPS")
        print("=" * 70)

    for config in GRAPHPDE_PARAM_GROUPS:
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
            group_dict = {
                'params': group_params,
                'lr': base_lr,
                'lr_scale': config['lr_scale'],
                'update_freq': config['update_freq'],
                'weight_decay': config['weight_decay'],
                'grad_clip': config['grad_clip'],
                'name': config['name'],
            }
            param_groups.append(group_dict)

            if verbose:
                effective_lr = base_lr * config['lr_scale']
                n_params = sum(p.numel() for p in group_params)
                print(f"  {config['name']:20s}: {len(group_params):3d} tensors, "
                      f"{n_params:>10,} params, lr={effective_lr:.6f}, "
                      f"freq={config['update_freq']}")

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
            'update_freq': 1,
            'weight_decay': default_weight_decay,
            'grad_clip': 1.0,
            'name': 'other',
        })
        if verbose:
            n_params = sum(p.numel() for p in remaining)
            print(f"  {'other':20s}: {len(remaining):3d} tensors, "
                  f"{n_params:>10,} params (default settings)")

    if verbose:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n  Total: {total_params:,} trainable parameters in {len(param_groups)} groups")
        print("=" * 70)

    return param_groups


# ============================================================================
# MULTI-FREQUENCY OPTIMIZER WRAPPER
# ============================================================================

class MultiFrequencyOptimizer:
    """
    Wrapper that handles multi-frequency updates for different parameter groups.

    Different parameter groups can update at different frequencies:
    - Physics parameters: every 10 steps
    - Embeddings: every step
    - etc.
    """

    def __init__(self, optimizer, verbose: bool = False):
        self.optimizer = optimizer
        self.verbose = verbose
        self.global_step = 0

        # Extract update frequencies
        self.group_freqs = {}
        for i, g in enumerate(optimizer.param_groups):
            name = g.get('name', f'group_{i}')
            self.group_freqs[name] = g.get('update_freq', 1)

    def zero_grad(self):
        """Zero all gradients."""
        self.optimizer.zero_grad()

    def step(self):
        """
        Perform optimization step with multi-frequency updates.
        Only updates groups scheduled for this step.
        """
        self.global_step += 1

        # Temporarily save gradients of groups not being updated
        saved_grads = {}

        for i, g in enumerate(self.optimizer.param_groups):
            name = g.get('name', f'group_{i}')
            freq = self.group_freqs.get(name, 1)

            if self.global_step % freq != 0:
                # Not time to update this group - zero its gradients temporarily
                for p in g['params']:
                    if p.grad is not None:
                        saved_grads[id(p)] = p.grad.clone()
                        p.grad.zero_()

        # Perform optimizer step
        self.optimizer.step()

        # Restore gradients (for groups that weren't updated, they accumulate)
        for i, g in enumerate(self.optimizer.param_groups):
            name = g.get('name', f'group_{i}')
            freq = self.group_freqs.get(name, 1)

            if self.global_step % freq != 0:
                for p in g['params']:
                    if id(p) in saved_grads:
                        # Accumulate gradient for next update
                        if p.grad is not None:
                            p.grad.add_(saved_grads[id(p)])
                        else:
                            p.grad = saved_grads[id(p)]

    def get_lr(self) -> Dict[str, float]:
        """Get current learning rates per group."""
        lrs = {}
        for i, g in enumerate(self.optimizer.param_groups):
            name = g.get('name', f'group_{i}')
            lrs[name] = g['lr'] * g.get('lr_scale', 1.0)
        return lrs

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict(),
            'global_step': self.global_step,
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.global_step = state_dict['global_step']


# ============================================================================
# PROGRESS TRACKER
# ============================================================================

class ProgressTracker:
    """Track best metrics during training."""

    def __init__(self):
        self.best_metrics = {
            'loss': float('inf'),
            'r2': float('-inf'),
            'mae': float('inf'),
        }
        self.best_epochs = {
            'loss': 0,
            'r2': 0,
            'mae': 0,
        }

    def update(self, epoch: int, metrics: Dict[str, float]) -> Dict[str, bool]:
        """Update best metrics. Returns dict of which metrics improved."""
        is_best = {}

        if metrics.get('loss', float('inf')) < self.best_metrics['loss']:
            self.best_metrics['loss'] = metrics['loss']
            self.best_epochs['loss'] = epoch
            is_best['loss'] = True
        else:
            is_best['loss'] = False

        if metrics.get('r2', float('-inf')) > self.best_metrics['r2']:
            self.best_metrics['r2'] = metrics['r2']
            self.best_epochs['r2'] = epoch
            is_best['r2'] = True
        else:
            is_best['r2'] = False

        if metrics.get('mae', float('inf')) < self.best_metrics['mae']:
            self.best_metrics['mae'] = metrics['mae']
            self.best_epochs['mae'] = epoch
            is_best['mae'] = True
        else:
            is_best['mae'] = False

        return is_best


# ============================================================================
# MAIN TRAINER CLASS
# ============================================================================

class TrainerGraphPDEMultistep:
    """
    Advanced Trainer for rolling multi-step GraphPDE training.

    Features:
    - Multi-frequency parameter updates
    - Gradient spike protection
    - Warmup + cosine annealing scheduler
    - Learnable LR controller
    - Per-module learning rate scaling
    """

    YEARS = [2001, 2006, 2011, 2016, 2021]
    PERIOD_MAP = {2001: 0, 2006: 1, 2011: 2, 2016: 3}

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        graph,
        optimizer,
        device='cuda',
        checkpoint_dir='./checkpoints',
        # Gradient protection
        use_spike_protection: bool = True,
        max_grad_norm: float = 2000.0,
        spike_threshold: float = 3.0,
        # Scheduler
        use_warmup_scheduler: bool = True,
        warmup_steps: int = 50,
        total_steps: int = 10000,
        min_lr_ratio: float = 0.01,
        # Learnable LR
        use_learnable_lr: bool = True,
        lr_update_interval: int = 10,
        lr_warmup_steps: int = 100,
        # Loss
        label_smoothing: float = 0.1,
        step_loss_weights: Optional[List[float]] = None,
        # Curriculum
        use_curriculum: bool = False,
        curriculum_warmup_epochs: int = 20,
        # Logging
        verbose: bool = True,
        log_interval: int = 10,
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.graph = graph
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.verbose = verbose
        self.log_interval = log_interval

        self.label_smoothing = label_smoothing
        self.step_loss_weights = step_loss_weights or [1.0, 1.0, 1.0, 1.0]
        self.use_curriculum = use_curriculum
        self.curriculum_warmup_epochs = curriculum_warmup_epochs

        # Wrap optimizer with multi-frequency support
        self.mf_optimizer = MultiFrequencyOptimizer(optimizer, verbose=verbose)

        # Gradient spike protection
        self.spike_protector = None
        if use_spike_protection:
            self.spike_protector = GradientSpikeProtector(
                model=model,
                max_grad_norm=max_grad_norm,
                spike_threshold=spike_threshold,
                excluded_patterns=['diffusion_coef'],  # Don't clip physics gradients
                verbose=verbose,
            )

        # Warmup scheduler
        self.scheduler = None
        if use_warmup_scheduler:
            self.scheduler = WarmupCosineScheduler(
                optimizer=optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
            )

        # Learnable LR controller
        self.lr_controller = None
        if use_learnable_lr:
            self.lr_controller = LearnableLRController(
                optimizer=optimizer,
                model=model,
                update_interval=lr_update_interval,
                warmup_steps=lr_warmup_steps,
                verbose=verbose,
            )

        # Progress tracking
        self.progress_tracker = ProgressTracker()

        # Prepare data
        self._prepare_city_mapping()
        self._prepare_ground_truth()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_r2 = float('-inf')
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = []
        self.patience_counter = 0

        # CSV logging
        self.metrics_csv_path = self.checkpoint_dir / 'metrics.csv'
        self.step_csv_path = self.checkpoint_dir / 'step_metrics.csv'
        self._initialize_csv_logging()

    def _prepare_city_mapping(self):
        """Precompute node to city mapping."""
        cities = self.graph['node_features']['city']
        unique_cities = sorted(set(cities))
        city_to_idx = {city: idx for idx, city in enumerate(unique_cities)}
        self.node_to_city = np.array([city_to_idx[city] for city in cities])
        self.node_to_city_tensor = torch.tensor(
            self.node_to_city, dtype=torch.long, device=self.device
        )
        self.n_nodes = self.graph['adjacency'].shape[0]

        if self.verbose:
            print(f"Prepared city mapping: {len(unique_cities)} cities, {self.n_nodes} nodes")

    def _prepare_ground_truth(self):
        """Build ground truth population tensors for each year."""
        if self.verbose:
            print("Preparing ground truth population tensors...")

        self.train_populations = {}
        self.train_census = {}

        for year in [2001, 2006, 2011, 2016]:
            if year in self.train_dataset.full_graph_cache:
                self.train_populations[year] = self.train_dataset.full_graph_cache[year]['population'].to(self.device)
                self.train_census[year] = self.train_dataset.full_graph_cache[year]['census'].to(self.device)

        self._build_2021_population('train')

        self.val_populations = {}
        self.val_census = {}

        for year in [2001, 2006, 2011, 2016]:
            if year in self.val_dataset.full_graph_cache:
                self.val_populations[year] = self.val_dataset.full_graph_cache[year]['population'].to(self.device)
                self.val_census[year] = self.val_dataset.full_graph_cache[year]['census'].to(self.device)

        self._build_2021_population('val')

        if self.verbose:
            print(f"  Train years: {sorted(self.train_populations.keys())}")
            print(f"  Val years: {sorted(self.val_populations.keys())}")

    def _build_2021_population(self, split='train'):
        """Build 2021 ground truth from pop_t1 where year_t == 2016."""
        dataset = self.train_dataset if split == 'train' else self.val_dataset
        populations = self.train_populations if split == 'train' else self.val_populations
        pop_2021 = dataset.get_population_for_year(2016, return_t1=True)
        populations[2021] = pop_2021.to(self.device)

    def _initialize_csv_logging(self):
        """Initialize CSV logging files."""
        # Epoch-level metrics
        with open(self.metrics_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'train_mae', 'train_rmse', 'train_mape', 'train_r2',
                'val_loss', 'val_mae', 'val_rmse', 'val_mape', 'val_r2',
                'learning_rate', 'is_best_loss', 'is_best_r2'
            ])

        # Step-level metrics
        with open(self.step_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'epoch', 'loss', 'grad_norm', 'was_spike', 'lr_factor'
            ])

        if self.verbose:
            print(f"Metrics logged to: {self.metrics_csv_path}")

    def _get_max_steps_for_epoch(self):
        """Get maximum prediction steps for curriculum learning."""
        if not self.use_curriculum:
            return 4

        progress = min(1.0, self.current_epoch / self.curriculum_warmup_epochs)
        max_steps = 1 + int(progress * 3)
        return min(max_steps, 4)

    def _predict_one_step(self, population_t, census_features, period_idx):
        """Run one step of prediction on the full graph."""
        n_nodes = population_t.shape[0]
        device = population_t.device

        diffusion_coef = self.model.diffusion_coef[period_idx]

        full_period_idx = torch.full(
            (n_nodes,), period_idx, dtype=torch.long, device=device
        )
        full_node_idx = torch.arange(n_nodes, dtype=torch.long, device=device)

        def diffusion_fn(state):
            return self.model.diffusion_weight * self.model.graph_laplacian(
                state, diffusion_coef
            )

        def reaction_fn(state):
            return self.model.reaction_module(
                state,
                self.node_to_city_tensor,
                census_features,
                full_node_idx,
                torch.zeros(n_nodes, dtype=torch.long, device=device),
                full_period_idx,
                debug_batch=False
            )

        predicted_population = self.model.ode_solver(
            population_t,
            diffusion_fn,
            reaction_fn,
            T=self.model.integration_time,
            debug_batch=False
        )

        return predicted_population

    def _compute_step_loss(self, predicted, ground_truth):
        """Compute loss between predicted and ground truth populations."""
        non_zero_mask = ground_truth > 0
        if non_zero_mask.sum() == 0:
            return torch.tensor(0.0, device=predicted.device)

        pred_masked = predicted[non_zero_mask]
        gt_masked = ground_truth[non_zero_mask]

        if self.label_smoothing > 0:
            return label_smoothed_mse_loss(pred_masked, gt_masked, self.label_smoothing)
        else:
            return F.mse_loss(pred_masked, gt_masked)

    def _rolling_multistep_pass(self, populations, census_features, max_steps=4):
        """Perform rolling multi-step prediction and compute total loss."""
        total_loss = torch.tensor(0.0, device=self.device)
        step_losses = {}
        n_loss_terms = 0

        all_predictions = []
        all_targets = []

        rolling_windows = [
            (2001, [(0, 2006), (1, 2011), (2, 2016), (3, 2021)]),
            (2006, [(1, 2011), (2, 2016), (3, 2021)]),
            (2011, [(2, 2016), (3, 2021)]),
            (2016, [(3, 2021)]),
        ]

        for start_year, steps in rolling_windows:
            if start_year not in populations:
                continue

            steps = steps[:max_steps]
            current_pop = populations[start_year]

            for step_idx, (period_idx, target_year) in enumerate(steps):
                census_year = [2001, 2006, 2011, 2016][period_idx]
                if census_year not in census_features:
                    continue

                census = census_features[census_year]
                predicted_pop = self._predict_one_step(current_pop, census, period_idx)

                if torch.isnan(predicted_pop).any() or torch.isinf(predicted_pop).any():
                    break

                predicted_pop = torch.clamp(predicted_pop, min=0.0, max=1e6)

                if target_year in populations:
                    gt_pop = populations[target_year]
                    step_loss = self._compute_step_loss(predicted_pop, gt_pop)
                    weight = self.step_loss_weights[min(step_idx, len(self.step_loss_weights) - 1)]

                    if not (torch.isnan(step_loss) or torch.isinf(step_loss)):
                        total_loss = total_loss + weight * step_loss
                        n_loss_terms += 1
                        step_losses[f"{start_year}_{target_year}"] = step_loss.item()

                    non_zero_mask = gt_pop > 0
                    if non_zero_mask.sum() > 0:
                        all_predictions.append(predicted_pop[non_zero_mask].detach())
                        all_targets.append(gt_pop[non_zero_mask].detach())

                current_pop = predicted_pop

        if n_loss_terms > 0:
            total_loss = total_loss / n_loss_terms

        return total_loss, step_losses, all_predictions, all_targets

    def train_epoch(self):
        """Train for one epoch using rolling multi-step prediction."""
        self.model.train()
        max_steps = self._get_max_steps_for_epoch()

        # Zero gradients
        self.mf_optimizer.zero_grad()

        # Forward pass
        total_loss, step_losses, all_predictions, all_targets = self._rolling_multistep_pass(
            self.train_populations, self.train_census, max_steps=max_steps
        )

        # Safety check
        if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss > 1e8:
            if self.verbose:
                print(f"  Epoch {self.current_epoch}: Invalid loss! Skipping...")
            return {'loss': float('inf'), 'mae': 0, 'rmse': 0, 'mape': 0, 'r2': 0}

        # Backward pass
        total_loss.backward()

        # Handle NaN gradients
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)

        # Spike protection
        grad_stats = {'grad_norm': 0, 'was_spike': False}
        if self.spike_protector is not None:
            grad_stats = self.spike_protector.protect(self.global_step)

        # Learnable LR update (before optimizer step, after backward)
        if self.lr_controller is not None:
            self.lr_controller.step(total_loss.item())

        # Optimizer step (with multi-frequency)
        self.mf_optimizer.step()
        self.global_step += 1

        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()

        # Compute metrics
        if all_predictions and all_targets:
            all_preds_cat = torch.cat(all_predictions)
            all_tgts_cat = torch.cat(all_targets)
            metrics = compute_metrics(all_preds_cat, all_tgts_cat)
        else:
            metrics = {'mae': 0, 'rmse': 0, 'mape': 0, 'r2': 0}

        metrics['loss'] = total_loss.item()
        metrics['max_steps'] = max_steps
        metrics['grad_norm'] = grad_stats['grad_norm']
        metrics['was_spike'] = grad_stats['was_spike']

        # Log step-level metrics
        if self.current_epoch % self.log_interval == 0:
            lr_factor = self.scheduler.get_lr_factor() if self.scheduler else 1.0
            with open(self.step_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.global_step, self.current_epoch, total_loss.item(),
                    grad_stats['grad_norm'], grad_stats['was_spike'], lr_factor
                ])

        # Log step losses periodically
        if self.verbose and self.current_epoch % 10 == 0:
            print(f"\n  Step losses (epoch {self.current_epoch}, max_steps={max_steps}):")
            for key, val in sorted(step_losses.items()):
                print(f"    {key}: {val:.4f}")

        return metrics

    @torch.no_grad()
    def validate(self):
        """Validate using rolling multi-step prediction."""
        self.model.eval()

        total_loss, step_losses, all_predictions, all_targets = self._rolling_multistep_pass(
            self.val_populations, self.val_census, max_steps=4
        )

        if all_predictions and all_targets:
            all_preds_cat = torch.cat(all_predictions)
            all_tgts_cat = torch.cat(all_targets)
            metrics = compute_metrics(all_preds_cat, all_tgts_cat)
        else:
            metrics = {'mae': 0, 'rmse': 0, 'mape': 0, 'r2': 0}

        metrics['loss'] = total_loss.item()

        return metrics

    def train(self, n_epochs, early_stopping_patience=20, start_epoch=0):
        """Main training loop with all NL framework features."""
        if self.verbose:
            print("\n" + "=" * 70)
            print("STARTING MULTI-STEP TRAINING WITH NL FRAMEWORK")
            print("=" * 70)
            print(f"Device: {self.device}")
            print(f"Epochs: {n_epochs}")
            print(f"Spike protection: {self.spike_protector is not None}")
            print(f"Warmup scheduler: {self.scheduler is not None}")
            print(f"Learnable LR: {self.lr_controller is not None}")
            print(f"Curriculum learning: {self.use_curriculum}")
            print(f"Step loss weights: {self.step_loss_weights}")
            print("=" * 70 + "\n")

        if start_epoch > 0:
            print(f"Resuming from epoch {start_epoch}")

        if start_epoch >= n_epochs:
            print(f"Already at epoch {start_epoch}, no training needed.")
            return

        for epoch in range(start_epoch, n_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['loss'])

            # Validate
            val_metrics = self.validate()
            self.val_losses.append(val_metrics['loss'])

            # Update progress tracker
            is_best = self.progress_tracker.update(epoch, val_metrics)

            # Check plateau (for learnable LR)
            if self.lr_controller is not None:
                self.lr_controller.check_plateau(val_metrics['loss'])

            # Get current LR
            if self.scheduler is not None:
                current_lrs = self.scheduler.get_last_lr()
                current_lr = current_lrs[0] if current_lrs else 0
            else:
                current_lr = self.mf_optimizer.param_groups[0]['lr']

            # Print epoch summary
            if self.verbose:
                max_steps_str = f" (max_steps={train_metrics.get('max_steps', 4)})" if self.use_curriculum else ""
                print(f"\nEpoch {epoch}/{n_epochs}{max_steps_str}")
                print(f"  Train Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.2f}, "
                      f"RMSE: {train_metrics['rmse']:.2f}, R2: {train_metrics['r2']:.4f}")
                print(f"  Val Loss:   {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.2f}, "
                      f"RMSE: {val_metrics['rmse']:.2f}, R2: {val_metrics['r2']:.4f}")
                print(f"  LR: {current_lr:.6f}, Grad Norm: {train_metrics.get('grad_norm', 0):.1f}")

            # Log to CSV
            with open(self.metrics_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    train_metrics['loss'], train_metrics['mae'], train_metrics['rmse'],
                    train_metrics['mape'], train_metrics['r2'],
                    val_metrics['loss'], val_metrics['mae'], val_metrics['rmse'],
                    val_metrics['mape'], val_metrics['r2'],
                    current_lr, is_best['loss'], is_best['r2']
                ])

            # Save metrics
            self.metrics_history.append({
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics
            })

            # Checkpointing
            if is_best['loss']:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_loss_model.pt', val_metrics)
                self.patience_counter = 0
                if self.verbose:
                    print(f"  -> New best model (loss={val_metrics['loss']:.4f}) saved!")

            if is_best['r2']:
                self.best_val_r2 = val_metrics['r2']
                self.save_checkpoint('best_r2_model.pt', val_metrics)
                if self.verbose:
                    print(f"  -> New best model (R2={val_metrics['r2']:.4f}) saved!")

            if not any(is_best.values()):
                self.patience_counter += 1

            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', val_metrics)

            # Always save latest
            self.save_checkpoint('latest_checkpoint.pt', val_metrics)

            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                if self.verbose:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

        if self.verbose:
            print(f"\nTraining complete!")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            print(f"Best validation R2: {self.best_val_r2:.4f}")

    def save_checkpoint(self, filename, metrics):
        """Save model checkpoint with all training state."""
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.mf_optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_r2': self.best_val_r2,
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history,
            'patience_counter': self.patience_counter,
            'progress_tracker': {
                'best_metrics': self.progress_tracker.best_metrics,
                'best_epochs': self.progress_tracker.best_epochs,
            },
            'training_mode': 'multistep_nl_framework',
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.lr_controller is not None:
            checkpoint['lr_controller_state_dict'] = self.lr_controller.state_dict()

        if self.spike_protector is not None:
            checkpoint['spike_protector_stats'] = self.spike_protector.get_stats()

        # Save physics parameters
        if hasattr(self.model, 'get_parameters_dict'):
            checkpoint['physics_parameters'] = self.model.get_parameters_dict()

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.mf_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_r2 = checkpoint.get('best_val_r2', float('-inf'))

        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        if 'metrics_history' in checkpoint:
            self.metrics_history = checkpoint['metrics_history']
        if 'patience_counter' in checkpoint:
            self.patience_counter = checkpoint['patience_counter']

        if 'progress_tracker' in checkpoint:
            self.progress_tracker.best_metrics = checkpoint['progress_tracker']['best_metrics']
            self.progress_tracker.best_epochs = checkpoint['progress_tracker']['best_epochs']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.lr_controller is not None and 'lr_controller_state_dict' in checkpoint:
            self.lr_controller.load_state_dict(checkpoint['lr_controller_state_dict'])

        if self.verbose:
            print(f"Loaded checkpoint from epoch {self.current_epoch}")
            print(f"  Best val loss: {self.best_val_loss:.4f}")
            print(f"  Best val R2: {self.best_val_r2:.4f}")

        return self.current_epoch + 1


if __name__ == "__main__":
    print("TrainerGraphPDEMultistep with NL Framework loaded successfully!")
