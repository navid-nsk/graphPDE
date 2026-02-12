"""
trainer_graphpde.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import csv
from tqdm import tqdm


def compute_metrics(predictions, targets):
    """Compute comprehensive evaluation metrics."""
    predictions = predictions.cpu().numpy()
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
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'median_ae': median_ae,
        'mean_relative_error': mean_relative_error
    }
    
    return metrics

def label_smoothed_mse_loss(predictions, targets, smoothing=0.1):
    """
    MSE Loss with Label Smoothing for regression.
    
    Args:
        predictions: Model predictions (batch_size, n_ethnicities)
        targets: Ground truth targets (batch_size, n_ethnicities)
        smoothing: Smoothing factor (0.0 = no smoothing, 1.0 = full smoothing)
                  Recommended: 0.05 to 0.15
    
    Returns:
        Smoothed MSE loss
    """
    # Standard MSE loss (main component)
    mse_loss = F.mse_loss(predictions, targets)
    
    # Smoothed target: mix real target with model's prediction
    # detach() prevents gradients from flowing back through the smoothing
    smoothed_targets = (1.0 - smoothing) * targets + smoothing * predictions.detach()
    
    # MSE with smoothed targets
    smooth_loss = F.mse_loss(predictions, smoothed_targets)
    
    # Combine: mostly use real loss, small amount of smoothed loss
    total_loss = (1.0 - smoothing) * mse_loss + smoothing * smooth_loss
    
    return total_loss
class TrainerGraphPDE:
    """
    Main trainer class for GraphPDE model with timing and loss tracking.
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        graph,
        optimizer,
        device='cuda',
        checkpoint_dir='./checkpoints',
        grad_clip=100.0,
        physics_loss_weight=0.0,
        label_smoothing=0.1
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.graph = graph
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.grad_clip = grad_clip
        self.physics_loss_weight = physics_loss_weight
        self.label_smoothing = label_smoothing 
        
        self._prepare_city_mapping()
        
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = []
        self.patience_counter = 0
        
        self.metrics_csv_path = self.checkpoint_dir / 'metrics.csv'
        self._initialize_metrics_csv()
    
    def _prepare_city_mapping(self):
        """Precompute node to city mapping"""
        cities = self.graph['node_features']['city']
        unique_cities = sorted(set(cities))
        city_to_idx = {city: idx for idx, city in enumerate(unique_cities)}
        self.node_to_city = np.array([city_to_idx[city] for city in cities])
        
    
    def _initialize_metrics_csv(self):
        """Initialize CSV file for metrics"""
        with open(self.metrics_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'train_mae', 'train_rmse', 'train_mape', 'train_r2',
                'val_loss', 'val_mae', 'val_rmse', 'val_mape', 'val_r2',
                'learning_rate'
            ])
    
    def _log_metrics_to_csv(self, epoch, train_metrics, val_metrics, lr):
        """Append metrics to CSV"""
        with open(self.metrics_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics['loss'],
                train_metrics['mae'],
                train_metrics['rmse'],
                train_metrics['mape'],
                train_metrics['r2'],
                val_metrics['loss'],
                val_metrics['mae'],
                val_metrics['rmse'],
                val_metrics['mape'],
                val_metrics['r2'],
                lr
            ])
    
    def _get_batch_city_indices(self, batch):
        """Get city indices for a batch"""
        node_indices = batch['node_idx'].cpu().numpy()
        city_indices = self.node_to_city[node_indices]
        return torch.tensor(city_indices, dtype=torch.long, device=self.device)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)

                # Get city indices
                city_indices = self._get_batch_city_indices(batch)

                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(batch, city_indices)
                
                predictions = output['pop_pred']
                targets = batch['target']
                
                # Safety checks
                if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                    continue

                predictions = torch.clamp(predictions, min=0.0, max=1e6)
                targets = torch.clamp(targets, min=0.0, max=1e6)

                max_pred = predictions.max().item()
                
                # Compute loss
                if self.label_smoothing > 0:
                    loss = label_smoothed_mse_loss(predictions, targets, self.label_smoothing)
                else:
                    loss = F.mse_loss(predictions, targets)

                # Check loss value
                if torch.isnan(loss) or torch.isinf(loss) or loss > 1e8:
                    continue
                
                # Backward pass
                loss.backward()
                
                # Handle NaN gradients
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)

                # Gradient clipping
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                   
                self.optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                all_predictions.append(predictions.detach())
                all_targets.append(targets.detach())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.2e}',
                    'max_pred': f'{max_pred:.1f}'
                })
        
        # Compute epoch metrics
        n_batches = len(self.train_loader)
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        metrics = compute_metrics(all_predictions, all_targets)
        metrics['loss'] = epoch_loss / n_batches
        
        return metrics
    
    @torch.no_grad()
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        
        epoch_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # Get city indices
            city_indices = self._get_batch_city_indices(batch)
            
            output = self.model(batch, city_indices)
            
            predictions = output['pop_pred']
            targets = batch['target']

            # Compute loss
            loss = F.mse_loss(predictions, targets)
            
            epoch_loss += loss.item()
            all_predictions.append(predictions)
            all_targets.append(targets)
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        metrics = compute_metrics(all_predictions, all_targets)
        metrics['loss'] = epoch_loss / len(self.val_loader)
        
        return metrics
    
    def train(self, n_epochs, scheduler=None, early_stopping_patience=10, start_epoch=0):
        """Main training loop"""
        if start_epoch >= n_epochs:
            return
        
        for epoch in range(start_epoch, n_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['loss'])
            
            # Validate
            val_metrics = self.validate()
            self.val_losses.append(val_metrics['loss'])
            
            # Learning rate scheduling
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{n_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.2f}, "
                  f"RMSE: {train_metrics['rmse']:.2f}, R²: {train_metrics['r2']:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.2f}, "
                  f"RMSE: {val_metrics['rmse']:.2f}, R²: {val_metrics['r2']:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Log to CSV
            self._log_metrics_to_csv(epoch, train_metrics, val_metrics, current_lr)
            
            # Save metrics
            self.metrics_history.append({
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics
            })
            
            # Checkpointing
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_model.pt', val_metrics, scheduler)
                self.patience_counter = 0
                print(f"  → New best model saved!")
            else:
                self.patience_counter += 1
            
            # Regular checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', val_metrics, scheduler)
            
            # Always save latest checkpoint
            self.save_checkpoint('latest_checkpoint.pt', val_metrics, scheduler)
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Training complete
        print(f"\nTraining complete")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, filename, metrics, scheduler=None):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        
        physics_params = self.model.get_parameters_dict()
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history,
            'patience_counter': self.patience_counter,
            'physics_parameters': physics_params
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, filename, scheduler=None):
        """Load model checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        if 'metrics_history' in checkpoint:
            self.metrics_history = checkpoint['metrics_history']
        if 'patience_counter' in checkpoint:
            self.patience_counter = checkpoint['patience_counter']
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"  Patience counter: {self.patience_counter}")
        
        return self.current_epoch + 1


if __name__ == "__main__":
    print("TrainerGraphPDE module loaded successfully!")