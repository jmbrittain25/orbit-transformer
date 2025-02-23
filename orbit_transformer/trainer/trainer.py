import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import json
import os
from datetime import datetime
import logging
from tqdm import tqdm

from ..loss import OrbitLossWrapper


def collate_fn(batch):
    # Separate the input sequences for r, theta, phi
    r_inputs = torch.stack([item['input'][:, 0] for item in batch])
    theta_inputs = torch.stack([item['input'][:, 1] for item in batch])
    phi_inputs = torch.stack([item['input'][:, 2] for item in batch])
    
    # Stack and separate the target sequences
    r_targets = torch.stack([item['output'][:, 0] for item in batch])
    theta_targets = torch.stack([item['output'][:, 1] for item in batch])
    phi_targets = torch.stack([item['output'][:, 2] for item in batch])
    
    return {
        'r_input': r_inputs,
        'theta_input': theta_inputs,
        'phi_input': phi_inputs,
        'r_target': r_targets,
        'theta_target': theta_targets,
        'phi_target': phi_targets
    }


class OrbitTrainer:
    """
    Trainer class specifically designed for orbital position prediction models
    using the OrbitLossWrapper loss function.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to train
    loss_fn : OrbitLossWrapper
        The loss function wrapper
    train_dataset : torch.utils.data.Dataset
        Training dataset
    val_dataset : Optional[torch.utils.data.Dataset]
        Validation dataset
    lr : float
        Learning rate
    batch_size : int
        Batch size for training
    num_workers : int
        Number of workers for data loading
    device : str
        Device to train on ('cpu', 'cuda', 'mps')
    log_dir : str
        Directory to save logs and checkpoints
    """
    
    def __init__(self,
                 model: torch.nn.Module,
                 loss_fn: OrbitLossWrapper,
                 train_dataset: torch.utils.data.Dataset,
                 val_dataset: Optional[torch.utils.data.Dataset] = None,
                 lr: float = 1e-4,
                 batch_size: int = 16,
                 num_workers: int = 0,
                 device: str = 'cpu',
                 log_dir: str = 'runs'):
        
        if not isinstance(loss_fn, OrbitLossWrapper):
            raise TypeError("loss_fn must be an instance of OrbitLossWrapper")
            
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.global_step = 0
        
        # Setup logging
        self.log_dir = self._setup_logging(log_dir)
        
        # Initialize logger
        self.logger = logging.getLogger('orbit_trainer')
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.log_dir, 'training.log'))
        fh.setLevel(logging.INFO)
        self.logger.addHandler(fh)
        
        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn
        )
        
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=True,
                collate_fn=collate_fn
            )
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        
        # Metrics history
        self.history = {
            'train_losses': [],
            'val_losses': [],
            'epoch_metrics': []
        }
    
    def _setup_logging(self, base_dir: str) -> str:
        """Setup logging directory with timestamp"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(base_dir, f'run_{timestamp}')
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint and training state"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'global_step': self.global_step
        }
        
        checkpoint_path = os.path.join(self.log_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save latest metrics
        metrics_path = os.path.join(self.log_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def train(self, epochs: int = 10, save_every: int = 1, log_every: int = 100) -> Dict[str, Any]:
        """
        Train the model for the specified number of epochs.
        
        Parameters
        ----------
        epochs : int
            Number of epochs to train
        save_every : int
            Save checkpoint every N epochs
        log_every : int
            Log metrics every N steps
            
        Returns
        -------
        history : Dict[str, Any]
            Training history containing losses and metrics
        """
        self.model.to(self.device)
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            self.logger.info(f"=== Starting Epoch {epoch}/{epochs} ===")
            
            # Training phase
            train_metrics = self.run_epoch(
                self.train_loader, 
                training=True,
                log_every=log_every
            )
            self.history['train_losses'].append(train_metrics['loss/total'])
            
            # Validation phase
            val_metrics = None
            if self.val_loader:
                val_metrics = self.run_epoch(
                    self.val_loader,
                    training=False
                )
                self.history['val_losses'].append(val_metrics['loss/total'])
                
                # Save best model
                if val_metrics['loss/total'] < best_val_loss:
                    best_val_loss = val_metrics['loss/total']
                    self.save_checkpoint(epoch, {
                        'train': train_metrics,
                        'val': val_metrics,
                        'best_val_loss': best_val_loss
                    })
            
            # Log epoch metrics
            epoch_metrics = {
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics
            }
            self.history['epoch_metrics'].append(epoch_metrics)
            self.logger.info(f"Epoch {epoch} metrics: {epoch_metrics}")
            
            # Regular checkpoint saving
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, epoch_metrics)
        
        return self.history
    
    def run_epoch(self, 
                  loader: DataLoader,
                  training: bool = True,
                  log_every: Optional[int] = None) -> Dict[str, float]:
        """Run one epoch of training or validation"""
        
        self.model.train(training)
        torch.set_grad_enabled(training)
        
        total_metrics = {}
        num_batches = 0
        
        progress_bar = tqdm(loader, desc='Training' if training else 'Validation')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            r_input = batch['r_input'].to(self.device)
            theta_input = batch['theta_input'].to(self.device)
            phi_input = batch['phi_input'].to(self.device)
            targets = {
                'r_target': batch['r_target'].to(self.device),
                'theta_target': batch['theta_target'].to(self.device),
                'phi_target': batch['phi_target'].to(self.device)
            }
            
            # Forward pass
            logits = self.model(r_input, theta_input, phi_input)
            loss, batch_metrics = self.loss_fn(logits, targets)
            
            # Backward pass and optimization
            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.global_step += 1
            
            # Accumulate metrics
            for k, v in batch_metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': batch_metrics['loss/total'],
                'step': self.global_step if training else 'val'
            })
            
            # Log intermediate metrics
            if training and log_every and batch_idx % log_every == 0:
                self.logger.info(
                    f"Step {self.global_step} - "
                    f"Loss: {batch_metrics['loss/total']:.4f}"
                )
        
        # Compute epoch averages
        return {k: v / num_batches for k, v in total_metrics.items()}
