import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import json
import os
from datetime import datetime
import logging
from tqdm import tqdm

from ..loss_model import LossModel
from ..dataset import OrbitTokenDataset
from ..transfer_model import TransferModel


def collate_fn(batch):
    """
    Collate function to handle six-token sequences (position and velocity).
    Assumes each item in batch has 'input' and 'output' tensors of shape (seq_len, 6).
    
    Args:
        batch: List of dicts, each with 'input' and 'output' keys.
    
    Returns:
        Dict with six input tensors (B, T) and six target tensors (B, output_length).
    """
    pos1_inputs = torch.stack([item['input'][:, 0] for item in batch])
    pos2_inputs = torch.stack([item['input'][:, 1] for item in batch])
    pos3_inputs = torch.stack([item['input'][:, 2] for item in batch])
    vel1_inputs = torch.stack([item['input'][:, 3] for item in batch])
    vel2_inputs = torch.stack([item['input'][:, 4] for item in batch])
    vel3_inputs = torch.stack([item['input'][:, 5] for item in batch])
    pos1_targets = torch.stack([item['output'][:, 0] for item in batch])
    pos2_targets = torch.stack([item['output'][:, 1] for item in batch])
    pos3_targets = torch.stack([item['output'][:, 2] for item in batch])
    vel1_targets = torch.stack([item['output'][:, 3] for item in batch])
    vel2_targets = torch.stack([item['output'][:, 4] for item in batch])
    vel3_targets = torch.stack([item['output'][:, 5] for item in batch])
    return {
        'pos1_input': pos1_inputs,
        'pos2_input': pos2_inputs,
        'pos3_input': pos3_inputs,
        'vel1_input': vel1_inputs,
        'vel2_input': vel2_inputs,
        'vel3_input': vel3_inputs,
        'pos1_target': pos1_targets,
        'pos2_target': pos2_targets,
        'pos3_target': pos3_targets,
        'vel1_target': vel1_targets,
        'vel2_target': vel2_targets,
        'vel3_target': vel3_targets
    }

class OrbitTrainer:
    """
    Trainer class for orbital prediction models supporting six-token sequences
    (position and velocity in spherical or Cartesian coordinates).
    
    Parameters
    ----------
    transfer_model : TransferModel
        The model to train.
    loss_model : LossModel
        The loss function.
    train_dataset : OrbitTokenDataset
        Training dataset.
    val_dataset : OrbitTokenDataset
        Validation dataset.
    lr : float
        Learning rate.
    batch_size : int
        Batch size for training.
    num_workers : int
        Number of workers for data loading.
    device : str
        Device to train on ('cpu', 'cuda', 'mps').
    log_dir : str
        Directory to save logs and checkpoints.
    """

    def __init__(
        self,
        transfer_model: TransferModel,
        loss_model: LossModel,
        train_dataset: OrbitTokenDataset,
        val_dataset: OrbitTokenDataset,
        lr: float = 1e-4,
        batch_size: int = 16,
        num_workers: int = 0,
        device: str = 'cpu',
        log_dir: str = 'runs'
    ):
        self.transfer_model = transfer_model
        self.loss_model = loss_model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.global_step = 0

        self.log_dir = self._setup_logging(log_dir)
        self.logger = logging.getLogger('orbit_trainer')
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.log_dir, 'training.log'))
        fh.setLevel(logging.INFO)
        self.logger.addHandler(fh)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn
        )

        self.optimizer = optim.AdamW(transfer_model.parameters(), lr=lr)
        self.history = {
            'train_losses': [],
            'val_losses': [],
            'epoch_metrics': []
        }

    def _setup_logging(self, base_dir: str) -> str:
        """Setup logging directory with timestamp."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(base_dir, f'run_{timestamp}')
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint and metrics."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.transfer_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'global_step': self.global_step
        }
        checkpoint_path = os.path.join(self.log_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        metrics_path = os.path.join(self.log_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def train(self, epochs: int = 10, save_every: int = 1, log_every: int = 100) -> Dict[str, Any]:
        """Train the model for a specified number of epochs."""
        self.transfer_model.to(self.device)
        best_val_loss = float('inf')

        for epoch in range(1, epochs + 1):
            self.logger.info(f"=== Starting Epoch {epoch}/{epochs} ===")

            train_metrics = self.run_epoch(self.train_loader, training=True, log_every=log_every)
            self.history['train_losses'].append(train_metrics['loss/total'])

            val_metrics = self.run_epoch(self.val_loader, training=False)
            self.history['val_losses'].append(val_metrics['loss/total'])

            if val_metrics['loss/total'] < best_val_loss:
                best_val_loss = val_metrics['loss/total']
                self.save_checkpoint(epoch, {'train': train_metrics, 'val': val_metrics, 'best_val_loss': best_val_loss})

            epoch_metrics = {'epoch': epoch, 'train': train_metrics, 'val': val_metrics}
            self.history['epoch_metrics'].append(epoch_metrics)
            self.logger.info(f"Epoch {epoch} metrics: {epoch_metrics}")

            if epoch % save_every == 0:
                self.save_checkpoint(epoch, epoch_metrics)

        return self.history

    def run_epoch(self, 
                  loader: DataLoader,
                  training: bool = True,
                  log_every: Optional[int] = None) -> Dict[str, float]:
        """
        Run one epoch of training or validation.
        
        Args:
            loader: DataLoader for the epoch.
            training: Whether to train or validate.
            log_every: Log frequency during training.
        
        Returns:
            Dict of average metrics for the epoch.
        """
        self.transfer_model.train(training)
        torch.set_grad_enabled(training)
        
        total_metrics = {}
        num_batches = 0
        
        progress_bar = tqdm(loader, desc='Training' if training else 'Validation')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            pos1_input = batch['pos1_input'].to(self.device)
            pos2_input = batch['pos2_input'].to(self.device)
            pos3_input = batch['pos3_input'].to(self.device)
            vel1_input = batch['vel1_input'].to(self.device)
            vel2_input = batch['vel2_input'].to(self.device)
            vel3_input = batch['vel3_input'].to(self.device)
            targets = {
                'pos1_target': batch['pos1_target'].to(self.device),
                'pos2_target': batch['pos2_target'].to(self.device),
                'pos3_target': batch['pos3_target'].to(self.device),
                'vel1_target': batch['vel1_target'].to(self.device),
                'vel2_target': batch['vel2_target'].to(self.device),
                'vel3_target': batch['vel3_target'].to(self.device)
            }
            
            # Debug logging for first batch
            if batch_idx == 0:
                self.logger.info(f"Input shapes: {pos1_input.shape}, {vel1_input.shape}")
                self.logger.info(f"Target shapes: {targets['pos1_target'].shape}, {targets['vel1_target'].shape}")
            
            # Forward pass
            logits = self.transfer_model(pos1_input, pos2_input, pos3_input,
                                         vel1_input, vel2_input, vel3_input)
            loss, batch_metrics = self.loss_model(logits, targets)
            
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

    def to_dict(self):
        """Serialize trainer configuration."""
        return {
            "transfer_model": self.transfer_model.to_dict(),
            "loss_model": self.loss_model.to_dict(),
            "train_dataset": self.train_dataset.to_dict(),
            "val_dataset": self.val_dataset.to_dict(),
            "lr": self.lr,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "device": self.device,
            "log_dir": self.log_dir
        }
    