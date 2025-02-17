import torch
import torch.nn as nn
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class LossConfig:
    """Configuration for the loss function components"""
    cross_entropy_weight: float = 1.0
    r_weight: float = 1.0
    theta_weight: float = 1.0
    phi_weight: float = 1.0


class OrbitLossWrapper:
    """
    Wrapper for handling orbital position prediction losses.
    Currently implements weighted cross entropy losses for r, theta, and phi predictions.
    
    Parameters
    ----------
    config : LossConfig
        Configuration object containing loss weights
    """
    
    def __init__(self, config: LossConfig):
        self.config = config
        
        # Initialize cross entropy losses for each dimension
        self.ce_r = nn.CrossEntropyLoss()
        self.ce_theta = nn.CrossEntropyLoss()
        self.ce_phi = nn.CrossEntropyLoss()
    
    def __call__(self, 
                 model_output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                 targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the total loss and individual loss components.
        
        Parameters
        ----------
        model_output : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple of (r_logits, theta_logits, phi_logits) from model
            Each has shape (batch_size, seq_len, vocab_size)
        targets : Dict[str, torch.Tensor]
            Dictionary containing 'r_target', 'theta_target', 'phi_target'
            Each has shape (batch_size,) for single-step prediction
            
        Returns
        -------
        total_loss : torch.Tensor
            The combined loss value
        metrics : Dict[str, float]
            Dictionary containing individual loss components
        """
        r_logits, theta_logits, phi_logits = model_output
        B, T, V = r_logits.shape  # batch_size, seq_len, vocab_size
        
        # For single-step prediction, we only care about the last timestep
        r_logits = r_logits[:, -1, :]      # Shape: (B, V)
        theta_logits = theta_logits[:, -1, :]
        phi_logits = phi_logits[:, -1, :]
        
        # Compute individual cross entropy losses
        r_loss = self.ce_r(r_logits, targets['r_target'].squeeze())
        theta_loss = self.ce_theta(theta_logits, targets['theta_target'].squeeze())
        phi_loss = self.ce_phi(phi_logits, targets['phi_target'].squeeze())
        
        # Apply weights
        weighted_r_loss = self.config.r_weight * r_loss
        weighted_theta_loss = self.config.theta_weight * theta_loss
        weighted_phi_loss = self.config.phi_weight * phi_loss
        
        # Combine losses
        total_ce_loss = (weighted_r_loss + weighted_theta_loss + weighted_phi_loss) / 3.0
        
        # Scale by overall CE weight
        total_loss = self.config.cross_entropy_weight * total_ce_loss
        
        # Gather metrics
        metrics = {
            'loss/total': total_loss.item(),
            'loss/r': r_loss.item(),
            'loss/theta': theta_loss.item(),
            'loss/phi': phi_loss.item(),
            'loss/ce_total': total_ce_loss.item()
        }
        
        return total_loss, metrics
