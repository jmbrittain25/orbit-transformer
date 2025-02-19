import torch
import torch.nn as nn
from typing import Tuple, Dict
from dataclasses import dataclass

from ..tokenizer import SphericalCoordinateTokenizer


@dataclass
class LossConfig:
    """Configuration for the loss function components"""
    cross_entropy_weight: float = 1.0
    position_weight: float = 0.5  # Starting with a larger weight since we're normalizing
    r_weight: float = 1.0
    theta_weight: float = 1.0
    phi_weight: float = 1.0


class OrbitLossWrapper:
    def __init__(self, config: LossConfig):
        self.config = config
        self.ce_r = nn.CrossEntropyLoss()
        self.ce_theta = nn.CrossEntropyLoss()
        self.ce_phi = nn.CrossEntropyLoss()
        
        # Get bin centers for coordinate conversion
        self.tokenizer = SphericalCoordinateTokenizer(
            r_bins=200, theta_bins=180, phi_bins=360,
            theta_min=0.0, theta_max=180.0,
            phi_min=-180.0, phi_max=180.0
        )
        self.r_centers, self.theta_centers, self.phi_centers = self.tokenizer.get_bin_centers()
        
        # Convert bin centers to torch tensors and cache them
        self.r_centers = torch.tensor(self.r_centers, dtype=torch.float32)
        self.theta_centers = torch.tensor(self.theta_centers, dtype=torch.float32)
        self.phi_centers = torch.tensor(self.phi_centers, dtype=torch.float32)
        
    def tokens_to_cartesian(self, r_tokens, theta_tokens, phi_tokens):
        """Convert token indices to Cartesian coordinates."""
        # Move centers to same device as input tokens
        if self.r_centers.device != r_tokens.device:
            self.r_centers = self.r_centers.to(r_tokens.device)
            self.theta_centers = self.theta_centers.to(r_tokens.device)
            self.phi_centers = self.phi_centers.to(r_tokens.device)
            
        # Get coordinate values from bin centers
        r = self.r_centers[r_tokens]
        theta = self.theta_centers[theta_tokens]
        phi = self.phi_centers[phi_tokens]
        
        # Convert to radians
        theta_rad = torch.deg2rad(theta)
        phi_rad = torch.deg2rad(phi)
        
        # Convert to Cartesian
        x = r * torch.sin(theta_rad) * torch.cos(phi_rad)
        y = r * torch.sin(theta_rad) * torch.sin(phi_rad)
        z = r * torch.cos(theta_rad)
        
        return torch.stack([x, y, z], dim=-1)
    
    def position_loss(self, pred_r, pred_theta, pred_phi, 
                     target_r, target_theta, target_phi):
        """
        Calculate radius-normalized position error between predicted and target positions.
        Returns both raw distance and normalized loss.
        """
        # Convert predictions to Cartesian coordinates
        pred_pos = self.tokens_to_cartesian(pred_r, pred_theta, pred_phi)
        target_pos = self.tokens_to_cartesian(target_r, target_theta, target_phi)
        
        # Calculate distances (in km)
        distances = torch.norm(pred_pos - target_pos, dim=-1)
        
        # Get target orbit radii for normalization
        target_r_vals = self.r_centers[target_r]
        
        # Calculate normalized distances (as fraction of orbit radius)
        normalized_distances = distances / target_r_vals
        
        return {
            'raw_distance': distances.mean(),
            'normalized': normalized_distances.mean()
        }
    
    def __call__(self, model_output, targets):
        r_logits, theta_logits, phi_logits = model_output
        B, T, V = r_logits.shape
        
        # Get predictions for last timestep
        r_logits = r_logits[:, -1, :]
        theta_logits = theta_logits[:, -1, :]
        phi_logits = phi_logits[:, -1, :]
        
        # Cross entropy losses
        r_loss = self.ce_r(r_logits, targets['r_target'].squeeze())
        theta_loss = self.ce_theta(theta_logits, targets['theta_target'].squeeze())
        phi_loss = self.ce_phi(phi_logits, targets['phi_target'].squeeze())
        
        # Get predicted token indices
        pred_r = torch.argmax(r_logits, dim=-1)
        pred_theta = torch.argmax(theta_logits, dim=-1)
        pred_phi = torch.argmax(phi_logits, dim=-1)
        
        # Calculate position losses
        pos_losses = self.position_loss(
            pred_r, pred_theta, pred_phi,
            targets['r_target'].squeeze(),
            targets['theta_target'].squeeze(),
            targets['phi_target'].squeeze()
        )
        
        # Combine losses
        weighted_r_loss = self.config.r_weight * r_loss
        weighted_theta_loss = self.config.theta_weight * theta_loss
        weighted_phi_loss = self.config.phi_weight * phi_loss
        weighted_pos_loss = self.config.position_weight * pos_losses['normalized']
        
        total_ce_loss = (weighted_r_loss + weighted_theta_loss + weighted_phi_loss) / 3.0
        total_loss = self.config.cross_entropy_weight * total_ce_loss + weighted_pos_loss
        
        metrics = {
            'loss/total': total_loss.item(),
            'loss/r': r_loss.item(),
            'loss/theta': theta_loss.item(),
            'loss/phi': phi_loss.item(),
            'loss/ce_total': total_ce_loss.item(),
            'loss/position/km': pos_losses['raw_distance'].item(),
            'loss/position/normalized': pos_losses['normalized'].item()
        }
        
        return total_loss, metrics
