import torch.nn as nn

from .loss import LossModel


class PositionCrossEntropyLossModel(LossModel):
    def __init__(self, r_weight=1.0, theta_weight=1.0, phi_weight=1.0):

        self.r_weight = r_weight
        self.theta_weight = theta_weight
        self.phi_weight = phi_weight

        self.ce_r = nn.CrossEntropyLoss()
        self.ce_theta = nn.CrossEntropyLoss()
        self.ce_phi = nn.CrossEntropyLoss()
        
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
        
        # Combine losses
        weighted_r_loss = self.r_weight * r_loss
        weighted_theta_loss = self.theta_weight * theta_loss
        weighted_phi_loss = self.phi_weight * phi_loss
        
        total_loss = (weighted_r_loss + weighted_theta_loss + weighted_phi_loss) / 3.0
        
        metrics = {
            'loss/total': total_loss.item(),
            'loss/r': r_loss.item(),
            'loss/theta': theta_loss.item(),
            'loss/phi': phi_loss.item(),
        }
        
        return total_loss, metrics

    def to_dict(self):
        return {
            "class_name": "PositionCrossEntropyLossModel",
            "r_weight": self.r_weight,
            "theta_weight": self.theta_weight,
            "phi_weight": self.phi_weight
        }
