import torch.nn as nn

from .loss import LossModel


class SixTokenCrossEntropyLossModel(LossModel):
    def __init__(self, pos1_weight=1.0, pos2_weight=1.0, pos3_weight=1.0,
                 vel1_weight=1.0, vel2_weight=1.0, vel3_weight=1.0):
        """
        Loss model for six-token predictions with individual cross-entropy losses.

        Args:
            pos1_weight (float): Weight for position token 1 loss.
            pos2_weight (float): Weight for position token 2 loss.
            pos3_weight (float): Weight for position token 3 loss.
            vel1_weight (float): Weight for velocity token 1 loss.
            vel2_weight (float): Weight for velocity token 2 loss.
            vel3_weight (float): Weight for velocity token 3 loss.
        """
        super().__init__()
        self.pos1_weight = pos1_weight
        self.pos2_weight = pos2_weight
        self.pos3_weight = pos3_weight
        self.vel1_weight = vel1_weight
        self.vel2_weight = vel2_weight
        self.vel3_weight = vel3_weight

        self.ce_pos1 = nn.CrossEntropyLoss()
        self.ce_pos2 = nn.CrossEntropyLoss()
        self.ce_pos3 = nn.CrossEntropyLoss()
        self.ce_vel1 = nn.CrossEntropyLoss()
        self.ce_vel2 = nn.CrossEntropyLoss()
        self.ce_vel3 = nn.CrossEntropyLoss()

    def __call__(self, model_output, targets):
        """
        Compute the total loss and metrics.

        Args:
            model_output (tuple): (pos1_logits, pos2_logits, pos3_logits, vel1_logits, vel2_logits, vel3_logits).
            targets (dict): Target tensors with keys 'pos1_target', 'pos2_target', etc.

        Returns:
            tuple: (total_loss, metrics_dict)
        """
        pos1_logits, pos2_logits, pos3_logits, vel1_logits, vel2_logits, vel3_logits = model_output
        B, T, _ = pos1_logits.shape

        # Use last timestep predictions
        pos1_logits = pos1_logits[:, -1, :]
        pos2_logits = pos2_logits[:, -1, :]
        pos3_logits = pos3_logits[:, -1, :]
        vel1_logits = vel1_logits[:, -1, :]
        vel2_logits = vel2_logits[:, -1, :]
        vel3_logits = vel3_logits[:, -1, :]

        # Compute individual losses
        pos1_loss = self.ce_pos1(pos1_logits, targets['pos1_target'].squeeze())
        pos2_loss = self.ce_pos2(pos2_logits, targets['pos2_target'].squeeze())
        pos3_loss = self.ce_pos3(pos3_logits, targets['pos3_target'].squeeze())
        vel1_loss = self.ce_vel1(vel1_logits, targets['vel1_target'].squeeze())
        vel2_loss = self.ce_vel2(vel2_logits, targets['vel2_target'].squeeze())
        vel3_loss = self.ce_vel3(vel3_logits, targets['vel3_target'].squeeze())

        # Weighted sum of losses
        total_loss = (self.pos1_weight * pos1_loss +
                      self.pos2_weight * pos2_loss +
                      self.pos3_weight * pos3_loss +
                      self.vel1_weight * vel1_loss +
                      self.vel2_weight * vel2_loss +
                      self.vel3_weight * vel3_loss) / 6.0

        metrics = {
            'loss/total': total_loss.item(),
            'loss/pos1': pos1_loss.item(),
            'loss/pos2': pos2_loss.item(),
            'loss/pos3': pos3_loss.item(),
            'loss/vel1': vel1_loss.item(),
            'loss/vel2': vel2_loss.item(),
            'loss/vel3': vel3_loss.item(),
        }

        return total_loss, metrics

    def to_dict(self):
        """Serialize loss parameters to a dictionary."""
        return {
            "class_name": "SixTokenCrossEntropyLossModel",
            "pos1_weight": self.pos1_weight,
            "pos2_weight": self.pos2_weight,
            "pos3_weight": self.pos3_weight,
            "vel1_weight": self.vel1_weight,
            "vel2_weight": self.vel2_weight,
            "vel3_weight": self.vel3_weight
        }
    