from abc import abstractmethod
import torch.nn as nn


class TransferModel(nn.Module):

    @abstractmethod
    def forward(self, r_tokens, theta_tokens, phi_tokens):
        """Perform the forward pass of the model."""
        pass

    @abstractmethod
    def to_dict(self):
        """Serialize parameters to a dictionary."""
        pass
