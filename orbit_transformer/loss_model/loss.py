from abc import ABC, abstractmethod


class LossModel(ABC):
    @abstractmethod
    def __call__(self, model_output, targets):
        """Compute the loss given model output and targets."""
        pass

    @abstractmethod
    def to_dict(self):
        """Serialize parameters to a dictionary."""
        pass
