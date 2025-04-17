from abc import ABC, abstractmethod


class Tokenizer(ABC):
    @abstractmethod
    def transform(self, df):
        """Transform a DataFrame into tokenized form."""
        pass

    @abstractmethod
    def get_bin_centers(self):
        """Return bin centers for all components."""
        pass

    @abstractmethod
    def to_dict(self):
        """Serialize parameters to a dictionary."""
        pass
