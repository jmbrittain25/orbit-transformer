import os
import yaml
import json

from ..dataset import OrbitTokenDataset
from ..trainer import OrbitTrainer


class Experiment:
    def __init__(self, data_handler, tokenizer, model, loss,
                 input_length=32, output_length=1, stride=1,
                 lr=1e-4, batch_size=32, epochs=10, log_dir="runs/experiment_1"):
        """
        Initialize the Experiment with a combined DataHandler.
        
        Args:
            data_handler (DataHandler): Instance handling data generation and splitting.
            tokenizer (Tokenizer): Tokenizer instance.
            model (Model): Model instance.
            loss (Loss): Loss function instance.
            input_length (int): Input sequence length.
            output_length (int): Output sequence length.
            stride (int): Stride for dataset creation.
            lr (float): Learning rate.
            batch_size (int): Batch size for training.
            epochs (int): Number of training epochs.
            log_dir (str): Directory for logs.
        """
        self.data_handler = data_handler
        self.tokenizer = tokenizer
        self.model = model
        self.loss = loss
        self.input_length = input_length
        self.output_length = output_length
        self.stride = stride
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.log_dir = log_dir
        self.output_dir = "experiment_output"
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, train_csv, val_csv):
        """Execute the experiment pipeline."""
        
        # Create datasets
        train_dataset = OrbitTokenDataset(
            csv_path=train_csv,
            input_length=self.input_length,
            output_length=self.output_length,
            stride=self.stride
        )
        val_dataset = OrbitTokenDataset(
            csv_path=val_csv,
            input_length=self.input_length,
            output_length=self.output_length,
            stride=self.stride
        )
        
        # Initialize trainer
        trainer = OrbitTrainer(
            model=self.model,
            loss=self.loss,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            lr=self.lr,
            batch_size=self.batch_size,
            epochs=self.epochs,
            log_dir=self.log_dir
        )
        
        # Train the model
        history = trainer.train()
        
        # Save configuration
        config_file = os.path.join(self.output_dir, "config.yaml")
        with open(config_file, "w") as f:
            yaml.dump(self.to_dict(), f)
        
        # Save training history
        history_file = os.path.join(self.output_dir, "history.json")
        with open(history_file, "w") as f:
            json.dump(history, f)
        
        print(f"Experiment completed. Config saved to {config_file}")

    def to_dict(self):
        """Serialize the entire experiment configuration."""
        return {
            "data_handler": self.data_handler.to_dict(),
            "tokenizer": self.tokenizer.to_dict(),
            "model": self.model.to_dict(),
            "loss": self.loss.to_dict(),
            "input_length": self.input_length,
            "output_length": self.output_length,
            "stride": self.stride,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "log_dir": self.log_dir
        }
    