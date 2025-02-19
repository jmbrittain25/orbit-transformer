import os
import torch

import orbit_transformer as ot


if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load paths from previous script
    total_raw_csv_path = os.path.join(".", "data", "orbits_dataset_1000_raw.csv")
    train_tokenized_csv_path = total_raw_csv_path.replace("raw.csv", "train_tokenized.csv")
    val_tokenized_csv_path = total_raw_csv_path.replace("raw.csv", "val_tokenized.csv")
    test_tokenized_csv_path = total_raw_csv_path.replace("raw.csv", "test_tokenized.csv")

    # Create datasets
    train_dataset = ot.OrbitTokenDataset(
        csv_path=train_tokenized_csv_path,
        input_length=32,  # Context window of 32 timesteps
        output_length=1,  # Predict next position
        stride=1          # Slide window by 1 each time
    )

    val_dataset = ot.OrbitTokenDataset(
        csv_path=val_tokenized_csv_path,
        input_length=32,
        output_length=1,
        stride=1
    )

    # Configure model
    config = ot.ThreeTokenGPTConfig(
        r_vocab_size=200,      # Matches tokenizer.r_bins
        theta_vocab_size=180,  # Matches tokenizer.theta_bins
        phi_vocab_size=360,    # Matches tokenizer.phi_bins
        d_model=256,           # Embedding dimension
        n_heads=4,             # Number of attention heads
        n_layers=4,            # Number of transformer layers
        d_ff=1024,             # Feed-forward dimension
        dropout=0.1,
        max_seq_len=512
    )

    # Create model
    model = ot.ThreeTokenGPTDecoder(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Configure loss
    loss_config = ot.LossConfig(
        cross_entropy_weight=1.0,
        position_weight=5.0,
        r_weight=1.0,
        theta_weight=1.0,
        phi_weight=1.0
    )
    loss_fn = ot.OrbitLossWrapper(loss_config)

    # Create trainer
    trainer = ot.OrbitTrainer(
        model=model,
        loss_fn=loss_fn,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=1e-4,
        batch_size=32,
        num_workers=0,  # Adjust based on your system
        device=device,
        log_dir='orbit_training_runs'
    )

    # Train
    print("Starting training...")
    history = trainer.train(
        epochs=10,
        save_every=1,     # Save checkpoint every epoch
        log_every=100     # Log metrics every 100 batches
    )

    print("Training complete!")
