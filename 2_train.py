import os
import torch

import orbit_transformer as ot


if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    total_raw_csv_path = os.path.join(".", "data", "orbits_dataset_1000_raw.csv")
    train_tokenized_csv_path = total_raw_csv_path.replace("raw.csv", "train_tokenized.csv")
    val_tokenized_csv_path = total_raw_csv_path.replace("raw.csv", "val_tokenized.csv")
    test_tokenized_csv_path = total_raw_csv_path.replace("raw.csv", "test_tokenized.csv")

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

    model = ot.ThreeTokenDecoderTransferModel(
        r_vocab_size=200,
        theta_vocab_size=180,
        phi_vocab_size=360,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        dropout=0.1,
        max_seq_len=512
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_model = ot.PositionCrossEntropyLossModel(
        r_weight=1.0,
        theta_weight=1.0,
        phi_weight=1.0
    )

    trainer = ot.OrbitTrainer(
        transfer_model=model,
        loss_model=loss_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=1e-4,
        batch_size=32,
        num_workers=0,
        device=device,
        log_dir='orbit_training_runs'
    )

    print("Starting training...")
    history = trainer.train(
        epochs=1,
        save_every=1,
        log_every=100
    )

    print("Training complete!")
