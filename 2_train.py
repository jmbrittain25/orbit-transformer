import argparse
import os
import json
import torch
from datetime import datetime
import pandas as pd

import orbit_transformer as ot


def run_experiment(args):
    """Run a single training experiment with the given arguments."""

    args_dict = {k: v for k, v in vars(args).items()}

    trade_name = args_dict["trade_name"]
    learning_rate = args_dict["learning_rate"]
    batch_size = args_dict["batch_size"]
    n_layers = args_dict["n_layers"]
    n_heads = args_dict["n_heads"]
    input_length = args_dict["input_length"]
    dataset_size = args_dict["dataset_size"]
    n_bins = args_dict["n_bins"]
    coordinate_system = args_dict["coordinate_system"]
    d_model = args_dict["d_model"]
    d_ff = args_dict["d_ff"]
    epochs = args_dict["epochs"]

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    train_raw_csv_path = os.path.join(".", "data", f"orbits_HEO_only_dataset_{dataset_size}_raw.csv")
    val_raw_csv_path = os.path.join(".", "data", "orbits_HEO_only_val_dataset_100_raw.csv")

    if coordinate_system == 'spherical':
        token_cols = ["eci_r_token", "eci_theta_token", "eci_phi_token",
                     "eci_vr_token", "eci_vtheta_token", "eci_vphi_token"]
    elif coordinate_system == 'cartesian':
        token_cols = ["eci_x_token", "eci_y_token", "eci_z_token",
                     "eci_vx_token", "eci_vy_token", "eci_vz_token"]
    else:
        raise ValueError("Coordinate system must be 'spherical' or 'cartesian'")

    tokenizer = ot.MultiEciRepresentationTokenizer(
        (n_bins, n_bins, n_bins), (n_bins, n_bins, n_bins),
        (n_bins, n_bins, n_bins), (n_bins, n_bins, n_bins)
    )

    train_raw_df = pd.read_csv(train_raw_csv_path)
    train_df = tokenizer.transform(train_raw_df)
    train_dataset = ot.OrbitTokenDataset(
        train_df,
        token_cols=token_cols,
        input_length=input_length,
        output_length=1,
        stride=1
    )

    val_raw_df = pd.read_csv(val_raw_csv_path)
    val_df = tokenizer.transform(val_raw_df)
    val_dataset = ot.OrbitTokenDataset(
        val_df,
        token_cols=token_cols,
        input_length=input_length,
        output_length=1,
        stride=1
    )

    model = ot.SixTokenDecoderTransferModel(
        pos1_vocab_size=n_bins,
        pos2_vocab_size=n_bins,
        pos3_vocab_size=n_bins,
        vel1_vocab_size=n_bins,
        vel2_vocab_size=n_bins,
        vel3_vocab_size=n_bins,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=0.1,
        max_seq_len=512
    )

    loss_model = ot.SixTokenCrossEntropyLossModel(
        pos1_weight=1.0,
        pos2_weight=1.0,
        pos3_weight=1.0,
        vel1_weight=1.0,
        vel2_weight=1.0,
        vel3_weight=1.0
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    run_name = "_".join([f"{key}{value}" for key, value in args_dict.items() 
                         if key not in ['trade_name', 'epochs']])
    output_dir = os.path.join("orbit_training_runs", trade_name, run_name)
    os.makedirs(output_dir, exist_ok=True)

    trainer = ot.OrbitTrainer(
        transfer_model=model,
        loss_model=loss_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=learning_rate,
        batch_size=batch_size,
        num_workers=0,
        device=device,
        log_dir=output_dir
    )

    print(f"Starting training with args: {args_dict}")
    history = trainer.train(
        epochs=epochs,
        log_every=100
    )
    print(f"Training completed for args: {args_dict}")

    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump({
            'args': args_dict,
            'final_val_loss': history['val_losses'][-1]
        }, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Run an orbit prediction experiment.")
    parser.add_argument('--trade_name', type=str, required=True, help="Name of the trade or experiment group.")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size.")
    parser.add_argument('--n_layers', type=int, default=4, help="Number of transformer layers.")
    parser.add_argument('--n_heads', type=int, default=4, help="Number of attention heads.")
    parser.add_argument('--input_length', type=int, default=32, help="Input sequence length.")
    parser.add_argument('--dataset_size', type=int, default=500, help="Number of orbits in the dataset.")
    parser.add_argument('--n_bins', type=int, default=128, help="Number of bins per component.")
    parser.add_argument('--coordinate_system', type=str, default='spherical', choices=['spherical', 'cartesian'], help="Coordinate system for tokens.")
    parser.add_argument('--d_model', type=int, default=256, help="Dimension of the model embeddings.")
    parser.add_argument('--d_ff', type=int, default=1024, help="Dimension of the feed-forward network.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")

    args = parser.parse_args()

    try:
        run_experiment(args)
    except Exception as e:
        print(f"Error in experiment with args {vars(args)}: {e}")
        with open(os.path.join("orbit_training_runs", args.trade_name, "errors.log"), 'a') as f:
            f.write(f"{datetime.now()} - Error with args {vars(args)}: {e}\n")

if __name__ == "__main__":
    main()
