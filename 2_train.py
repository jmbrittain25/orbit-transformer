import os
import torch
from itertools import product
import json

import orbit_transformer as ot


def run_experiment(trade_name, hyperparams):
    """Run a single training experiment with the given hyperparameters."""
    # Unpack hyperparameters
    learning_rate = hyperparams['learning_rate']
    batch_size = hyperparams['batch_size']
    n_layers = hyperparams['n_layers']
    n_heads = hyperparams['n_heads']
    input_length = hyperparams['input_length']
    dataset_size = hyperparams['dataset_size']
    r_bins = hyperparams['r_bins']
    theta_bins = hyperparams['theta_bins']
    phi_bins = hyperparams['phi_bins']
    epochs = hyperparams['epochs']

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Define dataset paths based on hyperparameters
    total_raw_csv_path = os.path.join(".", "data", f"orbits_dataset_total_{dataset_size}_raw.csv")
    train_raw_csv_path = total_raw_csv_path.replace("total", "train")
    val_raw_csv_path = total_raw_csv_path.replace("total", "val")

    train_tokenized_csv_path = train_raw_csv_path.replace("raw.csv", f"tokenized_{r_bins}_{theta_bins}_{phi_bins}.csv")
    val_tokenized_csv_path = val_raw_csv_path.replace("raw.csv", f"tokenized_{r_bins}_{theta_bins}_{phi_bins}.csv")

    if not os.path.exists(train_tokenized_csv_path):
        print(f"Skipping experiment - tokenized data not found for {train_tokenized_csv_path}")
        return

    # Create datasets
    train_dataset = ot.OrbitTokenDataset(
        csv_path=train_tokenized_csv_path,
        input_length=input_length,
        output_length=1,
        stride=1
    )
    val_dataset = ot.OrbitTokenDataset(
        csv_path=val_tokenized_csv_path,
        input_length=input_length,
        output_length=1,
        stride=1
    )

    # Initialize model
    model = ot.ThreeTokenDecoderTransferModel(
        r_vocab_size=r_bins,
        theta_vocab_size=theta_bins,
        phi_vocab_size=phi_bins,
        d_model=256,  # Fixed for simplicity; could be parameterized
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=1024,    # Fixed for simplicity; could be parameterized
        dropout=0.1,
        max_seq_len=512
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Define loss model
    loss_model = ot.PositionCrossEntropyLossModel(
        r_weight=1.0,
        theta_weight=1.0,
        phi_weight=1.0
    )

    # Set up log directory with a unique name based on hyperparameters
    run_name = f"lr{learning_rate}_bs{batch_size}_layers{n_layers}_heads{n_heads}_inputlen{input_length}_dataset{dataset_size}_rbins{r_bins}"
    output_dir = os.path.join("orbit_training_runs", trade_name, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize trainer
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

    # Train the model
    print(f"Starting training with hyperparameters: {hyperparams}")
    history = trainer.train(
        epochs=epochs,
        save_every=1,
        log_every=100
    )
    print(f"Training completed for hyperparameters: {hyperparams}")

    # Save summary metrics to the log directory
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump({
            'hyperparams': hyperparams,
            'final_val_loss': history['val_losses'][-1]
        }, f)

if __name__ == "__main__":

    trade_name = "20250302_learning_rate_batch_trade_v2"

    # Define hyperparameter grids directly in Python
    hyperparam_grids = {
        'learning_rate': [1e-5, 1e-4, 1e-3],
        'batch_size': [16, 32, 64],
        'n_layers': [2, 4, 6],
        'n_heads': [4, 8, 16],
        'input_length': [16, 32, 64],
        'dataset_size': [100, 500, 1_000],
        'r_bins': [100, 500, 1_000],
        'theta_bins': [100, 500, 1_000],
        'phi_bins': [100, 500, 1_000],
        'epochs': [10]  # Fixed value, but kept in dict for consistency
    }

    # For simplicity, let’s run a smaller experiment by fixing most parameters
    # and varying only a few (e.g., learning_rate and batch_size)
    fixed_params = {
        'learning_rate': 1e-4,
        'batch_size': 32,
        'n_layers': 4,
        'n_heads': 4,
        'input_length': 32,
        'dataset_size': 500,
        'r_bins': 100,
        'theta_bins': 100,
        'phi_bins': 100,
        'epochs': 10
    }

    # Generate combinations of the varying parameters
    varying_params = {
        'n_layers': hyperparam_grids['n_layers'],
        'n_heads': hyperparam_grids['n_heads'],
    }

    # Iterate over all combinations of the varying parameters
    for nl, nh in product(varying_params['n_layers'], varying_params['n_heads']):
        # Combine fixed and varying hyperparameters
        current_hyperparams = fixed_params.copy()
        current_hyperparams['n_layers'] = nl
        current_hyperparams['n_heads'] = nh

        # Run the experiment
        run_experiment(trade_name, current_hyperparams)

    print("All experiments completed!")




# if __name__ == "__main__":

#     device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
#     print(f"Using device: {device}")

#     total_raw_csv_path = os.path.join(".", "data", "orbits_dataset_1000_raw.csv")
#     train_tokenized_csv_path = total_raw_csv_path.replace("raw.csv", "train_tokenized.csv")
#     val_tokenized_csv_path = total_raw_csv_path.replace("raw.csv", "val_tokenized.csv")
#     test_tokenized_csv_path = total_raw_csv_path.replace("raw.csv", "test_tokenized.csv")

#     train_dataset = ot.OrbitTokenDataset(
#         csv_path=train_tokenized_csv_path,
#         input_length=32,  # Context window of 32 timesteps
#         output_length=1,  # Predict next position
#         stride=1          # Slide window by 1 each time
#     )

#     val_dataset = ot.OrbitTokenDataset(
#         csv_path=val_tokenized_csv_path,
#         input_length=32,
#         output_length=1,
#         stride=1
#     )

#     model = ot.ThreeTokenDecoderTransferModel(
#         r_vocab_size=200,
#         theta_vocab_size=180,
#         phi_vocab_size=360,
#         d_model=256,
#         n_heads=4,
#         n_layers=4,
#         d_ff=1024,
#         dropout=0.1,
#         max_seq_len=512
#     )

#     print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

#     loss_model = ot.PositionCrossEntropyLossModel(
#         r_weight=1.0,
#         theta_weight=1.0,
#         phi_weight=1.0
#     )

#     trainer = ot.OrbitTrainer(
#         transfer_model=model,
#         loss_model=loss_model,
#         train_dataset=train_dataset,
#         val_dataset=val_dataset,
#         lr=1e-4,
#         batch_size=32,
#         num_workers=0,
#         device=device,
#         log_dir='orbit_training_runs'
#     )

#     print("Starting training...")
#     history = trainer.train(
#         epochs=1,
#         save_every=1,
#         log_every=100
#     )

#     print("Training complete!")
