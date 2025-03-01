import os
import torch
import glob
import numpy as np
from datetime import datetime

import orbit_transformer as ot

def find_latest_checkpoint(run_dir):
    """Find the most recent checkpoint file in the run directory."""
    checkpoint_files = glob.glob(os.path.join(run_dir, 'checkpoint_epoch_*.pt'))
    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {run_dir}")
    
    # Extract epoch numbers and find the latest
    epochs = [int(f.split('_')[-1].replace('.pt', '')) for f in checkpoint_files]
    latest_idx = epochs.index(max(epochs))
    return checkpoint_files[latest_idx]

def move_optimizer_state_to_device(optimizer_state_dict, device):
    """Move optimizer state to specified device."""
    for param in optimizer_state_dict['state'].values():
        for k, v in param.items():
            if isinstance(v, torch.Tensor):
                param[k] = v.to(device)
    return optimizer_state_dict

def load_and_resume_training(run_dir=None, additional_epochs=10):
    """
    Load the latest checkpoint and resume training.
    
    Parameters
    ----------
    run_dir : str, optional
        Directory containing checkpoints. If None, finds most recent run.
    additional_epochs : int
        Number of additional epochs to train
    """
    # Find most recent run if not specified
    if run_dir is None:
        run_dirs = glob.glob(os.path.join('orbit_training_runs', 'run_*'))
        timestamps = [datetime.strptime(os.path.basename(d)[4:], '%Y%m%d_%H%M%S') 
                     for d in run_dirs]
        run_dir = run_dirs[np.argmax(timestamps)]
        print(f"Using most recent run directory: {run_dir}")

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint_path = find_latest_checkpoint(run_dir)
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint to CPU first
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Recreate model and load state
    config = ot.ThreeTokenGPTConfig(
        r_vocab_size=200,      # Make sure these match your original config
        theta_vocab_size=180,
        phi_vocab_size=360,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        dropout=0.1,
        max_seq_len=512
    )
    model = ot.ThreeTokenGPTDecoder(config)

    # Move model to device first
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Recreate datasets
    train_dataset = ot.OrbitTokenDataset(
        csv_path='data/orbits_dataset_1000_train_tokenized.csv',  # Update paths as needed
        input_length=32,
        output_length=1,
        stride=1
    )

    val_dataset = ot.OrbitTokenDataset(
        csv_path='data/orbits_dataset_1000_val_tokenized.csv',
        input_length=32,
        output_length=1,
        stride=1
    )

    # Configure loss
    loss_config = ot.LossConfig(
        cross_entropy_weight=1.0,
        position_weight=0.5,
        r_weight=1.0,
        theta_weight=1.0,
        phi_weight=1.0
    )
    loss_fn = ot.OrbitLossWrapper(loss_config)

    # Create trainer with loaded state
    trainer = ot.OrbitTrainer(
        transfer_model=model,
        loss_model=loss_fn,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=1e-4,  # You might want to adjust this for continued training
        batch_size=32,
        num_workers=0,
        device=device,
        log_dir=run_dir  # Continue logging to same directory
    )

    # Move optimizer state to correct device before loading
    optimizer_state = move_optimizer_state_to_device(
        checkpoint['optimizer_state_dict'],
        device
    )
    trainer.optimizer.load_state_dict(optimizer_state)
    
    trainer.global_step = checkpoint['global_step']

    # Resume training
    start_epoch = checkpoint['epoch']
    print(f"Resuming training from epoch {start_epoch}")

    history = trainer.train(
        epochs=start_epoch + additional_epochs,
        save_every=1,
        log_every=100
    )

    return history

if __name__ == "__main__":

    run_dir = os.path.join("orbit_training_runs", "run_20250217_131852")

    history = load_and_resume_training(run_dir=run_dir, additional_epochs=10)
