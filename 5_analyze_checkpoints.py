import os
import json
import torch
import numpy as np
import pandas as pd
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.iod import izzo

import orbit_transformer as ot
from orbit_transformer.orbit_utils import spherical_to_cartesian, spherical_velocity_to_cartesian


def tokens_to_state_vector(tokens, tokenizer, coordinate_system):
    """Convert six tokens to a Cartesian state vector (x, y, z, vx, vy, vz) in km and km/s."""
    bin_centers = tokenizer.get_bin_centers()
    
    if coordinate_system == 'spherical':
        # Extract spherical position and velocity bin centers
        r = bin_centers["spherical_position"]["r"][tokens[0]]
        theta = bin_centers["spherical_position"]["theta"][tokens[1]]
        phi = bin_centers["spherical_position"]["phi"][tokens[2]]
        vr = bin_centers["spherical_velocity"]["vr"][tokens[3]]
        vtheta = bin_centers["spherical_velocity"]["vtheta"][tokens[4]]
        vphi = bin_centers["spherical_velocity"]["vphi"][tokens[5]]
        
        # Convert to Cartesian
        pos = spherical_to_cartesian(r, theta, phi)
        vel = spherical_velocity_to_cartesian(vr, vtheta, vphi, theta, phi)
    else:  # cartesian
        # Extract Cartesian position and velocity bin centers
        x = bin_centers["cartesian_position"]["x"][tokens[0]]
        y = bin_centers["cartesian_position"]["y"][tokens[1]]
        z = bin_centers["cartesian_position"]["z"][tokens[2]]
        vx = bin_centers["cartesian_velocity"]["vx"][tokens[3]]
        vy = bin_centers["cartesian_velocity"]["vy"][tokens[4]]
        vz = bin_centers["cartesian_velocity"]["vz"][tokens[5]]
        
        pos = np.array([x, y, z])
        vel = np.array([vx, vy, vz])
        
    return np.concatenate([pos, vel])


def analyze_run(run_dir, val_raw_df, num_sequences=10, num_steps=20):
    """Analyze all checkpoints in a run directory."""

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    with open(os.path.join(run_dir, 'summary.json'), 'r') as f:
        summary = json.load(f)

    args = summary['args']
    n_bins = args['n_bins']
    coordinate_system = args['coordinate_system']

    if coordinate_system == 'spherical':
        token_cols = ["eci_r_token", "eci_theta_token", "eci_phi_token",
                      "eci_vr_token", "eci_vtheta_token", "eci_vphi_token"]
    else:
        token_cols = ["eci_x_token", "eci_y_token", "eci_z_token",
                      "eci_vx_token", "eci_vy_token", "eci_vz_token"]

    tokenizer = ot.MultiEciRepresentationTokenizer(
        (n_bins, n_bins, n_bins), (n_bins, n_bins, n_bins),
        (n_bins, n_bins, n_bins), (n_bins, n_bins, n_bins)
    )
    val_df = tokenizer.transform(val_raw_df.copy())
    val_dataset = ot.OrbitTokenDataset(
        val_df, token_cols=token_cols,
        input_length=args['input_length'], output_length=1, stride=1
    )

    # Select a subset of sequences
    indices = np.random.choice(len(val_dataset), num_sequences, replace=False)
    sequences = [val_dataset[i]['input'] for i in indices]  # Shape: (input_length, 6)

    # Process each checkpoint
    checkpoint_files = [f for f in os.listdir(run_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(run_dir, checkpoint_file)
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create and load model
        model = ot.SixTokenDecoderTransferModel(
            pos1_vocab_size=n_bins, pos2_vocab_size=n_bins, pos3_vocab_size=n_bins,
            vel1_vocab_size=n_bins, vel2_vocab_size=n_bins, vel3_vocab_size=n_bins,
            d_model=args['d_model'], n_heads=args['n_heads'], n_layers=args['n_layers'],
            d_ff=args['d_ff'], dropout=0.1, max_seq_len=512
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        delta_vs = []
        with torch.no_grad():
            for seq in sequences:
                current_seq = seq.clone().to(device)  # (input_length, 6)
                for _ in range(num_steps):

                    pos1_tokens = current_seq[:, 0].unsqueeze(0)
                    pos2_tokens = current_seq[:, 1].unsqueeze(0)
                    pos3_tokens = current_seq[:, 2].unsqueeze(0)
                    vel1_tokens = current_seq[:, 3].unsqueeze(0)
                    vel2_tokens = current_seq[:, 4].unsqueeze(0)
                    vel3_tokens = current_seq[:, 5].unsqueeze(0)

                    logits = model(
                        pos1_tokens, pos2_tokens, pos3_tokens,
                        vel1_tokens, vel2_tokens, vel3_tokens
                    )
                    predicted_tokens = torch.tensor([
                        torch.argmax(logits[i][0, -1]).item() for i in range(6)
                    ], device=device)

                    # Convert states
                    S_prev = tokens_to_state_vector(current_seq[-1].cpu().numpy(), tokenizer, coordinate_system)
                    S_pred = tokens_to_state_vector(predicted_tokens.cpu().numpy(), tokenizer, coordinate_system)
                    r_prev, v_prev = S_prev[:3], S_prev[3:]
                    r_pred, v_pred = S_pred[:3], S_pred[3:]

                    # TODO - calculate from data, not harcode!
                    dt = 60 * u.s  # Time step from data generation

                    # Compute delta-v using Lambert's problem
                    v1, v2 = izzo.lambert(Earth.k, r_prev * u.km, r_pred * u.km, dt)
                    transfer_delta_v = (v1 - v_prev * u.km / u.s).to(u.m / u.s).value
                    velocity_match_delta_v = (v2 - v_pred * u.km / u.s).to(u.m / u.s).value

                    transfer_delta_v_mag = np.linalg.norm(transfer_delta_v)
                    velocity_match_delta_v_mag = np.linalg.norm(velocity_match_delta_v)
                    total_dv = transfer_delta_v_mag + velocity_match_delta_v_mag

                    delta_vs.append(total_dv)

                    # Update sequence
                    current_seq = torch.cat([current_seq[1:], predicted_tokens.unsqueeze(0)], dim=0)

        # Save results
        avg_delta_v = np.mean(delta_vs)
        result = {'avg_delta_v': avg_delta_v, 'delta_vs': delta_vs}
        with open(os.path.join(run_dir, f'analysis_{checkpoint_file}.json'), 'w') as f:
            json.dump(result, f)


def main():
    base_dir = 'orbit_training_runs'
    val_raw_path = 'data/orbits_HEO_only_dataset_10000_val_raw.csv'  # Choose a fixed dataset_size
    val_raw_df = pd.read_csv(val_raw_path)

    for trade_name in os.listdir(base_dir):
        trade_dir = os.path.join(base_dir, trade_name)
        if not os.path.isdir(trade_dir):
            continue
        for run_name in os.listdir(trade_dir):
            run_dir = os.path.join(trade_dir, run_name)
            if not os.path.isdir(run_dir) or not os.path.exists(os.path.join(run_dir, 'summary.json')):
                continue
            print(f"Processing {run_dir}")
            analyze_run(run_dir, val_raw_df)


if __name__ == "__main__":
    main()
