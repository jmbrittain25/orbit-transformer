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
        r = bin_centers["spherical_position"]["r"][tokens[0]]
        theta = bin_centers["spherical_position"]["theta"][tokens[1]]
        phi = bin_centers["spherical_position"]["phi"][tokens[2]]
        vr = bin_centers["spherical_velocity"]["vr"][tokens[3]]
        vtheta = bin_centers["spherical_velocity"]["vtheta"][tokens[4]]
        vphi = bin_centers["spherical_velocity"]["vphi"][tokens[5]]
        pos = spherical_to_cartesian(r, theta, phi)
        vel = spherical_velocity_to_cartesian(vr, vtheta, vphi, theta, phi)
    else:  # cartesian
        x = bin_centers["cartesian_position"]["x"][tokens[0]]
        y = bin_centers["cartesian_position"]["y"][tokens[1]]
        z = bin_centers["cartesian_position"]["z"][tokens[2]]
        vx = bin_centers["cartesian_velocity"]["vx"][tokens[3]]
        vy = bin_centers["cartesian_velocity"]["vy"][tokens[4]]
        vz = bin_centers["cartesian_velocity"]["vz"][tokens[5]]
        pos = np.array([x, y, z])
        vel = np.array([vx, vy, vz])
    return np.concatenate([pos, vel])


def calculate_trajectory_metrics(true_positions, predicted_positions):
    """
    Calculate error metrics between true and predicted trajectories.
    Adapted from visualize_orbits.py.
    
    Returns:
        Dictionary of metrics: mean, max, RMS, final position errors (km).
    """
    true_arr = np.array(true_positions)
    pred_arr = np.array(predicted_positions)
    distances = np.sqrt(np.sum((true_arr - pred_arr) ** 2, axis=1))
    
    return {
        'mean_error_km': np.mean(distances),
        'max_error_km': np.max(distances),
        'rms_error_km': np.sqrt(np.mean(distances ** 2)),
        'final_error_km': distances[-1]
    }


def analyze_run(run_dir, val_raw_df, num_steps=None):
    """Analyze all checkpoints in a run directory, saving detailed prediction data."""

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Load run configuration
    with open(os.path.join(run_dir, 'summary.json'), 'r') as f:
        summary = json.load(f)

    args = summary['args']
    n_bins = args['n_bins']
    coordinate_system = args['coordinate_system']
    input_length = args['input_length']

    # Define token columns
    token_cols = (
        ["eci_r_token", "eci_theta_token", "eci_phi_token", "eci_vr_token", "eci_vtheta_token", "eci_vphi_token"]
        if coordinate_system == 'spherical'
        else ["eci_x_token", "eci_y_token", "eci_z_token", "eci_vx_token", "eci_vy_token", "eci_vz_token"]
    )

    # Initialize tokenizer and dataset
    tokenizer = ot.MultiEciRepresentationTokenizer(
        (n_bins, n_bins, n_bins), (n_bins, n_bins, n_bins),
        (n_bins, n_bins, n_bins), (n_bins, n_bins, n_bins)
    )
    val_df = tokenizer.transform(val_raw_df.copy())
    val_dataset = ot.OrbitTokenDataset(
        val_df, token_cols=token_cols, input_length=input_length, output_length=1, stride=1
    )

    # Get the sub dir the run is in
    sub_dir = [x for x in os.listdir(run_dir) if "." not in x][0]
    sub_dir = os.path.join(run_dir, sub_dir)

    # Process each checkpoint
    checkpoint_files = [f for f in os.listdir(sub_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
    for checkpoint_file in checkpoint_files:
        print(checkpoint_file)

        checkpoint_path = os.path.join(sub_dir, checkpoint_file)
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model
        model = ot.SixTokenDecoderTransferModel(
            pos1_vocab_size=n_bins, pos2_vocab_size=n_bins, pos3_vocab_size=n_bins,
            vel1_vocab_size=n_bins, vel2_vocab_size=n_bins, vel3_vocab_size=n_bins,
            d_model=args['d_model'], n_heads=args['n_heads'], n_layers=args['n_layers'],
            d_ff=args['d_ff'], dropout=0.1, max_seq_len=512
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        data = []
        all_true_positions = []
        all_pred_positions = []
        delta_vs = []

        with torch.no_grad():
            for idx, example in enumerate(val_dataset.examples):
                orbit_id, start_idx, end_idx, _ = example
                group_df = val_dataset.groups.get_group(orbit_id).sort_values('time_s')

                # Calculate maximum possible steps
                max_steps = len(group_df) - end_idx
                effective_steps = max_steps if num_steps is None else min(num_steps, max_steps)

                if effective_steps <= 0:
                    continue  # Skip if no future steps available

                # Retrieve true future states and metadata
                true_data = group_df.iloc[end_idx:end_idx + effective_steps][[
                    'x_eci_km', 'y_eci_km', 'z_eci_km', 'vx_eci_km_s', 'vy_eci_km_s', 'vz_eci_km_s',
                    'r_eci_km', 'theta_eci_deg', 'phi_eci_deg', 'time_s', 'sma_km', 'ecc', 'inc_deg',
                    'raan_deg', 'argp_deg', 'nu_deg'
                ]]
                true_states = true_data[['x_eci_km', 'y_eci_km', 'z_eci_km', 'vx_eci_km_s', 'vy_eci_km_s', 'vz_eci_km_s']].values
                true_positions = true_states[:, :3].tolist()
                all_true_positions.extend(true_positions)

                current_seq = val_dataset[idx]['input'].clone().to(device)  # (input_length, 6)
                pred_positions = []

                initial_raw = group_df.iloc[end_idx - 1][[
                    'x_eci_km', 'y_eci_km', 'z_eci_km', 'vx_eci_km_s', 'vy_eci_km_s', 'vz_eci_km_s',
                    'r_eci_km', 'theta_eci_deg', 'phi_eci_deg', 'sma_km', 'ecc', 'inc_deg',
                    'raan_deg', 'argp_deg', 'nu_deg'
                ]].to_dict()

                # Append initial state data (step=-1) for context
                data.append({
                    'orbit_id': orbit_id,
                    'sequence_start_idx': start_idx,
                    'step': -1,
                    'time_s': group_df.iloc[end_idx - 1]['time_s'],
                    'input_pos1_token': current_seq[-1, 0].item(),
                    'input_pos2_token': current_seq[-1, 1].item(),
                    'input_pos3_token': current_seq[-1, 2].item(),
                    'input_vel1_token': current_seq[-1, 3].item(),
                    'input_vel2_token': current_seq[-1, 4].item(),
                    'input_vel3_token': current_seq[-1, 5].item(),
                    'raw_x_eci_km': initial_raw['x_eci_km'],
                    'raw_y_eci_km': initial_raw['y_eci_km'],
                    'raw_z_eci_km': initial_raw['z_eci_km'],
                    'raw_vx_eci_km_s': initial_raw['vx_eci_km_s'],
                    'raw_vy_eci_km_s': initial_raw['vy_eci_km_s'],
                    'raw_vz_eci_km_s': initial_raw['vz_eci_km_s'],
                    'raw_r_eci_km': initial_raw['r_eci_km'],
                    'raw_theta_eci_deg': initial_raw['theta_eci_deg'],
                    'raw_phi_eci_deg': initial_raw['phi_eci_deg'],
                    'raw_sma_km': initial_raw['sma_km'],
                    'raw_ecc': initial_raw['ecc'],
                    'raw_inc_deg': initial_raw['inc_deg'],
                    'raw_raan_deg': initial_raw['raan_deg'],
                    'raw_argp_deg': initial_raw['argp_deg'],
                    'raw_nu_deg': initial_raw['nu_deg'],
                    'predicted_pos1_token': np.nan,
                    'predicted_pos2_token': np.nan,
                    'predicted_pos3_token': np.nan,
                    'predicted_vel1_token': np.nan,
                    'predicted_vel2_token': np.nan,
                    'predicted_vel3_token': np.nan,
                    'pred_x_eci_km': np.nan,
                    'pred_y_eci_km': np.nan,
                    'pred_z_eci_km': np.nan,
                    'pred_vx_eci_km_s': np.nan,
                    'pred_vy_eci_km_s': np.nan,
                    'pred_vz_eci_km_s': np.nan,
                    'position_error_km': np.nan,
                    'velocity_error_km_s': np.nan,
                    'transfer_delta_v_x': np.nan,
                    'transfer_delta_v_y': np.nan,
                    'transfer_delta_v_z': np.nan,
                    'transfer_delta_v_mag': np.nan,
                    'velocity_match_delta_v_x': np.nan,
                    'velocity_match_delta_v_y': np.nan,
                    'velocity_match_delta_v_z': np.nan,
                    'velocity_match_delta_v_mag': np.nan
                })

                for step in range(effective_steps):
                    # Prepare input tokens
                    pos1_tokens = current_seq[:, 0].unsqueeze(0)
                    pos2_tokens = current_seq[:, 1].unsqueeze(0)
                    pos3_tokens = current_seq[:, 2].unsqueeze(0)
                    vel1_tokens = current_seq[:, 3].unsqueeze(0)
                    vel2_tokens = current_seq[:, 4].unsqueeze(0)
                    vel3_tokens = current_seq[:, 5].unsqueeze(0)

                    # Predict next tokens
                    logits = model(pos1_tokens, pos2_tokens, pos3_tokens, vel1_tokens, vel2_tokens, vel3_tokens)
                    predicted_tokens = torch.tensor([torch.argmax(logits[i][0, -1]).item() for i in range(6)], device=device)

                    # Convert states
                    S_prev = tokens_to_state_vector(current_seq[-1].cpu().numpy(), tokenizer, coordinate_system)
                    S_pred = tokens_to_state_vector(predicted_tokens.cpu().numpy(), tokenizer, coordinate_system)
                    S_true = true_states[step]
                    pred_positions.append(S_pred[:3].tolist())

                    # Compute delta-v
                    r_prev, v_prev = S_prev[:3], S_prev[3:]
                    r_pred, v_pred = S_pred[:3], S_pred[3:]
                    dt = 60 * u.s  # Fixed timestep
                    try:
                        v1, v2 = izzo.lambert(Earth.k, r_prev * u.km, r_pred * u.km, dt)
                        transfer_delta_v = (v1 - v_prev * u.km / u.s).to(u.m / u.s).value
                        velocity_match_delta_v = (v2 - v_pred * u.km / u.s).to(u.m / u.s).value
                    except Exception as e:
                        print(f"Lambert solver failed at step {step} for sequence {idx}: {e}")
                        transfer_delta_v = np.array([np.nan, np.nan, np.nan])
                        velocity_match_delta_v = np.array([np.nan, np.nan, np.nan])

                    transfer_delta_v_mag = np.linalg.norm(transfer_delta_v) if not np.any(np.isnan(transfer_delta_v)) else np.nan
                    velocity_match_delta_v_mag = np.linalg.norm(velocity_match_delta_v) if not np.any(np.isnan(velocity_match_delta_v)) else np.nan
                    total_dv = transfer_delta_v_mag + velocity_match_delta_v_mag if not np.any(np.isnan([transfer_delta_v_mag, velocity_match_delta_v_mag])) else np.nan
                    delta_vs.append(total_dv)

                    # Compute errors
                    position_error = np.linalg.norm(S_true[:3] - S_pred[:3]) if not np.any(np.isnan(S_pred[:3])) else np.nan
                    velocity_error = np.linalg.norm(S_true[3:] - S_pred[3:]) if not np.any(np.isnan(S_pred[3:])) else np.nan

                    # Store data
                    raw_data = true_data.iloc[step].to_dict()
                    data.append({
                        'orbit_id': orbit_id,
                        'sequence_start_idx': start_idx,
                        'step': step,
                        'time_s': raw_data['time_s'],
                        'input_pos1_token': current_seq[-1, 0].item(),
                        'input_pos2_token': current_seq[-1, 1].item(),
                        'input_pos3_token': current_seq[-1, 2].item(),
                        'input_vel1_token': current_seq[-1, 3].item(),
                        'input_vel2_token': current_seq[-1, 4].item(),
                        'input_vel3_token': current_seq[-1, 5].item(),
                        'raw_x_eci_km': raw_data['x_eci_km'],
                        'raw_y_eci_km': raw_data['y_eci_km'],
                        'raw_z_eci_km': raw_data['z_eci_km'],
                        'raw_vx_eci_km_s': raw_data['vx_eci_km_s'],
                        'raw_vy_eci_km_s': raw_data['vy_eci_km_s'],
                        'raw_vz_eci_km_s': raw_data['vz_eci_km_s'],
                        'raw_r_eci_km': raw_data['r_eci_km'],
                        'raw_theta_eci_deg': raw_data['theta_eci_deg'],
                        'raw_phi_eci_deg': raw_data['phi_eci_deg'],
                        'raw_sma_km': raw_data['sma_km'],
                        'raw_ecc': raw_data['ecc'],
                        'raw_inc_deg': raw_data['inc_deg'],
                        'raw_raan_deg': raw_data['raan_deg'],
                        'raw_argp_deg': raw_data['argp_deg'],
                        'raw_nu_deg': raw_data['nu_deg'],
                        'predicted_pos1_token': predicted_tokens[0].item(),
                        'predicted_pos2_token': predicted_tokens[1].item(),
                        'predicted_pos3_token': predicted_tokens[2].item(),
                        'predicted_vel1_token': predicted_tokens[3].item(),
                        'predicted_vel2_token': predicted_tokens[4].item(),
                        'predicted_vel3_token': predicted_tokens[5].item(),
                        'pred_x_eci_km': S_pred[0],
                        'pred_y_eci_km': S_pred[1],
                        'pred_z_eci_km': S_pred[2],
                        'pred_vx_eci_km_s': S_pred[3],
                        'pred_vy_eci_km_s': S_pred[4],
                        'pred_vz_eci_km_s': S_pred[5],
                        'position_error_km': position_error,
                        'velocity_error_km_s': velocity_error,
                        'transfer_delta_v_x': transfer_delta_v[0],
                        'transfer_delta_v_y': transfer_delta_v[1],
                        'transfer_delta_v_z': transfer_delta_v[2],
                        'transfer_delta_v_mag': transfer_delta_v_mag,
                        'velocity_match_delta_v_x': velocity_match_delta_v[0],
                        'velocity_match_delta_v_y': velocity_match_delta_v[1],
                        'velocity_match_delta_v_z': velocity_match_delta_v[2],
                        'velocity_match_delta_v_mag': velocity_match_delta_v_mag
                    })

                    # Update sequence
                    current_seq = torch.cat([current_seq[1:], predicted_tokens.unsqueeze(0)], dim=0)

                all_pred_positions.extend(pred_positions)

            # Calculate trajectory metrics
            if all_true_positions and all_pred_positions:
                metrics = calculate_trajectory_metrics(all_true_positions, all_pred_positions)
                metrics['avg_delta_v_m_s'] = np.nanmean(delta_vs) if delta_vs else np.nan
                with open(os.path.join(sub_dir, f'analysis_metrics_{checkpoint_file}.json'), 'w') as f:
                    json.dump(metrics, f, indent=4)

        # Save to CSV
        df = pd.DataFrame(data)
        csv_path = os.path.join(sub_dir, f'analysis_{checkpoint_file}.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved detailed analysis to {csv_path}")


if __name__ == "__main__":
    val_raw_path = os.path.join(".", "data", "HEO_only_val_dataset_100_orbits.csv")
    val_raw_df = pd.read_csv(val_raw_path)

    trade_dir = os.path.join(".", "orbit_training_runs", "scaling_laws_v1")

    for run_name in os.listdir(trade_dir):
        run_dir = os.path.join(trade_dir, run_name)

        if not os.path.isdir(run_dir):
            continue

        if not os.path.exists(os.path.join(run_dir, 'summary.json')):
            continue

        print(f"Processing {run_dir}")
        analyze_run(run_dir, val_raw_df, num_steps=5)
