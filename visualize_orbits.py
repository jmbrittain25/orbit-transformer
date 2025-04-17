import os
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from typing import Tuple, List

import orbit_transformer as ot

def load_latest_model(run_dir=None):
    """Load the latest model checkpoint."""
    if run_dir is None:
        run_dirs = glob.glob(os.path.join('orbit_training_runs', 'run_*'))
        timestamps = [datetime.strptime(os.path.basename(d)[4:], '%Y%m%d_%H%M%S') 
                     for d in run_dirs]
        run_dir = run_dirs[np.argmax(timestamps)]
    
    # Find latest checkpoint
    checkpoint_files = glob.glob(os.path.join(run_dir, 'checkpoint_epoch_*.pt'))
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    
    # Load checkpoint
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    
    # Recreate model
    config = ot.ThreeTokenGPTConfig(
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
    model = ot.ThreeTokenGPTDecoder(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, device

def get_tokenizer():
    """Create tokenizer with same configuration as training."""
    return ot.SphericalCoordinateTokenizer(
        r_bins=200,
        theta_bins=180,
        phi_bins=360,
        theta_min=0.0,
        theta_max=180.0,
        phi_min=-180.0,
        phi_max=180.0,
        composite_tokens=False
    )

def tokens_to_coordinates(r_token: int, theta_token: int, phi_token: int, tokenizer) -> Tuple[float, float, float]:
    """Convert token indices back to spherical coordinates."""
    r_centers, theta_centers, phi_centers = tokenizer.get_bin_centers()
    
    r = r_centers[r_token]
    theta = theta_centers[theta_token]
    phi = phi_centers[phi_token]
    
    return r, theta, phi


def predict_trajectory(model, initial_sequence: torch.Tensor, num_steps: int, device: str) -> List[Tuple[int, int, int]]:
    """
    Predict orbit trajectory given initial sequence.
    Returns list of (r_token, theta_token, phi_token) tuples.
    """
    model.eval()
    
    # Initialize with given sequence
    r_seq = initial_sequence[:, 0].unsqueeze(0)  # Add batch dimension
    theta_seq = initial_sequence[:, 1].unsqueeze(0)
    phi_seq = initial_sequence[:, 2].unsqueeze(0)
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_steps):
            # Get model predictions
            r_logits, theta_logits, phi_logits = model(
                r_seq.to(device), 
                theta_seq.to(device), 
                phi_seq.to(device)
            )
            
            # Get token predictions (last timestep)
            r_pred = torch.argmax(r_logits[0, -1]).item()
            theta_pred = torch.argmax(theta_logits[0, -1]).item()
            phi_pred = torch.argmax(phi_logits[0, -1]).item()
            
            predictions.append((r_pred, theta_pred, phi_pred))
            
            # Update sequences
            r_seq = torch.cat([r_seq[:, 1:], torch.tensor([[r_pred]])], dim=1)
            theta_seq = torch.cat([theta_seq[:, 1:], torch.tensor([[theta_pred]])], dim=1)
            phi_seq = torch.cat([phi_seq[:, 1:], torch.tensor([[phi_pred]])], dim=1)
    
    return predictions

def calculate_trajectory_metrics(true_positions, predicted_positions):
    """
    Calculate various error metrics between true and predicted trajectories.
    
    Returns:
    - Dictionary of metrics including:
        - Mean distance error (km)
        - Max distance error (km)
        - RMS error (km)
        - Final position error (km)
    """
    # Convert position tuples to numpy arrays for easier calculation
    true_arr = np.array(true_positions)
    pred_arr = np.array(predicted_positions)
    
    # Calculate euclidean distances at each point
    distances = np.sqrt(np.sum((true_arr - pred_arr) ** 2, axis=1))
    
    # Calculate metrics
    mean_error = np.mean(distances)
    max_error = np.max(distances)
    rms_error = np.sqrt(np.mean(distances ** 2))
    final_error = distances[-1]
    
    # Calculate cumulative drift
    cumulative_error = np.cumsum(distances)
    total_drift = cumulative_error[-1]
    
    # Calculate error growth rate (using linear regression)
    timesteps = np.arange(len(distances))
    slope, _ = np.polyfit(timesteps, distances, 1)
    
    return {
        'mean_error_km': mean_error,
        'max_error_km': max_error,
        'rms_error_km': rms_error,
        'final_error_km': final_error,
        'total_drift_km': total_drift,
        'error_growth_rate_km_per_step': slope
    }

def plot_orbit_comparison(initial_positions, true_positions, predicted_positions, orbit_id=None):
    """
    Plot orbit trajectories in 3D with error metrics.
    
    Parameters:
        initial_positions: List of (x,y,z) tuples for input sequence
        true_positions: List of (x,y,z) tuples for true trajectory
        predicted_positions: List of (x,y,z) tuples for predicted trajectory
        orbit_id: Optional identifier for the orbit
    """
    # Create main figure with two subplots
    fig = plt.figure(figsize=(15, 8))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot Earth
    R_EARTH = 6371  # km
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = R_EARTH * np.outer(np.cos(u), np.sin(v))
    y = R_EARTH * np.outer(np.sin(u), np.sin(v))
    z = R_EARTH * np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x, y, z, color='lightblue', alpha=0.3)
    
    # Plot initial sequence with points
    init_x, init_y, init_z = zip(*initial_positions)
    ax1.plot(init_x, init_y, init_z, 'k-', label='Initial Sequence', linewidth=2, alpha=0.5)
    ax1.scatter(init_x, init_y, init_z, c='black', s=30, alpha=0.8)
    
    # Plot trajectories with points
    true_x, true_y, true_z = zip(*true_positions)
    pred_x, pred_y, pred_z = zip(*predicted_positions)
    
    ax1.plot(true_x, true_y, true_z, 'b-', label='True Orbit', linewidth=1, alpha=0.5)
    ax1.scatter(true_x, true_y, true_z, c='blue', s=30, alpha=0.8)
    
    ax1.plot(pred_x, pred_y, pred_z, 'r--', label='Predicted Orbit', linewidth=1, alpha=0.5)
    ax1.scatter(pred_x, pred_y, pred_z, c='red', s=30, alpha=0.8)
    
    # Start and end points (keeping these larger for emphasis)
    ax1.scatter(init_x[0], init_y[0], init_z[0], c='black', marker='o', s=100, label='Sequence Start')
    ax1.scatter(init_x[-1], init_y[-1], init_z[-1], c='black', marker='s', s=100, label='Sequence End')
    
    # Set labels and title
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    ax1.set_zlabel('Z (km)')
    title = f'Orbit Comparison - ID: {orbit_id}' if orbit_id else 'Orbit Comparison'
    ax1.set_title(title)
    
    # Equal aspect ratio
    max_range = np.array([
        max(true_x) - min(true_x),
        max(true_y) - min(true_y),
        max(true_z) - min(true_z)
    ]).max() / 2.0
    
    mid_x = (max(true_x) + min(true_x)) * 0.5
    mid_y = (max(true_y) + min(true_y)) * 0.5
    mid_z = (max(true_z) + min(true_z)) * 0.5
    
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax1.legend()
    
    # Error metrics plot
    ax2 = fig.add_subplot(122)
    
    # Calculate distance error at each timestep
    true_arr = np.array(true_positions)
    pred_arr = np.array(predicted_positions)
    distances = np.sqrt(np.sum((true_arr - pred_arr) ** 2, axis=1))
    
    # Plot error over time
    timesteps = np.arange(len(distances))
    ax2.plot(timesteps, distances, 'b-', label='Position Error')
    ax2.fill_between(timesteps, distances, alpha=0.2)
    
    # Calculate and display metrics
    metrics = calculate_trajectory_metrics(true_positions, predicted_positions)
    
    metrics_text = (
        f"Error Metrics:\n"
        f"Mean Error: {metrics['mean_error_km']:.2f} km\n"
        f"Max Error: {metrics['max_error_km']:.2f} km\n"
        f"RMS Error: {metrics['rms_error_km']:.2f} km\n"
        f"Final Error: {metrics['final_error_km']:.2f} km\n"
        f"Error Growth: {metrics['error_growth_rate_km_per_step']:.2f} km/step"
    )
    
    # Add metrics text box
    ax2.text(0.95, 0.95, metrics_text,
             transform=ax2.transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('Prediction Step')
    ax2.set_ylabel('Position Error (km)')
    ax2.set_title('Prediction Error Over Time')
    ax2.grid(True)
    
    plt.tight_layout()
    return fig, metrics

def visualize_predictions(test_dataset, run_dir=None, num_orbits=3, prediction_steps=60):
    """Visualize predictions for multiple test orbits."""
    model, device = load_latest_model(run_dir=run_dir)
    tokenizer = get_tokenizer()
    
    # Store metrics for all orbits
    all_metrics = []
    
    # Load the original test CSV to get full orbit data
    test_csv_path = test_dataset.csv_path
    df = pd.read_csv(test_csv_path)
    
    # Get unique orbit IDs
    unique_orbit_ids = df['orbit_id'].unique()[:num_orbits]
    
    for orbit_id in unique_orbit_ids:
        print(f"\nProcessing orbit: {orbit_id}")
        
        # Get the data for this orbit
        orbit_df = df[df['orbit_id'] == orbit_id].sort_values('time_s')
        
        # Get sequences and make predictions
        initial_tokens = torch.tensor([
            orbit_df['eci_r_token'].values[:32],
            orbit_df['eci_theta_token'].values[:32],
            orbit_df['eci_phi_token'].values[:32]
        ]).T
        
        predicted_tokens = predict_trajectory(model, initial_tokens, prediction_steps, device)
        
        # Convert all positions
        initial_positions = []
        for r_token, theta_token, phi_token in zip(
            initial_tokens[:, 0], initial_tokens[:, 1], initial_tokens[:, 2]
        ):
            r, theta, phi = tokens_to_coordinates(
                r_token.item(), theta_token.item(), phi_token.item(), tokenizer
            )
            x, y, z = spherical_to_cartesian(r, theta, phi)
            initial_positions.append((x, y, z))
        
        true_tokens = list(zip(
            orbit_df['eci_r_token'].values[32:32+prediction_steps],
            orbit_df['eci_theta_token'].values[32:32+prediction_steps],
            orbit_df['eci_phi_token'].values[32:32+prediction_steps]
        ))
        
        true_positions = []
        for r_token, theta_token, phi_token in true_tokens:
            r, theta, phi = tokens_to_coordinates(r_token, theta_token, phi_token, tokenizer)
            x, y, z = spherical_to_cartesian(r, theta, phi)
            true_positions.append((x, y, z))
        
        predicted_positions = []
        for r_token, theta_token, phi_token in predicted_tokens:
            r, theta, phi = tokens_to_coordinates(r_token, theta_token, phi_token, tokenizer)
            x, y, z = spherical_to_cartesian(r, theta, phi)
            predicted_positions.append((x, y, z))
        
        # Plot comparison and get metrics
        fig, metrics = plot_orbit_comparison(initial_positions, true_positions, predicted_positions, orbit_id)
        metrics['orbit_id'] = orbit_id
        all_metrics.append(metrics)
        
        plt.show()
        plt.close()
    
    # Print summary statistics
    print("\nSummary Statistics Across All Orbits:")
    metric_names = ['mean_error_km', 'max_error_km', 'rms_error_km', 'final_error_km']
    for metric in metric_names:
        values = [m[metric] for m in all_metrics]
        print(f"\n{metric}:")
        print(f"  Mean: {np.mean(values):.2f} km")
        print(f"  Std:  {np.std(values):.2f} km")
        print(f"  Min:  {np.min(values):.2f} km")
        print(f"  Max:  {np.max(values):.2f} km")


if __name__ == "__main__":
    # Load test dataset
    test_dataset = ot.OrbitTokenDataset(
        csv_path='data/orbits_dataset_1000_test_tokenized.csv',
        input_length=32,
        output_length=1,
        stride=1
    )
    
    # run_dir = os.path.join("orbit_training_runs", "run_20250216_214057", "run_20250217_082010")

    run_dir = os.path.join("orbit_training_runs", "run_20250217_131852", "run_20250217_191849")


    visualize_predictions(test_dataset, run_dir=run_dir, num_orbits=10)
