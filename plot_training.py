import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def find_latest_run(base_dir='orbit_training_runs'):
    """Find the most recent training run directory."""
    run_dirs = glob.glob(os.path.join(base_dir, 'run_*'))
    if not run_dirs:
        raise ValueError(f"No run directories found in {base_dir}")
    
    # Parse timestamps from directory names and find most recent
    timestamps = [datetime.strptime(os.path.basename(d)[4:], '%Y%m%d_%H%M%S') for d in run_dirs]
    latest_idx = np.argmax(timestamps)
    return run_dirs[latest_idx]

def load_metrics(run_dir):
    """Load metrics from a training run directory."""
    metrics_path = os.path.join(run_dir, 'metrics.json')
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"No metrics.json found in {run_dir}")
    
    with open(metrics_path, 'r') as f:
        return json.load(f)

def plot_training_curves(metrics, save_dir=None):
    """Plot various training curves from the metrics."""
    plt.close()

    # Create figure with subplots - now 2x3 to include position metrics
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    
    fig.suptitle('Training Metrics', fontsize=16)
    
    # Get epochs from metrics
    epochs = range(1, len(metrics['train_losses']) + 1)
    
    # Plot overall losses
    ax1.plot(epochs, metrics['train_losses'], label='Train')
    if metrics['val_losses']:
        ax1.plot(epochs, metrics['val_losses'], label='Validation')
    ax1.set_title('Overall Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Extract component losses
    r_losses = []
    theta_losses = []
    phi_losses = []
    position_raw = []
    position_norm = []
    
    for epoch_metric in metrics['epoch_metrics']:
        train_metrics = epoch_metric['train']
        r_losses.append(train_metrics['loss/r'])
        theta_losses.append(train_metrics['loss/theta'])
        phi_losses.append(train_metrics['loss/phi'])
        position_raw.append(train_metrics['loss/position/km'])
        position_norm.append(train_metrics['loss/position/normalized'])
    
    # Plot token component losses
    ax2.plot(epochs, r_losses, label='r')
    ax2.plot(epochs, theta_losses, label='theta')
    ax2.plot(epochs, phi_losses, label='phi')
    ax2.set_title('Token Component Losses')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Plot raw position error
    ax3.plot(epochs, position_raw, label='Position Error')
    ax3.set_title('Raw Position Error')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Distance (km)')
    ax3.grid(True)
    
    # Plot normalized position error
    ax4.plot(epochs, position_norm, label='Normalized Error')
    ax4.set_title('Normalized Position Error')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Error (fraction of radius)')
    ax4.grid(True)
    
    # Plot loss ratios
    total_losses = np.array(metrics['train_losses'])
    ce_total = np.array([m['train'].get('loss/ce_total', 0.0) for m in metrics['epoch_metrics']])
    pos_contribution = np.array(position_norm)  # Assuming position_weight is applied in loss function
    
    ax5.plot(epochs, ce_total / total_losses, label='CE Loss Contribution')
    ax5.plot(epochs, pos_contribution / total_losses, label='Position Loss Contribution')
    ax5.set_title('Loss Component Contributions')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Fraction of Total Loss')
    ax5.legend()
    ax5.grid(True)
    
    # Plot validation metrics if available
    if metrics['val_losses']:
        val_position_raw = []
        val_position_norm = []
        
        for epoch_metric in metrics['epoch_metrics']:
            if epoch_metric['val']:
                val_metrics = epoch_metric['val']
                val_position_raw.append(val_metrics['loss/position/km'])
                val_position_norm.append(val_metrics['loss/position/normalized'])
        
        ax6.plot(epochs, position_raw, label='Train')
        ax6.plot(epochs, val_position_raw, label='Validation')
        ax6.set_title('Position Error Comparison')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Distance (km)')
        ax6.legend()
        ax6.grid(True)
    
    plt.tight_layout()
    
    if save_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(os.path.join(save_dir, f'training_curves_{timestamp}.png'))
    
    plt.show()
    
def plot_learning_curves(run_dir=None):
    """Main function to load and plot training metrics."""
    if run_dir is None:
        run_dir = find_latest_run()
    
    print(f"Loading metrics from: {run_dir}")
    metrics = load_metrics(run_dir)
    
    plot_training_curves(metrics, save_dir=run_dir)
    print(f"Plots saved to: {run_dir}")

if __name__ == "__main__":
    # run_dir = os.path.join("orbit_training_runs", "run_20250216_214057", "run_20250217_082010")

    run_dir = os.path.join("orbit_training_runs", "run_20250217_131852", "run_20250217_191849")

    plot_learning_curves(run_dir=run_dir)
