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
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
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
    
    for epoch_metric in metrics['epoch_metrics']:
        train_metrics = epoch_metric['train']
        r_losses.append(train_metrics['loss/r'])
        theta_losses.append(train_metrics['loss/theta'])
        phi_losses.append(train_metrics['loss/phi'])
    
    # Plot component losses
    ax2.plot(epochs, r_losses, label='r')
    ax2.plot(epochs, theta_losses, label='theta')
    ax2.plot(epochs, phi_losses, label='phi')
    ax2.set_title('Component Losses')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Plot loss ratios
    total_losses = np.array(metrics['train_losses'])
    ax3.plot(epochs, r_losses / total_losses, label='r/total')
    ax3.plot(epochs, theta_losses / total_losses, label='theta/total')
    ax3.plot(epochs, phi_losses / total_losses, label='phi/total')
    ax3.set_title('Loss Ratios')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Ratio')
    ax3.legend()
    ax3.grid(True)
    
    # Plot validation metrics if available
    if metrics['val_losses']:
        val_r_losses = []
        val_theta_losses = []
        val_phi_losses = []
        
        for epoch_metric in metrics['epoch_metrics']:
            if epoch_metric['val']:
                val_metrics = epoch_metric['val']
                val_r_losses.append(val_metrics['loss/r'])
                val_theta_losses.append(val_metrics['loss/theta'])
                val_phi_losses.append(val_metrics['loss/phi'])
        
        ax4.plot(epochs, val_r_losses, label='r')
        ax4.plot(epochs, val_theta_losses, label='theta')
        ax4.plot(epochs, val_phi_losses, label='phi')
        ax4.set_title('Validation Component Losses')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.legend()
        ax4.grid(True)
    
    plt.tight_layout()

    if save_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(os.path.join(save_dir, f'training_curves_{timestamp}.png'))
    else:
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
    run_dir = os.path.join("orbit_training_runs", "run_20250216_214057", "run_20250217_082010")
    plot_learning_curves(run_dir=run_dir)
