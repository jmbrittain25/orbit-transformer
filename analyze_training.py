import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_run_data(run_dir):
    """Load metrics and summary from a run directory."""

    sub_dir = [x for x in os.listdir(run_dir) if "." not in x][0]
    metrics_path = os.path.join(run_dir, sub_dir, "metrics.json")
    
    summary_path = os.path.join(run_dir, "summary.json")
    
    if not os.path.exists(metrics_path) or not os.path.exists(summary_path):
        return None
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    return {
        "run_dir": run_dir,
        "metrics": metrics,
        "hyperparams": summary['hyperparams'],
        "final_val_loss": summary['final_val_loss']
    }

def plot_loss_curves(runs_data, output_dir, plot_type="both"):
    """Plot training and validation loss curves for all runs."""
    plt.figure(figsize=(12, 8))
    
    for run in runs_data:
        label = f"lr={run['hyperparams']['learning_rate']}, bs={run['hyperparams']['batch_size']}"
        epochs = range(1, len(run['metrics']['train_losses']) + 1)
        
        if plot_type in ["train", "both"]:
            plt.plot(epochs, run['metrics']['train_losses'], label=f"{label} (Train)", linestyle='-')
        if plot_type in ["val", "both"] and run['metrics']['val_losses']:
            plt.plot(epochs, run['metrics']['val_losses'], label=f"{label} (Val)", linestyle='--')
    
    plt.title("Training and Validation Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"loss_curves_{plot_type}.png"))
    plt.close()

def plot_final_val_loss_heatmap(runs_data, output_dir):
    """Create a heatmap of final validation loss across learning rates and batch sizes."""
    # Extract data for heatmap
    learning_rates = sorted(set(run['hyperparams']['learning_rate'] for run in runs_data))
    batch_sizes = sorted(set(run['hyperparams']['batch_size'] for run in runs_data))
    
    heatmap_data = np.zeros((len(learning_rates), len(batch_sizes)))
    
    for i, lr in enumerate(learning_rates):
        for j, bs in enumerate(batch_sizes):
            for run in runs_data:
                if run['hyperparams']['learning_rate'] == lr and run['hyperparams']['batch_size'] == bs:
                    heatmap_data[i, j] = run['final_val_loss']
                    break
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", xticklabels=batch_sizes, yticklabels=learning_rates,
                cmap="YlGnBu", cbar_kws={'label': 'Final Validation Loss'})
    plt.title("Final Validation Loss Heatmap")
    plt.xlabel("Batch Size")
    plt.ylabel("Learning Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_val_loss_heatmap.png"))
    plt.close()

def plot_component_losses(runs_data, output_dir):
    """Plot the component losses (r, theta, phi) for each run."""
    plt.figure(figsize=(12, 8))
    
    for run in runs_data:
        label = f"lr={run['hyperparams']['learning_rate']}, bs={run['hyperparams']['batch_size']}"
        epochs = range(1, len(run['metrics']['epoch_metrics']) + 1)
        
        r_losses = [epoch['train']['loss/r'] for epoch in run['metrics']['epoch_metrics']]
        theta_losses = [epoch['train']['loss/theta'] for epoch in run['metrics']['epoch_metrics']]
        phi_losses = [epoch['train']['loss/phi'] for epoch in run['metrics']['epoch_metrics']]
        
        plt.plot(epochs, r_losses, label=f"{label} (r)", linestyle='-')
        plt.plot(epochs, theta_losses, label=f"{label} (theta)", linestyle='--')
        plt.plot(epochs, phi_losses, label=f"{label} (phi)", linestyle=':')
    
    plt.title("Component Losses (Training)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "component_losses.png"))
    plt.close()

def summarize_best_runs(runs_data):
    """Summarize the best runs based on final validation loss."""
    sorted_runs = sorted(runs_data, key=lambda x: x['final_val_loss'])
    
    summary = []
    for i, run in enumerate(sorted_runs[:3]):  # Top 3 runs
        summary.append({
            "rank": i + 1,
            "learning_rate": run['hyperparams']['learning_rate'],
            "batch_size": run['hyperparams']['batch_size'],
            "final_val_loss": run['final_val_loss'],
            "run_dir": run['run_dir']
        })
    
    with open(os.path.join(output_dir, "best_runs_summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print("\nTop 3 Best Runs (by Final Validation Loss):")
    for entry in summary:
        print(f"Rank {entry['rank']}: lr={entry['learning_rate']}, bs={entry['batch_size']}, "
              f"Final Val Loss={entry['final_val_loss']:.4f}, Dir={entry['run_dir']}")

def main(target_dir, output_dir):
    # Collect all runs
    runs_data = []
    for run_dir in os.listdir(target_dir):

        if "analysis" in run_dir:
            continue

        full_path = os.path.join(target_dir, run_dir)
        if os.path.isdir(full_path):
            run_data = load_run_data(full_path)
            if run_data:
                runs_data.append(run_data)
    
    if not runs_data:
        print("No valid run data found.")
        return
    
    # Generate plots
    plot_loss_curves(runs_data, output_dir, plot_type="both")  # Training and validation loss curves
    plot_final_val_loss_heatmap(runs_data, output_dir)         # Heatmap of final validation loss
    plot_component_losses(runs_data, output_dir)               # Component losses (r, theta, phi)
    
    # Summarize best runs
    summarize_best_runs(runs_data)

if __name__ == "__main__":

    # Directory where the runs are stored
    trade_name = "20250302_learning_rate_batch_trade_v1"
    target_dir = os.path.join("orbit_training_runs", trade_name)

    # Ensure output directory exists for saving plots
    output_dir = os.path.join(target_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    main(target_dir, output_dir)
