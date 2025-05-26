import os
import json
import time
import pandas as pd


def collect_summary_data(trade_dir):
    """
    Collect summary information from training runs to write to a CSV file.
    
    Args:
        trade_dir (str): Directory containing training run subdirectories
        
    Returns:
        list: List of dictionaries with 'run_name', 'min_val_loss', and run parameters
    """
    if not os.path.exists(trade_dir):
        raise FileNotFoundError(f"Directory {trade_dir} does not exist.")
    
    summary_list = []
    
    for run_name in os.listdir(trade_dir):
        run_path = os.path.join(trade_dir, run_name)
        if not os.path.isdir(run_path):
            continue
        
        sub_dirs = [x for x in os.listdir(run_path) if "." not in x]
        if not sub_dirs:
            continue
        sub_dir = sub_dirs[0]

        print(sub_dir)
        
        metrics_path = os.path.join(run_path, sub_dir, "metrics.json")
        summary_path = os.path.join(run_path, sub_dir, "summary.json")
        
        if not (os.path.exists(metrics_path) and os.path.exists(summary_path)):
            continue
        
        print(metrics_path)
        print(summary_path)

        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            val_losses = metrics.get("val_losses", [])
            params = summary.get("args", {})
            
            if not val_losses:
                print(f"No val_losses for {run_name}")
                continue
            
            min_val_loss = min(val_losses)
            
            summary_dict = {
                'run_name': run_name,
                'min_val_loss': min_val_loss,
                **params
            }
            summary_list.append(summary_dict)
                
        except Exception as e:
            print(f"Error processing run {run_path}: {e}")
            continue
    
    return summary_list

if __name__ == "__main__":
    trade_dir = os.path.join(".", "orbit_training_runs", "scaling_laws_v2")
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(trade_dir, f"trade_summary_{timestamp}.csv")
    
    summary_list = collect_summary_data(trade_dir)
    
    if summary_list:
        param_keys = [key for key in summary_list[0].keys() 
                     if key not in ['run_name', 'min_val_loss']]
        columns = ['run_name', 'min_val_loss'] + sorted(param_keys)
        
        df = pd.DataFrame(summary_list, columns=columns)
        df.to_csv(csv_path, index=False)
        print(f"Saved summary to {csv_path}")
    else:
        print("No summary data to save.")