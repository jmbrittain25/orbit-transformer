import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle


def collect_run_data(trade_dir):

    if not os.path.exists(trade_dir):
        raise FileNotFoundError(f"Directory {trade_dir} does not exist.")
    
    run_data = defaultdict(list)
    
    for run_name in os.listdir(trade_dir):
        print(run_name)

        run_path = os.path.join(trade_dir, run_name)
        if not os.path.isdir(run_path):
            print("not os.path.isdir(run_path)")
            continue
        
        # Load metrics.json
        sub_dir = [x for x in os.listdir(run_path) if "." not in x][0]
        metrics_path = os.path.join(run_path, sub_dir, "metrics.json")
        summary_path = os.path.join(run_path, "summary.json")
        
        if not (os.path.exists(metrics_path) and os.path.exists(summary_path)):
            print("not (os.path.exists(metrics_path) and os.path.exists(summary_path))")
            continue
        
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            val_losses = metrics.get("val_losses", [])
            params = summary.get("args", {})
            
            n_bins = params.get("n_bins")
            coord_sys = params.get("coordinate_system")
            
            if n_bins is None or coord_sys is None or not val_losses:
                print("n_bins is None or coord_sys is None or not val_losses")
                continue
            
            run_data[(n_bins, coord_sys)].append((val_losses, params))
        
        except Exception as e:
            print(f"Error processing run {run_path}: {e}")
            continue
    
    return run_data




if __name__ == "__main__":

    trade_dir = os.path.join(".", "orbit_training_runs", "scaling_laws_v1")
    pickle_path = os.path.join(trade_dir, "trade_loss_data.pickle")

    run_data = collect_run_data(trade_dir)

    with open(pickle_path, 'wb') as f:
        pickle.dump(run_data, f)
