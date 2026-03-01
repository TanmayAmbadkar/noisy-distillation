import os
import json
import glob
import numpy as np

def aggregate_multirun(run_dir):
    """
    Finds all metrics.json files inside run_dir (recursive),
    aggregates mean and std for each key, and saves to summary.json.
    """
    metrics_files = glob.glob(os.path.join(run_dir, "**", "metrics.json"), recursive=True)
    
    if not metrics_files:
        print(f"No metrics.json files found in {run_dir}. Did runs complete successfully?")
        return None
        
    print(f"Found {len(metrics_files)} runs to aggregate in {run_dir}")
    
    all_metrics = {}
    
    for f in metrics_files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                for k, v in data.items():
                    if isinstance(v, (int, float)):
                        if k not in all_metrics:
                            all_metrics[k] = []
                        all_metrics[k].append(v)
        except Exception as e:
            print(f"Could not read {f}: {e}")
            
    summary = {}
    csv_rows = ["metric,mean,std,count\n"]
    
    for k, vals in all_metrics.items():
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        count = len(vals)
        
        summary[k] = {
            "mean": float(mean_val),
            "std": float(std_val),
            "count": count
        }
        
        csv_rows.append(f"{k},{mean_val:.4f},{std_val:.4f},{count}\n")
        
    summary_json_path = os.path.join(run_dir, "summary.json")
    summary_csv_path = os.path.join(run_dir, "summary.csv")
    
    with open(summary_json_path, 'w') as fp:
        json.dump(summary, fp, indent=4)
        
    with open(summary_csv_path, 'w') as fp:
        fp.writelines(csv_rows)
        
    print(f"Aggregated summary saved to {summary_json_path}")
    print(f"Aggregated CSV saved to {summary_csv_path}")
    
    return summary
