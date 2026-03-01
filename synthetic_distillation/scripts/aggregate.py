import os
import sys
from src.logging.aggregator import aggregate_multirun

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/aggregate.py outputs/multirun/YYYY-MM-DD/HH-MM-SS/")
        sys.exit(1)
        
    run_dir = sys.argv[1]
    
    if not os.path.exists(run_dir):
        print(f"Error: {run_dir} does not exist.")
        sys.exit(1)
        
    aggregate_multirun(run_dir)
