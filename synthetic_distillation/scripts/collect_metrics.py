import os
import shutil
import glob

def collect_metrics():
    # Define the output directory
    output_base = "all_sweep_metrics"
    os.makedirs(output_base, exist_ok=True)
    
    # Define the tasks: (glob_pattern, prefix)
    tasks = [
        ("outputs/Humanoid-v5/mixed/2026-03-04_23-18-31/seed_*", "humanoid"),
        ("outputs/HalfCheetah-v5/mixed/2026-03-04_22-46-58/seed_*", "halfcheetah"),
        ("outputs/Hopper-v5/mixed/2026-03-04_22-16-30/seed_*", "hopper"),
        ("outputs/Ant-v5/mixed/2026-03-05_01-38-01/seed_*", "ant")
    ]
    
    count = 0
    for pattern, prefix in tasks:
        # Find all seed directories
        seed_dirs = glob.glob(pattern)
        for seed_dir in seed_dirs:
            # Extract seed from directory name (e.g., "seed_906003" -> "906003")
            folder_name = os.path.basename(seed_dir)
            if "_" in folder_name:
                seed = folder_name.split("_")[1]
            else:
                seed = folder_name
                
            src_file = os.path.join(seed_dir, "metrics.json")
            if os.path.exists(src_file):
                dst_file = os.path.join(output_base, f"{prefix}_metrics_{seed}.json")
                shutil.copy2(src_file, dst_file)
                print(f"✓ Copied {prefix} (seed {seed})")
                count += 1
            else:
                print(f"✗ Missing metrics.json in {seed_dir}")
                
    print(f"\nSuccessfully consolidated {count} files into '{output_base}/'")

if __name__ == "__main__":
    collect_metrics()
