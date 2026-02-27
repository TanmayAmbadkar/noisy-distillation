import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def aggregate_metric(json_path, regime, key):
    with open(json_path) as f:
        data = json.load(f)

    values = []
    for seed in data:
        if regime in data[seed] and key in data[seed][regime]:
            values.append(data[seed][regime][key])
    
    if not values:
        return 0.0, 0.0
        
    return np.mean(values), np.std(values)

def plot_metric(json_path, key, regimes, out_name):
    means = []
    stds = []

    for r in regimes:
        m, s = aggregate_metric(json_path, r, key)
        means.append(m)
        stds.append(s)

    plt.figure()
    plt.bar(regimes, means, yerr=stds, capsize=5, color=['skyblue', 'lightgreen', 'salmon', 'plum'][:len(regimes)])
    plt.ylabel(key.replace("_", " ").title())
    plt.title(f"{key.replace('_', ' ').title()} Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_name)
    plt.close()

def plot_noise_robustness(json_path, regimes, noise_levels, out_name):
    with open(json_path) as f:
        data = json.load(f)

    plt.figure()
    
    for regime in regimes:
        rewards_mean = []
        rewards_std = []
        
        # Add base 0 noise reward
        m, s = aggregate_metric(json_path, regime, "reward")
        rewards_mean.append(m)
        rewards_std.append(s)
        
        for sigma in noise_levels:
            m, s = aggregate_metric(json_path, regime, f"reward_noise_{sigma}")
            rewards_mean.append(m)
            rewards_std.append(s)
            
        rewards_mean = np.array(rewards_mean)
        rewards_std = np.array(rewards_std)
        levels = [0.0] + noise_levels
        
        plt.plot(levels, rewards_mean, marker='o', label=regime)
        plt.fill_between(levels, rewards_mean - rewards_std, rewards_mean + rewards_std, alpha=0.2)

    plt.xlabel("Observation noise std (\u03c3)")
    plt.ylabel("Reward")
    plt.title("Robustness to Observation Noise")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(out_name)
    plt.close()

def generate_all_plots(results_dir):
    json_path = os.path.join(results_dir, "results.json")
    if not os.path.exists(json_path):
        print(f"No results.json found in {results_dir}")
        return
        
    plot_dir = os.path.join(results_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    regimes_main = ["teacher", "regime_A_BC", "regime_B_UniformMSE", "regime_C_MixedMSE"]
    
    plots = [
        ("reward", "Reward vs Baseline"),
        ("grad_norm_mean", "Gradient Norm"),
        ("logit_mean", "Logit Magnitude"),
        ("lipschitz_mean", "Local Lipschitz Estimate")
    ]
    
    print("Generating main comparison plots...")
    for key, _ in plots:
        plot_metric(json_path, key, regimes_main, os.path.join(plot_dir, f"{key}_comparison.png"))
        
    print("Generating robustness plot...")
    plot_noise_robustness(json_path, regimes_main, [0.01, 0.05, 0.1], os.path.join(plot_dir, "robustness_curve.png"))
    
    print("Generating coverage/capacity plots...")
    regimes_coverage = ["regime_B_TightMSE", "regime_B_UniformMSE", "regime_B_WideMSE"]
    plot_metric(json_path, "reward", regimes_coverage, os.path.join(plot_dir, "coverage_reward.png"))
    
    regimes_capacity = ["regime_B_Cap128", "regime_B_UniformMSE", "regime_B_Cap32"]
    plot_metric(json_path, "reward", regimes_capacity, os.path.join(plot_dir, "capacity_reward.png"))
    
    print(f"All plots saved to {plot_dir}")

if __name__ == "__main__":
    # Find newest results dir
    base_dir = "results"
    if os.path.exists(base_dir):
        dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if dirs:
            newest_dir = max(dirs, key=os.path.getmtime)
            generate_all_plots(newest_dir)
        else:
            print("No valid results directories found.")
