# Configurable Parameters

This document specifies all the parameters you can dynamically vary using Hydra overrides from the terminal. 

You can override any parameter by appending `group.parameter=value` to your command, like `python main.py algo.lr=0.001 model.distil_layers=1`.

## 1. Algorithm (`algo.*`)
These control the PPO Teacher Policy training.

*   `algo.lr`: Learning rate (Default: `3e-4`)
*   `algo.gamma`: Discount factor for future rewards (Default: `0.99`)
*   `algo.gae_lambda`: Bias-variance tradeoff parameter for Generalized Advantage Estimation (Default: `0.95`)
*   `algo.clip_eps`: PPO surrogate clipping threshold epsilon (Default: `0.2`)
*   `algo.entropy_coef`: Entropy loss weight for exploration (Default: `0.01`)
*   `algo.ppo_epochs`: Number of network update epochs per rollout (Default: `10`)
*   `algo.rollout_steps`: Number of steps collected per environment rollout per update (Default: `2048`)
*   `algo.minibatch_size`: Minibatch size used during the PPO update (Default: `256`)
*   `algo.total_timesteps`: The absolute total number of environment interactions to train the completely (Default: `819200`)
*   `algo.num_envs`: How many independent environment threads to spawn in parallel (Default: `1`)

## 2. Distillation (`distill.*`)
These control the Policy Distillation sequence from the Teacher to the Student.

*   **`distill=...`**: The overarching distillation model algorithm to use. Options: `trajectory`, `uniform`, `gaussian`, `uniform_data_bounds`, `mixed`.
*   `distill.epochs`: Total number of gradient descent epochs the student trains against the offline rollout dataset (Default: `100`)
*   `distill.distil_samples`: The absolute integer size of the offline fixed dataset to preemptively generate before iterating students (Default: `400000`). **Can accept lists array structures!** E.g., `distill.distil_samples="[10000, 50000]"` will compute exactly the grid sweep combination of architectures x dataset constraints!
*   `distill.batch_size`: Batch size per step when inferring against the teacher (Default: `2048`)
*   `distill.loss`: Loss function used to evaluate student distance from the teacher. Options: `mean_mse`, `logit_mse`, `cross_entropy`, `sample_mse`.

**Mode-Specific Sub-parameters (`distill.sampling.*`):**
*   `distill.sampling.mode`: Redundant definition of the regime (e.g., `gaussian`).
*   `distill.sampling.std`: *(Gaussian Only)* The standard deviation multiplier for noise injection (Default: `0.1`).
*   `distill.sampling.expansion`: *(Data-Bounds / Mixed Only)* Buffer fraction expansion past the strict Min/Max of the real trajectories (Default: `0.1`).
*   `distill.sampling.mix_ratio`: *(Mixed Only)* The fractional balance representing the percentage of real trajectory points versus synthetic states (Default: `0.5`).

## 3. Network Architectures (`model.*`)
These control the sizes of the Teacher and Student Multi-Layer Perceptrons.

*   `model.layers`: Number of hidden layers in the baseline **Teacher** architecture (Default: `2`)
*   `model.neurons`: Number of neurons per hidden layer in the **Teacher** (Default: `64`)
*   `model.distil_layers`: Exact number of hidden layers allocated for the **Student** (Default: `2`)
*   `model.distil_neurons`: Fractional scaler determining the width of the **Student** layers relative to the Teacher ($0.0 - 1.0$) (Default: `1.0`). EG: `0.25` creates a student with exactly one quarter of the teacher's neurons. **Can accept lists array structures!** E.g., `model.distil_neurons="[1.0, 0.5, 0.25]"` will iteratively construct, train, and track $N$ independently evaluated students over the same parsed dataset constraint!

## 4. Environment (`env.*`)
These define what environment you train and evaluate on.

*   **`env=...`**: Choose the pre-configured base environment file. Options: `cartpole`, `lunarlander`, `hopper`, `halfcheetah`, `ant`, `bipedalwalker`, `humanoid`.
*   `env.name`: The exact Gymnasium environment registry ID (e.g. `HalfCheetah-v4`).
*   `env.type`: The structural action space of the environment (e.g., `discrete`, `continuous`). Controls what agent distribution abstractions are spawned dynamically.

## 5. Experiment Global Tracking
*   `seed`: The overarching deterministic seed for the entire run ensuring perfect Python, PyTorch, Numpy, and Gymnasium reproduceability (Default: `42`).
*   `device`: String explicitly declaring hardware logic (e.g., `"cpu"`, `"cuda:0"`, `"cuda:1"`).
