# Proximal Policy Optimization (PPO) Framework

This folder contains a clean, optimized, and thoroughly documented implementation of Proximal Policy Optimization (PPO). It is designed with good software engineering practices to be modular, extensible, and easy to understand.

## Structure

- **`agent.py`**: Contains the neural network architectures.
  - `BaseAgent`: An abstract base class defining the required interface for an agent (value estimation, action distribution, etc.).
  - `DiscreteAgent`: An implementation for environments with discrete action spaces (using `Categorical` distributions).
  - `ContinuousAgent`: An implementation for environments with continuous action spaces (using parameterized `Normal` distributions).
- **`ppo.py`**: The core algorithmic logic.
  - `PPO`: The main agent that handles rollout collection, Advantage computation (GAE), and policy/value function objective optimizations.
  - `LinearLRSchedule`: A learning rate scheduler that decays linearly.
  - `PPOLogger`: Handles reporting metrics to the console and to TensorBoard.
- **`main_ppo.py`**: The entrypoint script.
  - Handles parsing arguments, setting up synchronized vectorized environments (`Gymnasium`), constructing the correct agent based on the action space, instantiating the PPO algorithm, and running the training loop.
  - Includes a helper function to evaluate a saved policy.

## Key Features

- **Continuous and Discrete Environments Supported**: Automatically swaps out the network and distribution logic based on the action space type.
- **Generalized Advantage Estimation (GAE)**: Uses GAE for stable variance reduction in policy gradient estimates.
- **PPO Clipped Objective**: Prevents destructively large policy updates.
- **Value Function Clipping**: Optional clipping of the value function loss to stabilize critic updates.
- **Vectorized Environments**: Trains faster by interacting with multiple environments in parallel using `gym.vector.SyncVectorEnv`.
- **Early Stopping via KL Divergence**: Optionally halts a policy update epoch early if the approximate KL divergence exceeds a target threshold, ensuring the policy stays within the trust region.
- **TensorBoard Logging**: Comprehensive logging of rewards, episode lengths, policy loss, value loss, entropy loss, and clipping fractions.

## Usage

To train an agent on a default environment (e.g. `CartPole-v1`), simply execute the main script:

```bash
python main_ppo.py
```

The script supports overriding several configuration parameters when called as a module, or by adjusting the default kwargs inside `main_ppo.py`. 

### Evaluating a Model

If `save_model=True` is passed to the `run_ppo` function, the model automatically saves to the `runs/` directory and is evaluated.

### Tensorboard

If `use_tensorboard=True` is provided, metrics will be dumped into a `runs/` directory. You can visualize them using:

```bash
tensorboard --logdir runs/
```

## Extensibility

To implement a new network architecture, subclass `BaseAgent` from `agent.py` and implement the 4 required abstract methods: `estimate_value_from_observation`, `get_action_distribution`, `sample_action_and_compute_log_prob`, and `compute_action_log_probabilities_and_entropy`.
