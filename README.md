# Synthetic Policy Distillation Framework

A production-grade, reproducible research framework for synthetic state coverage in Policy Distillation, primarily focused on discrete vs continuous environment sampling distributions.

## Architecture & Features

This framework is built upon **Hydra** composition for fully deterministic multi-seed experiment tracking, paired with **TensorBoard** and structured **JSON** logging.

- **Unified Environments**: Transparent handling of discrete (`CartPole`) and continuous (`Hopper` / `HalfCheetah` / `Ant`) Gymnasium environments via vector-wrapper adapters.
- **Optimized PPO Teacher**: Integrated with a heavily tuned Proximal Policy Optimization implementation (featuring GAE normalization, early KL-stopping, continuous RPO stochasticity, and gradient clipping).
- **Extensible Distiller**: Abstract distillation loss constraints (`cross_entropy`, `logit_mse`, `mean_mse`, `sample_mse`).
- **Batched Smoothness Evaluator**: Ultra-fast $L_2$ derivative calculations leveraging PyTorch `.backward()` and Monte-Carlo local Lipschitz estimations on entire rollouts simultaneously without python looping structures. 
- **Robustness Evaluator**: Automated validation passes for policies under varying configurations of observation noise to capture generalization capacity out-of-distribution.
- **Synthetic Sampler**: Plug-n-play generation logic extending classical behavior cloning trajectory buffers into bounded spaces using `uniform_global`, `uniform_data_bounds`, `gaussian`, and `mixed` modes.

## Directory Structure

```text
├── synthetic_distillation/
│   ├── configs/                 # Hydra YAML configuration modules
│   ├── src/
│   │   ├── algorithms/          # Highly optimized Actor/Critic PPO
│   │   ├── distillation/        # The Distiller abstraction
│   │   ├── environments/        # Vector env factories
│   │   ├── evaluation/          # Batched Smoothness & Robustness metrics
│   │   ├── data/                # Synthetic sampling modules
│   │   ├── models/              # Discrete & Continuous Agent implementations
│   │   └── logging/             # TBLogger and Multirun Aggregators
│   ├── scripts/
│   │   ├── train_teacher.py     # PPO Teacher entrypoint
│   │   ├── distill_student.py   # Distiller entrypoint
│   │   ├── run_experiment.py    # Metric & Evaluation entrypoints
│   │   └── aggregate.py         # Hydra multi-seed metric aggregation
│   └── main.py                  # Core sequence runner
│
├── docs/                        # Detailed architectural guides and run instructions
└── README.md                    # This overview file
```

## Getting Started

Check out the `/docs/` folder for specific details on how to use the framework:
- [Installation and Execution Details](docs/getting_started.md)
- [How to Run Hydra Sweeps](docs/sweeps_and_aggregation.md)
- [Adding New Environments & Metrics](docs/extending_the_framework.md)

## Basic Usage example

```bash
cd synthetic_distillation
python main.py
```
This will automatically launch the baseline `config.yaml` targetting `cartpole` with a 100-step distillation process and track results instantly!
