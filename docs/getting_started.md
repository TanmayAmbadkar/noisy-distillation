# Getting Started

## Dependencies

You'll need `torch`, `gymnasium[mujoco]`, `tensorboard`, and the `hydra` scaling extensions:
```bash
pip install hydra-core hydra-colorlog func_to_script torch tensorboard numpy "gymnasium[mujoco]"
```

## Running the framework

The pipeline executes everything through `main.py` located centrally in `/synthetic_distillation`:
```bash
cd synthetic_distillation
python main.py
```

Under the hood, `main.py` is dynamically controlled by `synthetic_distillation/configs/config.yaml`.
You can override parameters directly from the terminal without changing the YAML files:

```bash
# Run on Hopper instead
python main.py env=hopper

# Change the PPO updates length and batch size
python main.py env=halfcheetah algo.updates=1000 algo.rollout_steps=2048

# Swap sampling modes to 'gaussian' and increase expansion boundaries
python main.py env=ant distill=gaussian distill.sampling.std=0.15
```

## Folder Outputs

The framework natively segregates outputs securely by configuration using the format `${env.name}/${distill.mode}/seed_${seed}`. For example, the runs might populate into:
```text
outputs/
  └── Hopper-v4/
      └── gaussian/
          └── seed_0/
              ├── .hydra/                # Saved overrides/configs
              ├── tensorboard/           # Stored TB plots
              ├── metrics.json           # All computed outputs!
              ├── teacher.pt
              └── student.pt
```
