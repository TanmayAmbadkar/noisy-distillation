# Extending the Framework

## Adding a new Environment
To add a new environment, simply add a `yaml` file to `configs/env/` like `configs/env/mountaincar.yaml`.

```yaml
name: MountainCar-v0
type: discrete
max_episode_steps: 200
```
Hydra automatically resolves the file. **Note**: All environments automatically receive `NormalizeReward` and `TransformReward` (clipping) wrappers during initialization in `make_env.py` to ensure training stability.

## Adding a new synthetic logic form
If you wish to augment existing `mixed` or `gaussian` bounds within the distillation process, you add the rule inside `synthetic_sampler.py`.

```python
        elif self.mode == "custom_perlin_noise":
            return self._sample_custom(batch_size)
```
Add parameters to the `distill/custom.yaml` config, and they will be passed transparently.

## Adding new evaluation metrics
Update `evaluation/smoothness.py` or create a new file like `evaluation/calibration.py`. Ensure that you add the dictionary returns back into `evaluate_all()` defined inside `scripts/run_experiment.py` and the outputs will automatically populate the local `metrics.json` file.
