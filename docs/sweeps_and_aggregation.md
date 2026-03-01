# Advanced Sweeps and Aggregation

Hydra is configured to track large-scale research multiruns utilizing the `-m` tag out of the box.

## Launching a Grid Sweep

You can construct Cartesian sweeps natively inline with `main.py`:
```bash
python main.py -m \
       seed=0,1,2,3,4 \
       env=hopper,halfcheetah,ant \
       distill.sampling.mode=trajectory,gaussian,uniform_data_bounds,mixed
```

Or you can use the predefined experimental template logic inside of `configs/experiment/full_study.yaml`:
```bash
python main.py +experiment=full_study
```
This triggers the `sweeper` directives which expands all combinations simultaneously into separate target isolation groups within `outputs/multirun/YYYY-MM-DD/HH-MM-SS/`.

## Aggregating Sweeps

Because output data is separated via the Hydra directory structure under standard formatting `seed_N` runs, the logger has an aggregator tool.

```bash
python scripts/aggregate.py outputs/multirun/2026-02-27/15-30-00/
```

This will recurse through all of the `metrics.json` outputs built by the robustness evaluator and generate:
- `summary.json`: A standard layout mapping `{"grad_norm_mean": {"mean": 0.5, "std": 0.05, "count": 5}}`
- `summary.csv`: Tabular equivalents ready to be imported into plotting abstractions!
