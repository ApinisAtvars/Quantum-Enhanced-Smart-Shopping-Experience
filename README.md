# Quantum Annealing for Matrix Factorization

This repository contains a compact research implementation for comparing classical matrix factorization baselines against a QUBO-based nonnegative/binary matrix factorization pipeline.

The code is organized so the main navigation is straightforward:

- `models/` contains the ALS, SGD, NBMF, QUBO, and sampler implementations
- `utils/` contains shared data loading, metrics, synthetic data generation, and I/O helpers
- `experiments/` contains runnable scripts for training, benchmarking, convergence, scalability, and plotting
- `config/` contains all configuration files in one place
- `data/` contains MovieLens data
- `results/` and `plots/` contain generated outputs
- `advancement_logs/` contains the written project summary

## Main Entry Points

Run the full research pipeline:

```bash
python run_all.py
```

Run individual classical training scripts:

```bash
python -m experiments.train_als --data data/u.data --out outputs/als --config config/als.json
python -m experiments.train_sgd --data data/u.data --out outputs/sgd --config config/sgd.json
python -m experiments.train_nbmf --config config/experiment.yaml
```

## Repository Layout

```text
models/
  als.py
  sgd.py
  nbmf.py
  qubo.py
  samplers.py

utils/
  data.py
  io.py
  metrics.py
  synthetic.py
  benchmark.py
  logging.py

experiments/
  train_als.py
  train_sgd.py
  train_nbmf.py
  benchmark.py
  convergence.py
  correctness.py
  scalability.py
  plot_results.py

config/
  als.json
  sgd.json
  experiment.yaml
```

## What The Pipeline Produces

- `results/correctness_results.csv`
- `results/benchmark_primary_results.csv`
- `results/benchmark_supplemental_results.csv`
- `results/convergence_results.csv`
- `results/convergence_detailed.csv`
- `results/runtime_breakdown_results.csv`
- `results/scalability_results.csv`
- `plots/*.png`

## Notes

- MovieLens is the main reference dataset.
- Small synthetic matrices are used for QUBO sanity checks and tractable validation.
- The current structure is intentionally flatter than the earlier layout: one config directory, one models directory, one utils directory, one experiments directory.
