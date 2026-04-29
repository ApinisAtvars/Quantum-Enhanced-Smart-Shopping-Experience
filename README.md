# Matrix Factorization to QUBO Workflow

This project documents a practical progression from classical matrix factorization to QUBO-based optimization.

I started by implementing and validating two classical baselines:
- ALS (Alternating Least Squares)
- SGD (Stochastic Gradient Descent)

After establishing those baselines, I reformulated the factorization into a classical QUBO problem and used simulated annealing as the main solving strategy.

The experiments in this repository focus on comparing behavior across these stages, including reconstruction quality, runtime behavior, and convergence trends.

In short: classical MF first, QUBO conversion next, then simulated annealing experiments and analysis.
# NBMF-QUBO — Final Evidence Package

This repository contains the reproducible implementation and final evidence package for a comparison between classical matrix-factorization baselines (ALS, biased SGD, multiplicative-update NMF) and a QUBO-based binary-`H` factorization (NBMF) solved with classical annealing samplers (`SimulatedAnnealing`, `PathIntegralAnnealing`, `ExactSolver` oracle).

All paper claims come from a **single audited run** declared by [`protocol_final.md`](protocol_final.md). Numbers and figures live exclusively in `results_final/` and `plots_final/`. The audit narrative lives in `final_logs/`.

---

## 1. Open these first

| File | Why |
|---|---|
| [`protocol_final.md`](protocol_final.md) | Binding experimental protocol — the single source of truth. |
| [`config/experiment.yaml`](config/experiment.yaml) | Binding configuration; agrees with the protocol. |
| [`final_logs/03_final_claims_dossier.md`](final_logs/03_final_claims_dossier.md) | Claim-by-claim evidence table — start here when writing the paper. |
| [`final_logs/02_final_consistency_audit.md`](final_logs/02_final_consistency_audit.md) | Protocol ↔ config ↔ code ↔ CSV ↔ plot audit. |
| [`final_logs/04_human_decisions_needed.md`](final_logs/04_human_decisions_needed.md) | Editorial decisions that still require a human call. |
| [`final_logs/01_final_run_log.md`](final_logs/01_final_run_log.md) | Per-step timing for the audited run. |

## 2. Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py
```

Tested on Python 3.13 (macOS arm64). The full pipeline completes in ~3.5 minutes on a M2Pro MacBook.

## 3. Pipeline

`run_all.py` executes the following steps in order. Each step can also be run on its own with `python -m experiments.<step>` from the repo root.

| # | Step | Module | Outputs |
|---|---|---|---|
| 1 | Correctness validation | `experiments.correctness` | `correctness_results.csv` |
| 2 | Primary + supplemental benchmark + per-iteration traces | `experiments.benchmark` | `benchmark_primary_results.csv`, `benchmark_supplemental_results.csv`, `convergence_results.csv`, `runtime_breakdown_results.csv` |
| 3 | Convergence analysis (synthetic) | `experiments.convergence` | `convergence_detailed.csv` |
| 4 | Scalability sweep + ExactSolver feasibility | `experiments.scalability` | `scalability_results.csv` |
| 5 | Aggregate primary benchmark with 95% CIs | `experiments.summarize` | `benchmark_primary_summary.csv` |
| 6 | Plot generation (10 figures) | `experiments.plot_results` | `plots_final/*.png` |
| 7 | Machine-checkable compliance audit | `experiments.compliance` | `compliance_matrix.csv` |

A successful run finishes with every row of `compliance_matrix.csv` showing `PASS` and `final_logs/01_final_run_log.md` ending with a per-step summary block.

## 4. Outputs (the only sources of paper numbers and figures)

### `results_final/` — binding CSVs

- `correctness_results.csv`
- `benchmark_primary_results.csv`, `benchmark_primary_summary.csv`, `benchmark_supplemental_results.csv`
- `convergence_results.csv`, `convergence_detailed.csv`
- `runtime_breakdown_results.csv`
- `scalability_results.csv`
- `compliance_matrix.csv`

### `plots_final/` — binding figures

- `convergence_primary.png`, `convergence_by_rank.png`, `convergence_wallclock.png`
- `correctness_small.png`
- `benchmark_error_bar.png`, `benchmark_runtime_bar.png`
- `runtime_breakdown_stacked.png`
- `scalability_runtime_vs_size.png`, `scalability_error_vs_rank.png`
- `feasibility_exactsolver.png`

## 5. Repository map

| Path | Role |
|---|---|
| `protocol_final.md` | Binding protocol (source of truth). |
| `config/experiment.yaml` | Binding configuration. |
| `run_all.py` | One-command pipeline orchestrator. |
| `models/` | NBMF (`nbmf.py`), ALS, biased SGD, QUBO build, samplers. |
| `utils/` | Data loading (`io.py`), synthetic generation, logging. |
| `experiments/` | The 7 pipeline scripts plus `baseline_wrappers.py`. |
| `data/u.data` | MovieLens 100K raw ratings (no preprocessing applied). |
| `results_final/` | Binding CSVs (paper numbers). |
| `plots_final/` | Binding figures (paper figures). |
| `final_logs/` | Run log, consistency audit, claims dossier, decisions (numbered 01–04). |
| `requirements.txt` | Pinned dependencies for the audited run. |
| `archive/` | Local-only historical snapshots (gitignored, not paper evidence). |

## 6. Scope

- **Classical baselines:** `Repo_ALS` (dense-adapted regularized ALS), `Repo_SGD` (dense-adapted biased SGD with global mean and biases).
- **NBMF:** binary-`H` factorization with per-column QUBO subproblems, solved by `SimulatedAnnealingSampler`, `PathIntegralAnnealingSampler`, and `dimod.ExactSolver` (oracle / feasibility probe only).
- **Datasets:** MovieLens 100K dense top-N subblock (primary); noiseless synthetic `W_true ~ U(0,1)`, `H_true ~ Bernoulli(0.5)` (correctness, convergence, scalability, feasibility).
- **Measurements:** reconstruction error, relative error, per-iteration traces, cumulative wall-clock, runtime decomposition (NNLS / QUBO build / sampler / postprocess).
