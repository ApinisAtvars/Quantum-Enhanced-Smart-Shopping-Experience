# AI-Generated Implementation Summary

Date: 2026-04-07

## What Was Reused

The codebase from Weeks 3–4 was almost entirely reusable. The following modules were kept as-is:

- **models/als.py** — Alternating Least Squares for dense matrix factorization (per-user/per-item regularized least-squares solves).
- **models/sgd.py** — Biased SGD with global mean, user bias, item bias, and latent factors.
- **models/nbmf.py** — Nonneg/Binary MF via alternating NNLS + QUBO. Each iteration updates W row-wise via scipy NNLS, then H column-wise by solving a QUBO per column.
- **models/qubo.py** — QUBO construction: minimizes ||v_j - Wq||^2 over binary q, expanding to diagonal (W^T W)_{ii} - 2(W^T v)_i and off-diagonal 2(W^T W)_{ij} terms.
- **models/samplers.py** — Sampler dispatch for SimulatedAnnealing, PathIntegral, Tabu, SteepestDescent, ExactSolver (all from dwave-samplers/dimod).
- **utils/** — Data loading (MovieLens u.data), synthetic data generation (nonneg W, binary H), metrics, IO utilities, logging.
- **experiments/baseline_wrappers.py** — Adapts ALS and BiasedSGD to operate on dense matrices for fair comparison with NBMF.

## What Was Changed

1. **config/experiment.yaml** — Simplified from 10 seeds to 3, reduced matrix sizes and rank grid for practical runtime (~90s total). Benchmark subblock reduced from 200×200 to 50×50.
2. **experiments/benchmark.py** — Streamlined to run 4 primary methods (ALS, SGD, NBMF+SA, NBMF+PathIntegral) and 1 supplemental (NMF-MU). Removed Tabu/SteepestDescent from benchmark (kept available in the sampler registry for manual use).
3. **experiments/convergence.py** — Runs on 20×20 synthetic with ranks {2, 5}, compares NBMF+SA, NBMF+PathIntegral, NBMF+Exact, ALS, and SGD.
4. **experiments/scalability.py** — Synthetic sizes {10, 20, 50} and MovieLens {30, 50} with ranks {2, 5, 10}. ExactSolver feasibility on sizes {8, 10, 15}.
5. **experiments/plot_results.py** — Rewritten to produce clean plots without the old compliance matrix generation.
6. **run_all.py** — New master script that runs the full pipeline (correctness → benchmark → convergence → scalability → plots) and reports status.
7. **Archive** — Previous plots/, results/, advancement_logs/, and protocol.md moved to `archive/legacy_run_20260407_163135/`.

## Experiments Run

### 1. Correctness Validation (8×8 synthetic, ranks 2 and 4, 3 seeds)
- **Methods**: NBMF+Exact, NBMF+SA, NBMF+PathIntegral
- **Result**: 100% H recovery rate for all methods on all seeds. Reconstruction error ≈ machine epsilon (~1e-15 to 1e-16). The QUBO formulation is correct and all samplers find the ground truth on small noiseless problems.

### 2. Main Benchmark (50×50 MovieLens subblock, rank 5, 15 iterations, 3 seeds)

| Method             | Relative Error (mean ± std) | Runtime (mean ± std) |
|--------------------|----------------------------|---------------------|
| Repo_ALS           | 0.3209 ± 0.0002            | 0.026s ± 0.012s     |
| Repo_SGD           | 0.3632 ± 0.0054            | 0.232s ± 0.009s     |
| NBMF_SA            | 0.3674 ± 0.0039            | 0.369s ± 0.014s     |
| NBMF_PathIntegral  | 0.3674 ± 0.0039            | 1.781s ± 0.013s     |
| NMF_MU (suppl.)    | 0.3003 ± 0.0000            | 0.001s ± 0.000s     |

**Observations**:
- ALS achieves the lowest error among the primary methods (~0.321 relative error) and is by far the fastest (26ms for 15 iterations).
- NBMF+SA and NBMF+PathIntegral converge to the same final error (~0.367), which is slightly worse than both classical methods. This is expected: the binary constraint on H limits expressiveness.
- PathIntegral is ~4.8× slower than SA due to its sampling mechanism, with no accuracy benefit in this setting.
- NMF-MU (multiplicative updates) achieves the best error overall at ~0.300 — expected since it has no binary constraint and is the natural dense NMF solver.

### 3. Convergence Analysis (20×20 synthetic, ranks 2 and 5, 3 seeds, 15 iterations)
- On rank-2 synthetic problems, NBMF+Exact/SA/PathIntegral all converge to machine-epsilon error within 1–2 iterations.
- ALS converges rapidly as well. SGD is slower per-iteration but also reaches near-zero error.
- On rank-5, similar pattern with slightly more iterations needed.
- Wall-clock convergence shows ALS reaching convergence fastest, followed by SA, then PathIntegral.

### 4. Scalability Sweep
- **Synthetic**: Sizes 10–50, ranks 2–10, SA sampler. Runtime scales roughly linearly with n×m (each extra column adds one QUBO solve per iteration). Error is near-zero for rank ≤ 5 on synthetic. At rank=10 on 20×20, occasional nonzero residuals (~0.57 in one seed) indicate SA occasionally gets stuck.
- **MovieLens**: 30×30 and 50×50 subblocks, ranks 2 and 5. Relative errors ~0.30–0.41 depending on rank and size.
- **ExactSolver feasibility**: Runs in <200ms up to 15×15 with rank=10. Exponential blowup begins beyond rank ~15 (2^k search space).

### 5. Runtime Breakdown (NBMF variants on 50×50 MovieLens)
- **NBMF+SA**: 97% of time in sampler (annealing), 1.5% NNLS, 1.5% QUBO build, <0.1% postprocess.
- **NBMF+PathIntegral**: 99.2% in sampler, 0.3% NNLS, 0.3% QUBO build. PathIntegral per-QUBO-solve ~2.35ms vs SA ~0.48ms.
- Average QUBO solve time: SA ≈ 0.48ms, PathIntegral ≈ 2.35ms.
- Total QUBO solves per full run: 750 (15 iterations × 50 columns).

## Generated Outputs

### CSVs (results/)
- `correctness_results.csv` — 18 rows (3 methods × 2 ranks × 3 seeds)
- `benchmark_primary_results.csv` — 12 rows (4 methods × 3 seeds)
- `benchmark_supplemental_results.csv` — 3 rows (NMF_MU × 3 seeds)
- `convergence_results.csv` — 225 rows (per-iteration traces from benchmark)
- `convergence_detailed.csv` — 450 rows (per-iteration traces from synthetic)
- `runtime_breakdown_results.csv` — 6 rows (2 NBMF variants × 3 seeds)
- `scalability_results.csv` — 31 rows (synthetic + MovieLens + feasibility)

### Plots (plots/)
- `correctness_small.png` — Bar chart of final error for Exact vs SA vs PathIntegral on 8×8
- `benchmark_bars.png` — Side-by-side bars for relative error and runtime
- `convergence_primary.png` — Error vs iteration on MovieLens subblock
- `convergence_by_rank.png` — Error vs iteration by rank on synthetic
- `convergence_wallclock.png` — Error vs wall-clock time on synthetic
- `runtime_breakdown_stacked.png` — Stacked bar of NBMF component times
- `scalability.png` — Runtime vs size and error vs rank
- `feasibility_exact.png` — ExactSolver runtime scaling with rank

## Limitations

1. **Binary constraint**: NBMF restricts H to {0,1}, which limits approximation quality vs continuous NMF. This is inherent to the QUBO formulation, not a bug.
2. **Scale**: QUBO-based MF scales as O(iterations × columns × QUBO_solve_time), making it impractical beyond ~100 columns without parallelization or hardware annealing.
3. **SA vs hardware**: All annealing here uses D-Wave's classical simulators (SimulatedAnnealingSampler, PathIntegralAnnealingSampler). Real quantum annealer behavior may differ.
4. **MovieLens subblock**: The 50×50 dense subblock is a heavily filtered view of ML-100K. Results may not generalize to the full sparse dataset.
5. **No hyperparameter tuning**: SA uses 20 reads / 200 sweeps throughout. Different settings could shift the tradeoffs.
