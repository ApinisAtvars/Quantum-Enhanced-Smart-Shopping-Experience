# NBMF-QUBO — Final Publication Protocol (`protocol_final.md`)

**Version:** 1.0  
**Created:** 2026-04-22  
**Supersedes:** every prior protocol, planning note, and reduced-run README in this repository, including `archive/finalization_backup_20260422T232612Z/legacy_run_20260407_163135/protocol.md` and any narrative summaries preserved under `archive/finalization_backup_20260422T232612Z/`.

---

## 1. Title and Purpose

**Title.** NBMF-QUBO Final Publication Protocol.

**Purpose.** Define the single, binding, pre-declared experimental protocol for the final publication run of this project. Every numerical claim in the paper must trace back to outputs produced under this protocol.

## 2. Source-of-Truth Statement

- This file is the **single source of truth** for the final publication run.
- The authoritative configuration file is `config/experiment.yaml` and it must agree with this document.
- All results used in the paper must live in `results_final/` and `plots_final/`.
- All audit and summary markdown for the final run must live in `final_logs/`.
- Nothing in `archive/`, `results/` (legacy), or `plots/` (legacy) is valid evidence for the paper.

## 3. Non-Negotiable Rules

1. No silent parameter change. Any deviation requires a logged amendment in Section 21.
2. No method substitution. If a method is declared for a tier, it must appear in the outputs of that tier or be explicitly logged as an infeasibility with reason.
3. `NBMF_Exact` is a correctness oracle and feasibility probe only. It **must never** appear in the primary benchmark table.
4. `Repo_ALS` and `Repo_SGD` are **mandatory** primary classical baselines and **mandatory** convergence methods.
5. `NBMF_SA` and `NBMF_PathIntegral` are **mandatory** primary annealing methods.
6. All stochastic primary comparisons must use the declared seed list in full.
7. No claim about quantum hardware behavior. All annealing runs use D-Wave Ocean classical simulators.
8. No claim of asymptotic scaling beyond the tested range.
9. Every row in every summary table in `final_logs/` must come from parsing a file in `results_final/`; manual copy of numbers from old notes is forbidden.

## 4. Final Method Inventory

| ID | Family | Status | Allowed Tiers | Forbidden Tiers | Role |
|---|---|---|---|---|---|
| `Repo_ALS` | Classical (latent-factor ALS, dense adapted) | Required | Convergence, Primary Benchmark | Correctness, Feasibility | Primary baseline |
| `Repo_SGD` | Classical (biased SGD, dense adapted) | Required | Convergence, Primary Benchmark | Correctness, Feasibility | Primary baseline (with declared extra capacity: biases + global mean) |
| `NBMF_SA` | NBMF + `SimulatedAnnealingSampler` | Required | Correctness, Convergence, Primary Benchmark, Scalability | — | Primary annealing method |
| `NBMF_PathIntegral` | NBMF + `PathIntegralAnnealingSampler` | Required | Correctness, Convergence, Primary Benchmark, Scalability | — | Primary annealing method |
| `NBMF_Exact` | NBMF + `dimod.ExactSolver` | Required in tractable regime | Correctness, Convergence (k ≤ 10), Feasibility | Primary Benchmark, Primary Scalability | Oracle / feasibility probe |
| `NMF_MU` | Dense NMF (Lee–Seung multiplicative updates) | Optional supplemental | Supplemental Benchmark | Primary Benchmark | Supplemental continuous-H reference |
| `NBMF_Tabu` | NBMF + `TabuSampler` | Excluded from final run | — | All | Not used (decision, Section 4.2) |
| `NBMF_SteepestDescent` | NBMF + `SteepestDescentSampler` | Excluded from final run | — | All | Not used (decision, Section 4.2) |
| `NMF_SGD` | Projected-gradient dense NMF | Excluded from final run | — | All | Not implemented; see Section 4.3 |

### 4.1 Mandatory Presence Rules

| Tier | Required methods (exactly these) |
|---|---|
| Correctness | `NBMF_Exact`, `NBMF_SA`, `NBMF_PathIntegral` |
| Convergence | `Repo_ALS`, `Repo_SGD`, `NBMF_SA`, `NBMF_PathIntegral`, `NBMF_Exact` |
| Primary Benchmark | `Repo_ALS`, `Repo_SGD`, `NBMF_SA`, `NBMF_PathIntegral` |
| Supplemental Benchmark | `NMF_MU` (same data as primary) |
| Scalability (primary) | `NBMF_SA`, `NBMF_PathIntegral` |
| Feasibility | `NBMF_Exact` |

### 4.2 Decision — Tabu and SteepestDescent

The legacy "Supreme Protocol" listed `NBMF_Tabu` and `NBMF_SteepestDescent` as convergence and supplemental methods. The final protocol **excludes both** from the final run because:

- they add no novel claim the paper intends to make,
- they would dilute the primary convergence plots with 2 extra curves of minimal narrative value,
- the reduced-run execution already removed them, and restoring them would enlarge runtime without new insight.

They remain available in `models/samplers.py` for any future work and are mentioned as such in the paper.

### 4.3 Decision — NMF_SGD

`NMF_SGD` (projected-gradient dense NMF) is **not implemented** in this codebase. It is excluded from the final run. The paper must not reference it as a baseline.

### 4.4 Decision — NMF_MU

`NMF_MU` is **included as a supplemental reference** on the same MovieLens subblock used by the primary benchmark. It is never put into the primary table. It is allowed to appear in the supplemental benchmark CSV and optionally on convergence figures as a dotted supplemental curve if declared in the plotting section.

## 5. Exact Inclusion / Exclusion Rules by Tier

- Include a method in a tier only if Section 4 declares it for that tier.
- Exclude a method from a tier if Section 4 forbids it for that tier, or if it exceeds the declared feasibility limit (Section 13).
- `NBMF_Exact` is allowed in convergence only for ranks `{2, 5, 10}` on the declared synthetic size.
- `NMF_MU` must never appear in `benchmark_primary_results.csv`.

## 6. Dataset Definitions and Preprocessing

### 6.1 MovieLens

| Field | Value |
|---|---|
| Dataset | MovieLens 100K |
| File | `data/u.data` |
| Separator | tab |
| Columns | `userId, itemId, rating, timestamp` |
| Preprocessing | none (raw ratings, no normalization, no threshold) |

### 6.2 MovieLens Dense Subblock Extraction

Implemented by `utils.io.extract_dense_subblock(data_path, n_users, n_items)`:

1. Count interactions per user and per item.
2. Pick the top `n_users` users and top `n_items` items by activity.
3. Build a dense `(n_users, n_items)` rating matrix with `0` for unobserved entries and a matching binary mask.
4. Return `(rating_matrix, mask_matrix, meta)`.

All dense-adapted methods receive `rating_matrix` as their `V`.

### 6.3 Synthetic Data

Implemented by `utils.synthetic.generate_synthetic(n, m, k, noise_std=0.0, seed)`:

- `W_true ~ U(0, 1)` shape `(n, k)`
- `H_true ~ Bernoulli(0.5)` shape `(k, m)`
- `V = W_true @ H_true` with `noise_std = 0.0` for correctness, convergence, scalability, and feasibility

## 7. Seed Lists

| Use case | Exact seed list |
|---|---|
| Primary benchmark | `[42, 43, 44, 45, 46]` |
| Supplemental benchmark | `[42, 43, 44, 45, 46]` |
| Correctness | `[42, 43, 44, 45, 46]` |
| Convergence | `[42, 43, 44, 45, 46]` |
| Scalability | `[42, 43, 44]` |
| Feasibility | `[42]` |

**Decision rationale.** The legacy "Supreme Protocol" declared 10 seeds for primary work. The final protocol uses **5 seeds** because:

- the annealing samplers are low variance at this scale (the reduced run showed `std < 0.01` on relative error),
- 5 seeds are sufficient for t-based 95% CIs to be meaningful,
- this keeps total wall-clock below 15 minutes on the target machine while still quintupling the reduced-run's seed count,
- 3 seeds for scalability keeps that tier's wall-clock under a few minutes while still giving a usable mean ± std per `(size, rank, sampler)` point.

## 8. Matrix Sizes by Tier

| Tier | Data source | Size(s) |
|---|---|---|
| Correctness | synthetic | `10x10` |
| Convergence | synthetic | `30x30` |
| Primary benchmark | MovieLens dense subblock | `100x100` |
| Supplemental benchmark | MovieLens dense subblock | `100x100` (same `V` as primary) |
| Scalability synthetic | synthetic | `10x10`, `20x20`, `50x50`, `100x100` |
| Scalability MovieLens | MovieLens dense subblocks | `30x30`, `50x50`, `100x100` |
| Feasibility | synthetic | `8x8`, `10x10`, `12x12`, `15x15` |

## 9. Rank Grids by Tier

| Tier | Ranks |
|---|---|
| Correctness | `{2, 5}` |
| Convergence | `{2, 5, 10}` |
| Primary benchmark | `{5}` |
| Supplemental benchmark | `{5}` (matches primary) |
| Scalability synthetic | `{2, 5, 10}` |
| Scalability MovieLens | `{2, 5}` |
| Feasibility | `{2, 5, 10}` |

## 10. Iteration Budgets

| Tier | Iterations |
|---|---|
| Correctness | `15` |
| Convergence | `15` |
| Primary benchmark | `15` |
| Supplemental benchmark | `15` |
| Scalability | `15` |
| Feasibility | `15` |

One iteration means:

- **NBMF methods:** one full NNLS sweep over all rows of `W` plus one QUBO solve per column of `H`.
- **Repo_ALS:** one full `fit_step()` updating all user then all item factors.
- **Repo_SGD:** one full epoch over the `n × m` flattened entries with a seeded shuffle.
- **NMF_MU:** one multiplicative-update pass over `H` then `W`.

## 11. Sampler Hyperparameters

| Sampler | num_reads | num_sweeps | Seed policy |
|---|---:|---:|---|
| `SimulatedAnnealingSampler` (`NBMF_SA`) | `20` | `200` | `base_seed + it * m + j` per QUBO call |
| `PathIntegralAnnealingSampler` (`NBMF_PathIntegral`) | `20` | n/a | `base_seed + it * m + j` per QUBO call |
| `dimod.ExactSolver` (`NBMF_Exact`) | n/a | n/a | deterministic |

Baseline hyperparameters:

| Method | factors | learning rate | L2 regularization |
|---|---:|---:|---:|
| `Repo_ALS` | `k` | — | `0.0` |
| `Repo_SGD` | `k` | `0.01` | `0.0` |
| `NMF_MU` | `k` | — | — |

## 12. ExactSolver Feasibility Rules

`dimod.ExactSolver` enumerates all `2^k` binary assignments for every QUBO call. Cost per iteration is therefore `O(m · 2^k)`.

Hard rules:

| Rule | Value |
|---|---|
| Max allowed rank anywhere | `10` |
| Allowed in primary benchmark | **No** |
| Allowed in scalability primary comparison | **No** |
| Feasibility sizes | `8x8, 10x10, 12x12, 15x15` |
| Feasibility ranks | `{2, 5, 10}` |
| Per-run hard timeout | `600` seconds (see Section 13) |

If a feasibility cell times out or raises, write a row with `feasible = false`, `total_runtime = NaN`, and a non-empty `note` column.

## 13. Runtime Instrumentation

Every NBMF run must log:

- `total_runtime`
- `total_nnls_time`
- `total_qubo_build_time`
- `total_annealing_time`
- `total_postprocess_time`
- `num_qubo_solves = num_iterations * m`
- `avg_qubo_solve_time = total_annealing_time / num_qubo_solves`

Every classical baseline run must log:

- `total_runtime`
- `avg_iter_time`

The per-run hard timeout is `600` seconds. This value **must** appear in `experiments/scalability.py` as the constant used for feasibility. Code and this protocol must match.

## 14. Required CSV Outputs

All CSVs live under `results_final/`.

| File | Produced by | Required columns (minimum) |
|---|---|---|
| `correctness_results.csv` | `experiments/correctness.py` | `method, matrix_size, n, m, rank, seed, h_recovery_rate, final_error, final_relative_error, total_runtime` |
| `benchmark_primary_results.csv` | `experiments/benchmark.py` | `method, rank, seed, final_error, final_relative_error, total_runtime, avg_iter_time, matrix_size` (plus NNLS / QUBO / annealing / postprocess for NBMF methods) |
| `benchmark_primary_summary.csv` | `experiments/summarize.py` (new) | `method, n, mean, std, ci_lower, ci_upper` for both `final_relative_error` and `total_runtime` |
| `benchmark_supplemental_results.csv` | `experiments/benchmark.py` | same schema as primary |
| `convergence_results.csv` | `experiments/benchmark.py` | per-iteration traces on MovieLens for primary methods: `iteration, error, relative_error, iter_time, cumulative_time, method, seed, rank, tier` |
| `convergence_detailed.csv` | `experiments/convergence.py` | per-iteration traces on synthetic: `iteration, error, relative_error, iter_time, cumulative_time, method, data_source, matrix_size, rank, seed` |
| `runtime_breakdown_results.csv` | `experiments/benchmark.py` | per-seed per-method: `method, rank, seed, total_runtime, nnls_time, qubo_build_time, annealing_time, postprocess_time, num_qubo_solves, avg_qubo_solve_time` |
| `scalability_results.csv` | `experiments/scalability.py` | `data_source, matrix_size, n, m, rank, sampler, seed, final_error, final_relative_error, total_runtime, avg_iter_time, total_nnls_time, total_qubo_build_time, total_annealing_time, total_postprocess_time, num_qubo_solves, avg_qubo_solve_time, feasible (where applicable), note` |
| `compliance_matrix.csv` | `experiments/compliance.py` (new) | `requirement, status, tier, evidence_file, evidence_detail` |

## 15. Required PNG Outputs

All PNGs live under `plots_final/`.

| File | Source CSV(s) |
|---|---|
| `convergence_primary.png` | `convergence_results.csv` (MovieLens primary traces, `tier == 'primary'`) |
| `convergence_wallclock.png` | `convergence_detailed.csv` (synthetic, error vs cumulative time, per rank) |
| `convergence_by_rank.png` | `convergence_detailed.csv` |
| `correctness_small.png` | `correctness_results.csv` |
| `benchmark_error_bar.png` | `benchmark_primary_results.csv` |
| `benchmark_runtime_bar.png` | `benchmark_primary_results.csv` |
| `runtime_breakdown_stacked.png` | `runtime_breakdown_results.csv` |
| `scalability_runtime_vs_size.png` | `scalability_results.csv` (non-feasibility rows) |
| `scalability_error_vs_rank.png` | `scalability_results.csv` (non-feasibility rows) |
| `feasibility_exactsolver.png` | `scalability_results.csv` (`data_source == 'feasibility'`) |

## 16. Required Markdown Outputs

All of these live under `final_logs/`:

- `01_final_run_log.md` — timestamped log of each tier's command, duration, output files, and warnings.
- `02_final_consistency_audit.md` — protocol vs config vs code vs CSV vs plot vs summary audit.
- `03_final_claims_dossier.md` — claim-by-claim evidence table and paper-writing brief.
- `04_human_decisions_needed.md` — remaining editorial decisions that require a human call (not resolvable from evidence).

## 17. Compliance Checklist

A machine-checkable compliance matrix (`results_final/compliance_matrix.csv`) must confirm all of:

1. Primary benchmark CSV contains exactly `{Repo_ALS, Repo_SGD, NBMF_SA, NBMF_PathIntegral}` as distinct methods.
2. Primary benchmark CSV contains exactly 5 distinct seeds matching Section 7.
3. Primary benchmark CSV contains exactly one `matrix_size` equal to the actual extracted MovieLens subblock shape produced by the `100x100` request (may be ≤ 100×100 if dataset has fewer top-N users/items; the actual shape is recorded).
4. Primary benchmark rank is `5` for every row.
5. Primary benchmark CSV does **not** contain `NBMF_Exact`, `NBMF_Tabu`, `NBMF_SteepestDescent`, `NMF_MU`, or `NMF_SGD`.
6. Correctness CSV contains exactly `{NBMF_Exact, NBMF_SA, NBMF_PathIntegral}`, 5 seeds, sizes `{10x10}`, ranks `{2, 5}`.
7. Convergence detailed CSV contains exactly `{Repo_ALS, Repo_SGD, NBMF_SA, NBMF_PathIntegral, NBMF_Exact}` on `30x30`, ranks `{2, 5, 10}`, 5 seeds.
8. Runtime breakdown CSV has all required columns (`nnls_time`, `qubo_build_time`, `annealing_time`, `postprocess_time`, `num_qubo_solves`, `avg_qubo_solve_time`) populated for both NBMF methods.
9. Scalability CSV contains both samplers `{simulated_annealing, path_integral}` on synthetic `{10, 20, 50, 100}` × ranks `{2, 5, 10}` (with `k < n` filter) and MovieLens `{30, 50, 100}` × ranks `{2, 5}`, 3 seeds each.
10. Feasibility subset has `data_source == feasibility`, sizes `{8, 10, 12, 15}`, ranks `{2, 5, 10}` (with `k < n` filter), 1 seed.
11. All aggregate CSVs include `mean, std, count, ci_lower, ci_upper` for `final_relative_error` and `total_runtime`.
12. Each required PNG in Section 15 exists and was created at or after the CSV it depends on.
13. The TIMEOUT constant in `experiments/scalability.py` is `600`, matching Section 12 and Section 13.

Any FAIL must be fixed and the relevant tier rerun before the final claims dossier is accepted.

## 18. Scope of Claims

The paper may claim:

- Reconstruction quality on the tested `100×100` MovieLens dense subblock at rank 5.
- Per-iteration and wall-clock convergence behavior on the tested 30×30 synthetic problem.
- Runtime decomposition (NNLS / QUBO build / annealing / postprocess) for `NBMF_SA` and `NBMF_PathIntegral` on the tested primary benchmark.
- Empirical scaling trends within the tested synthetic size and rank range.
- Correctness of the QUBO formulation on noiseless 10×10 synthetic problems.
- ExactSolver feasibility limits within the tested sizes and ranks.

The paper may **not** claim:

- Generalization to the full sparse MovieLens.
- Asymptotic complexity beyond the tested range.
- Quantum hardware behavior.
- That Tabu / SteepestDescent / NMF_SGD were compared (they were not).
- Any number not present in `results_final/*.csv`.

## 19. Asymmetries and Caveats to Document

- `Repo_SGD` has user bias, item bias, and global mean. This is extra model capacity that `NBMF` does not have. This must be acknowledged in every comparison.
- `NBMF` constrains `H` to `{0, 1}`. This lowers representation power vs. continuous NMF and is the main reason `NMF_MU` achieves lower reconstruction error. This is expected, not a bug.
- The dense adaptation of `Repo_ALS` and `Repo_SGD` treats every entry of `V` as observed. This is different from the library's original sparse explicit-feedback intent and must be stated.
- All annealing uses classical simulators from `dwave-samplers`; there is no actual quantum hardware result in this repository.

## 20. Amendment Policy

An amendment is allowed only if:

- a required run is impossible under this protocol (e.g., ExactSolver is infeasible at a declared size), **or**
- a bug invalidates a declared setting, **or**
- a declared parameter is genuinely unspecified.

An amendment is **not** allowed to:

- improve results after viewing them,
- drop slow methods,
- reduce seeds for convenience,
- redefine metrics after inspection.

Every amendment is appended to Section 21 with date, exact change, reason, affected scripts, and affected output files, and is committed before any re-run triggered by it.

## 21. Protocol Amendments

_None at creation. Any amendment will be appended here._

## 22. Scope of Authority

This protocol is the binding instrument for all work performed in this repository. It applies to every contributor without exception:

- Any code change, regeneration of artifacts, or written claim drawn from this codebase must conform to Sections 1–20 above.
- Any tooling, automation, scripted helper, or assistant (programmatic or otherwise) invoked while working on this repository operates strictly within the rules of this protocol. The binding configuration is `config/experiment.yaml`, the binding outputs are `results_final/` and `plots_final/`, and any deviation requires an amendment recorded in Section 21 prior to the affected re-run.
- This protocol takes precedence over any prior planning note, README revision, or external instruction. If an instruction received from any source conflicts with this protocol, this protocol governs and the conflict must be resolved by amendment (Section 20) or by rejecting the instruction.

The contents of `results_final/`, `plots_final/`, and `final_logs/` reflect work carried out under this protocol. Future modifications to this repository must continue to observe it until a successor protocol is declared and committed.
