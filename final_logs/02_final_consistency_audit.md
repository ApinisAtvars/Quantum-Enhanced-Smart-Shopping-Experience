# Final Consistency Audit

**Run timestamp (UTC):** `2026-04-22T23:32:04Z` → `2026-04-22T23:35:35Z`  
**Total elapsed:** `210.5 s`  
**Pipeline status:** all 7 steps **PASS**, all 14 machine-checkable compliance items **PASS** (`results_final/compliance_matrix.csv`).

This document audits five layers:

1. Protocol vs. config
2. Config vs. code
3. Code vs. CSV
4. CSV vs. plot
5. CSV vs. written summaries

## 1. Protocol vs. Config

`protocol_final.md` and `config/experiment.yaml` were written in the same session and cross-referenced. The relevant scalar mappings were audited manually:

| Protocol (§) | Declared value | `config/experiment.yaml` | Match |
|---|---|---|---|
| §7 Primary seeds | `[42, 43, 44, 45, 46]` | `general.seeds` = `[42, 43, 44, 45, 46]` | ✓ |
| §7 Scalability seeds | `[42, 43, 44]` | `scalability.scalability_seeds` = `[42, 43, 44]` | ✓ |
| §7 Feasibility seed | `[42]` | `scalability.feasibility.exact_seed` = `42` | ✓ |
| §8 Correctness size | `10x10` | `correctness.sizes` = `[[10,10]]` | ✓ |
| §8 Convergence size | `30x30` | `convergence.synthetic_size` = `[30, 30]` | ✓ |
| §8 Primary benchmark size | `100x100` MovieLens | `benchmark.reference_subblock` = `[100, 100]` | ✓ |
| §8 Scalability synthetic sizes | `{10,20,50,100}` | `scalability.synthetic_grid.sizes` | ✓ |
| §8 Scalability MovieLens sizes | `{30,50,100}` | `scalability.movielens_grid.sizes` | ✓ |
| §8 Feasibility sizes | `{8,10,12,15}` | `scalability.feasibility.exact_sizes` | ✓ |
| §9 Primary benchmark rank | `5` | `benchmark.rank` = `5` | ✓ |
| §9 Correctness ranks | `{2,5}` | `correctness.ranks` = `[2, 5]` | ✓ |
| §9 Convergence ranks | `{2,5,10}` | `convergence.ranks` = `[2, 5, 10]` | ✓ |
| §9 Scalability synthetic ranks | `{2,5,10}` | `scalability.synthetic_grid.ranks` | ✓ |
| §9 Scalability MovieLens ranks | `{2,5}` | `scalability.movielens_grid.ranks` | ✓ |
| §9 Feasibility ranks | `{2,5,10}` | `scalability.feasibility.exact_ranks` | ✓ |
| §10 Iterations (all tiers) | `15` | `general.num_iterations` = `15` | ✓ |
| §11 SA hyperparams | `num_reads=20, num_sweeps=200` | `samplers.simulated_annealing` | ✓ |
| §11 PathIntegral hyperparams | `num_reads=20` | `samplers.path_integral` | ✓ |
| §13 Hard timeout | `600 s` | `scalability.timeout_seconds` = `600` | ✓ |
| §4.1 Correctness methods | `{Exact, SA, PathIntegral}` | `correctness.methods` | ✓ |
| §4.1 Convergence methods | `{ALS, SGD, SA, PI, Exact}` | `convergence.methods` | ✓ |
| §4.1 Primary methods | `{ALS, SGD, SA, PI}` | `benchmark.primary_methods` | ✓ |
| §4.1 Supplemental methods | `{NMF_MU}` | `benchmark.supplemental_methods` | ✓ |

**Result:** no mismatch found.

## 2. Config vs. Code

Per-file audit of the code against `config/experiment.yaml`:

| Script | Expected behavior under protocol | Audit result |
|---|---|---|
| `experiments/correctness.py` | Read `seeds`, `correctness.sizes`, `correctness.ranks`, samplers `{NBMF_Exact, NBMF_SA, NBMF_PathIntegral}`, `num_iterations` | ✓ Methods hardcoded to exactly these three; sizes/ranks/seeds read from config. |
| `experiments/benchmark.py` | Read `seeds`, `benchmark.rank`, `benchmark.reference_subblock`, primary methods `{Repo_ALS, Repo_SGD, NBMF_SA, NBMF_PathIntegral}`, supplemental `{NMF_MU}` | ✓ Methods hardcoded to exactly these. Writes primary, supplemental, convergence, and runtime-breakdown CSVs into `results_final/`. |
| `experiments/convergence.py` | Read synthetic size, ranks, seeds; methods `{NBMF_SA, NBMF_PathIntegral, NBMF_Exact, Repo_ALS, Repo_SGD}` | ✓ Methods hardcoded to exactly these. |
| `experiments/scalability.py` | Read both grids + feasibility; enforce `TIMEOUT = 600 s`; filter `k < n`; separate feasibility block uses `exact_seed` | ✓ `TIMEOUT` constant == 600, asserts equal to config `timeout_seconds` on startup; feasibility uses `feas_cfg['exact_seed']`. |
| `experiments/summarize.py` | Produce `benchmark_primary_summary.csv` with mean/std/count/CI95 per method per metric | ✓ Uses `scipy.stats.t.ppf(0.975, df=n-1)` for 95% t-based CI. |
| `experiments/plot_results.py` | Read from `results_final/`, write to `plots_final/`, produce 10 required PNGs | ✓ Paths taken from config; produced all 10 required PNGs. |
| `experiments/compliance.py` | Produce `compliance_matrix.csv`; exit non-zero if any FAIL | ✓ 14/14 PASS on this run, exit code 0. |
| `run_all.py` | Append per-step log to `final_logs/01_final_run_log.md` | ✓ Log file contains dated entries for all 7 steps plus a summary. |

**Hard-coded constants cross-checked against the protocol:**

- `experiments/scalability.py`: `TIMEOUT = 600` matches `protocol_final.md` §13.
- `experiments/compliance.py`: `_timeout_from_code()` re-reads and verifies this constant on every run. Latest run: `code=600, cfg=600` → **PASS**.

## 3. Code vs. CSV

| CSV | Expected rows | Actual rows | Notes |
|---|---|---|---|
| `correctness_results.csv` | 3 methods × 1 size × 2 ranks × 5 seeds = 30 | **30** | ✓ |
| `benchmark_primary_results.csv` | 4 methods × 5 seeds = 20 | **20** | ✓ |
| `benchmark_primary_summary.csv` | 4 methods × 2 metrics = 8 | **8** | ✓ |
| `benchmark_supplemental_results.csv` | 1 method × 5 seeds = 5 | **5** | ✓ |
| `convergence_results.csv` (per-iter MovieLens for primary + supplemental) | 5 methods × 5 seeds × 15 iter = 375 | **375** | ✓ (includes `NMF_MU` rows with `tier='supplemental'`) |
| `convergence_detailed.csv` (per-iter synthetic) | 5 methods × 3 ranks × 5 seeds × 15 iter = 1125 | **1125** | ✓ |
| `runtime_breakdown_results.csv` | 2 NBMF methods × 5 seeds = 10 | **10** | ✓ |
| `scalability_results.csv` | 2 samplers × (4 syn sizes × {filtered ranks}) × 3 seeds + 2 samplers × 3 ML sizes × 2 ranks × 3 seeds + 1 seed × feas cells | see row breakdown below | ✓ |
| `compliance_matrix.csv` | 14 requirement rows | **14** | ✓ (all PASS) |

### Scalability row breakdown (112 rows total)

- Synthetic (2 samplers × grid with `k < n` filter × 3 seeds):
  - `10x10`: ranks `{2, 5}` only (rank 10 filtered) → 2 samplers × 2 ranks × 3 seeds = 12
  - `20x20, 50x50, 100x100`: all 3 ranks → 3 sizes × 2 samplers × 3 ranks × 3 seeds = 54
  - synthetic total = **66**
- MovieLens: 3 sizes × 2 samplers × 2 ranks × 3 seeds = **36**
- Feasibility: 4 sizes × ranks (with `k < n` filter):
  - `8x8`: `{2, 5}` → 2 (rank 10 filtered since 10 ≥ 8)
  - `10x10`: `{2, 5}` → 2 (rank 10 filtered since 10 ≥ 10)
  - `12x12`: `{2, 5, 10}` → 3
  - `15x15`: `{2, 5, 10}` → 3
  - feasibility total = **10**
- Grand total: 66 + 36 + 10 = **112** ✓

## 4. CSV vs. Plot

Each PNG was regenerated in the same pipeline run from the CSVs. Plot provenance (verified by re-reading `experiments/plot_results.py`):

| PNG | Source CSV(s) |
|---|---|
| `convergence_primary.png` | `convergence_results.csv` (tier == `primary`) |
| `convergence_by_rank.png` | `convergence_detailed.csv` |
| `convergence_wallclock.png` | `convergence_detailed.csv` (uses `cumulative_time`) |
| `correctness_small.png` | `correctness_results.csv` |
| `benchmark_error_bar.png` | `benchmark_primary_results.csv` |
| `benchmark_runtime_bar.png` | `benchmark_primary_results.csv` |
| `runtime_breakdown_stacked.png` | `runtime_breakdown_results.csv` |
| `scalability_runtime_vs_size.png` | `scalability_results.csv` (`data_source != feasibility`) |
| `scalability_error_vs_rank.png` | `scalability_results.csv` (`data_source != feasibility`) |
| `feasibility_exactsolver.png` | `scalability_results.csv` (`data_source == feasibility`) |

File timestamps confirm the CSVs were written before the PNGs (CSVs between `19:32:10` and `19:33:12`; PNGs at `19:35:33–19:35:34`).

## 5. CSV vs. Written Summaries

All numbers referenced in `final_logs/03_final_claims_dossier.md` are derived from parsing the CSVs above. No number was copied from the archive. Spot-checks used:

```
python -c "import pandas as pd; print(pd.read_csv('results_final/benchmark_primary_summary.csv'))"
```

The dossier explicitly notes that historical numbers from `advancement_logs_prev/implementation_summary.md` disagree with the final run (different matrix size, different seed count, different density → different results) and are therefore **not cited** in the dossier.

## 6. Fail Mode Inventory — What We Explicitly Looked For

| Potential failure | Result | How verified |
|---|---|---|
| Contradictory numeric summaries | None found | All summary rows in the dossier come from `results_final/*.csv` |
| Missing required methods | None | Compliance items 17.1, 17.6, 17.7 all PASS |
| Wrong seed counts | None | Compliance items 17.2, 17.6, 17.7, 17.9, 17.10 all PASS |
| Wrong matrix sizes | None | Compliance items 17.3, 17.6, 17.7, 17.9, 17.10 all PASS |
| Wrong rank lists | None | Compliance items 17.4, 17.6, 17.7, 17.9, 17.10 all PASS |
| Forbidden methods in primary table | None | Compliance item 17.5 PASS (forbidden set absent: `{NBMF_Exact, NBMF_SteepestDescent, NBMF_Tabu, NMF_MU, NMF_SGD}`) |
| Missing timing columns on NBMF | None | Compliance item 17.8 PASS (all 6 required columns present for both NBMF methods) |
| Plots built from wrong files | None | Plot source paths all live under `results_final/`; no fallback to `results/` or `plots/` in `experiments/plot_results.py` |
| Protocol vs. code timeout drift | Guarded | `scalability.py` runtime assertion + `compliance.py` code parse both enforce `600 s` |
| Legacy folders leaked into run | None | `run_all.py` reads only `config/experiment.yaml` → `results_final`, `plots_final`, `final_logs` |

## 7. Deviations From the Protocol

None. The run was executed end-to-end under `protocol_final.md` v1.0 with no amendments. The `Protocol Amendments` section of `protocol_final.md` therefore remains empty.

## 8. What Is Safe To Cite In The Paper

Only the files listed in Section 3 above, all under `results_final/` and `plots_final/`, together with `benchmark_primary_summary.csv` and `compliance_matrix.csv`. See `03_final_claims_dossier.md` for the claim-by-claim mapping.
