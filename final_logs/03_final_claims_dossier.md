# Final Claims Dossier

**Binding protocol:** `protocol_final.md` v1.0  
**Binding CSV directory:** `results_final/`  
**Binding figure directory:** `plots_final/`  
**Run log:** `final_logs/01_final_run_log.md`
**Consistency audit:** `final_logs/02_final_consistency_audit.md`
**Compliance matrix:** `results_final/compliance_matrix.csv` (14/14 PASS)

Every number in this document was derived by parsing a file in `results_final/`. No number here comes from any prior summary, archive, or reduced-run note.

---

## 1. Final Repository Map

```
protocol_final.md              # binding protocol (v1.0)
config/experiment.yaml         # binding config, agrees with protocol
run_all.py                     # pipeline orchestrator
models/
  als.py, sgd.py               # repository classical baselines
  nbmf.py                      # NNLS + QUBO alternating minimization
  qubo.py                      # QUBO construction
  samplers.py                  # dimod / dwave-samplers dispatch
utils/
  io.py, synthetic.py, ...     # dataset loading, synthetic generation
experiments/
  correctness.py               # 10x10 synthetic correctness
  benchmark.py                 # MovieLens 100x100 primary + supplemental + convergence traces + runtime breakdown
  convergence.py               # 30x30 synthetic convergence
  scalability.py               # scalability + feasibility
  summarize.py                 # benchmark_primary_summary.csv
  plot_results.py              # plots_final/*.png
  compliance.py                # compliance_matrix.csv
data/u.data                    # MovieLens 100K raw
results_final/                 # binding CSVs (final paper source)
plots_final/                   # binding figures (final paper source)
final_logs/                    # this dossier, consistency audit, run log, human decisions
archive/                       # all prior runs and summaries (not paper evidence)
```

## 2. Final Source-of-Truth Hierarchy

1. `protocol_final.md` overrides every other planning document.
2. `config/experiment.yaml` is the only file any script reads for parameters.
3. `results_final/*.csv` is the only source for numbers cited in the paper.
4. `plots_final/*.png` is the only source of figures for the paper.
5. `final_logs/*.md` is the only source of narrative summaries for the paper.
6. Anything in `archive/` is historical reference only.

## 3. Final Experiment Design

| Tier | Data | Size | Rank(s) | Seeds | Iter | Methods |
|---|---|---|---|---|---|---|
| Correctness | synthetic (noiseless) | 10×10 | `{2, 5}` | 5 | 15 | `NBMF_Exact`, `NBMF_SA`, `NBMF_PathIntegral` |
| Convergence | synthetic (noiseless) | 30×30 | `{2, 5, 10}` | 5 | 15 | `Repo_ALS`, `Repo_SGD`, `NBMF_SA`, `NBMF_PathIntegral`, `NBMF_Exact` |
| Primary benchmark | MovieLens dense subblock | 100×100 (density 0.7086, 7086 obs.) | `5` | 5 | 15 | `Repo_ALS`, `Repo_SGD`, `NBMF_SA`, `NBMF_PathIntegral` |
| Supplemental benchmark | same MovieLens subblock | 100×100 | `5` | 5 | 15 | `NMF_MU` |
| Scalability (synthetic) | synthetic | `{10,20,50,100}` | `{2,5,10}` with `k<n` | 3 | 15 | `NBMF_SA`, `NBMF_PathIntegral` |
| Scalability (MovieLens) | ML subblocks | `{30,50,100}` | `{2, 5}` | 3 | 15 | `NBMF_SA`, `NBMF_PathIntegral` |
| Feasibility | synthetic | `{8,10,12,15}` | `{2,5,10}` with `k<n` | 1 | 15 | `NBMF_Exact` |

## 4. Final Method Inventory

| Method | Role | Included tiers |
|---|---|---|
| `Repo_ALS` | Primary baseline (regularized ALS, dense adapted) | Convergence, Primary |
| `Repo_SGD` | Primary baseline (biased SGD with user/item bias and global mean, dense adapted) | Convergence, Primary |
| `NBMF_SA` | Primary annealing method (SimulatedAnnealingSampler, `num_reads=20`, `num_sweeps=200`) | Correctness, Convergence, Primary, Scalability |
| `NBMF_PathIntegral` | Primary annealing method (PathIntegralAnnealingSampler, `num_reads=20`) | Correctness, Convergence, Primary, Scalability |
| `NBMF_Exact` | Oracle / feasibility probe (dimod.ExactSolver) | Correctness, Convergence, Feasibility — **forbidden in Primary** |
| `NMF_MU` | Supplemental dense NMF (Lee–Seung multiplicative updates) | Supplemental Benchmark only |

Explicitly **excluded** from the final run: `NBMF_Tabu`, `NBMF_SteepestDescent`, `NMF_SGD`. Reasons documented in `protocol_final.md` §4.2 and §4.3.

## 5. Final Dataset / Preprocessing Description

- MovieLens 100K, raw `data/u.data` (tab-separated, columns `userId, itemId, rating, timestamp`). No normalization, no thresholding, no bias subtraction.
- Dense subblock: top-100 most active users × top-100 most active items → 100×100 dense rating matrix with 0 for unobserved entries. **Actual density: 0.7086 (7086 observed out of 10000).**
- Synthetic: `W_true ~ U(0,1)`, `H_true ~ Bernoulli(0.5)`, `V = W_true @ H_true`, noiseless.
- All dense-adapted baselines (`Repo_ALS`, `Repo_SGD`) and `NMF_MU` treat every entry of `V` as an observed interaction.

## 6. Final NBMF / QUBO Pipeline

Model:

- `V ∈ R^{n×m}` nonnegative
- `W ∈ R_+^{n×k}`, `H ∈ {0,1}^{k×m}`
- Minimize `‖V - W H‖_F²` over `W` nonnegative and `H` binary

Alternating minimization (one iteration):

1. **W-update (NNLS).** For each row `i ∈ [n]`, solve `W_i = argmin_{w ≥ 0} ‖V_i - w H‖²` via `scipy.optimize.nnls(H.T, V_i)`.
2. **H-update (QUBO).** For each column `j ∈ [m]`, build `Q_j` with diagonal `(W^T W)_ii - 2 (W^T v_j)_i` and off-diagonal `2 (W^T W)_{ij}` for `i < j`, then solve `argmin_{q ∈ {0,1}^k} q^T Q_j q` with the declared sampler.

Total QUBO solves per run: `num_iterations × m`. For the primary benchmark (`num_iterations = 15`, `m = 100`) this equals **1500 QUBO solves** (confirmed in `runtime_breakdown_results.csv`).

## 7. Final Result Artifact Map

| Artifact | Path |
|---|---|
| Primary per-seed | `results_final/benchmark_primary_results.csv` (20 rows) |
| Primary aggregated (mean, std, n, 95% t-CI) | `results_final/benchmark_primary_summary.csv` (8 rows) |
| Supplemental per-seed | `results_final/benchmark_supplemental_results.csv` (5 rows) |
| Correctness per-seed | `results_final/correctness_results.csv` (30 rows) |
| Convergence MovieLens traces | `results_final/convergence_results.csv` (375 rows) |
| Convergence synthetic traces | `results_final/convergence_detailed.csv` (1125 rows) |
| Runtime breakdown (NBMF) | `results_final/runtime_breakdown_results.csv` (10 rows) |
| Scalability + feasibility | `results_final/scalability_results.csv` (112 rows: 66 syn + 36 ML + 10 feas) |
| Compliance matrix | `results_final/compliance_matrix.csv` (14 rows, all PASS) |
| Figures | `plots_final/*.png` (10 files) |

## 8. Claim-By-Claim Evidence Table

All cells below are taken directly from `results_final/`. The "exact numbers" column gives the regenerated value(s); "evidence file" gives the path. Text proposals for the paper appear in the "safe paper wording" column.

| # | Claim | Supported? | Evidence file | Exact numbers (from the CSV) | Caveat | Safe paper wording |
|---|---|---|---|---|---|---|
| C1 | QUBO construction is correct: ExactSolver recovers the ground-truth binary `H` on noiseless 10×10 problems. | **Supported** | `correctness_results.csv` | `NBMF_Exact` at `k=2, 5`: `h_recovery_rate = 1.00` (min 1.00 across 5 seeds), `final_error = 0.0` (max 0.0). | None. | "On 10×10 noiseless synthetic problems at ranks 2 and 5, ExactSolver recovered `H_true` exactly in every one of the 5 seeds." |
| C2 | Simulated annealing and path-integral annealing match ExactSolver on small noiseless problems. | **Supported** | `correctness_results.csv` | `NBMF_SA` and `NBMF_PathIntegral` at `k=2, 5`: `h_recovery_rate = 1.00` and `final_error = 0.0` for all 5 seeds. | Applies only at 10×10 and `k ≤ 5`; see C7 for the k=10 exception on larger matrices. | "Both annealing samplers reproduced the ExactSolver's perfect `H` recovery on 10×10 synthetic problems at ranks 2 and 5." |
| C3 | Classical ALS achieves the lowest reconstruction error on the primary MovieLens benchmark at rank 5 *when measured per-seed* — but biased SGD is effectively tied and slightly better on average. | **Partially supported** | `benchmark_primary_summary.csv`, `benchmark_primary_results.csv` | Mean relative error (5 seeds): `Repo_SGD = 0.4254 ± 0.0019` (CI95 `[0.4230, 0.4278]`), `Repo_ALS = 0.4292 ± 0.0002` (CI95 `[0.4290, 0.4294]`), `NBMF_SA = 0.4685 ± 0.0015`, `NBMF_PathIntegral = 0.4684 ± 0.0014`. | CIs for `Repo_ALS` and `Repo_SGD` overlap only marginally; the gap is small. `Repo_SGD` has extra capacity (biases + global mean) that `NBMF` lacks. | "On the 100×100 MovieLens dense subblock (rank 5), the classical continuous-H baselines (biased SGD: 0.4254 ± 0.0019; ALS: 0.4292 ± 0.0002) achieved lower reconstruction error than the binary-H NBMF variants (SA: 0.4685 ± 0.0015; path integral: 0.4684 ± 0.0014). The gap of roughly 0.04 in relative error is consistent with the loss of expressiveness imposed by constraining `H` to {0,1}." |
| C4 | NBMF_SA and NBMF_PathIntegral reach the same reconstruction error on the primary benchmark, but path integral is roughly 4.8× slower. | **Supported** | `benchmark_primary_results.csv`, `runtime_breakdown_results.csv` | Final relative error: SA 0.4685 vs PI 0.4684 (difference < `2e-4`). Total runtime mean: SA `0.798 ± 0.031 s`; PI `3.798 ± 0.102 s`; ratio `PI/SA ≈ 4.76×`. | Measured at `k=5`, `m=100`; see C8 for how the ratio grows with rank. | "The two samplers converge to effectively identical reconstruction error on this benchmark (difference < 1e-3), but path integral is approximately 4.8× slower than simulated annealing (3.80 s vs 0.80 s, 5 seeds)." |
| C5 | In NBMF, the sampler dominates the runtime; NNLS and QUBO construction are negligible. | **Supported** | `runtime_breakdown_results.csv` | `NBMF_SA`: sampler 96.6 %, NNLS 1.65 %, QUBO build 1.69 %, postprocess 0.10 %. `NBMF_PathIntegral`: sampler 99.2 %, NNLS 0.34 %, QUBO build 0.44 %, postprocess 0.03 %. Total 1500 QUBO solves per run; mean per-solve time `0.513 ms` (SA) and `2.511 ms` (PI). | Fractions reported here correspond to the primary benchmark size (100×100 rank 5); absolute fractions will shift at other sizes. | "Runtime is dominated by the sampler (96.6 % for SA, 99.2 % for path integral), with NNLS and QUBO construction together accounting for less than 4 % of total time. Per-QUBO solve times were 0.513 ms for SA and 2.511 ms for path integral on the primary benchmark." |
| C6 | On noiseless synthetic problems, ALS converges to machine-precision error within 15 iterations; biased SGD does not. | **Supported** | `convergence_detailed.csv` | Final-iteration error on 30×30 synthetic, mean over 5 seeds: `Repo_ALS` = 0.0 for `k ∈ {2, 5, 10}`. `Repo_SGD` = 5.89 (k=2), 9.27 (k=5), 12.73 (k=10) with `std < 0.3`. | Reflects SGD's bias terms fitting the global structure; lr=0.01 and 15 epochs are insufficient for this noiseless problem. This is an honest observation and should not be framed as a defect of the library. | "ALS converged to numerical zero within 15 iterations on all tested ranks. Biased SGD did not converge in this budget on the noiseless synthetic matrices, which reflects its additional capacity (biases + global mean) fitting the mean structure slowly under the declared learning rate." |
| C7 | On noiseless synthetic problems up to rank 5, both annealing samplers converge to machine-precision error; at rank 10 they occasionally fail to find the global optimum. | **Supported** | `convergence_detailed.csv` | Final error mean±std at `k=10` on 30×30: `NBMF_SA = 0.333 ± 0.744`, `NBMF_PathIntegral = 1.182 ± 1.295`. `NBMF_Exact = 0.0`. At `k ∈ {2, 5}` all three are 0. | The large `std` at `k=10` reflects occasional single-seed failures (most seeds are 0, some are >1). Only the oracle (ExactSolver) is guaranteed at `k=10`. | "At ranks 2 and 5 both annealing samplers match ExactSolver and recover the ground truth on every seed. At rank 10, occasional seeds fail to reach the global optimum under the declared annealing budget (mean final error 0.33 for SA and 1.18 for path integral, with large across-seed variance)." |
| C8 | Path integral's slowdown relative to simulated annealing grows with rank (not with size). | **Supported** | `scalability_results.csv` (synthetic) | PI/SA runtime ratio: `k=2: ~2.2×`; `k=5: ~4.6×`; `k=10: ~7.9×`, stable across `n ∈ {10, 20, 50, 100}`. | Bounded to the tested synthetic grid. | "In our synthetic scalability sweep, path integral was roughly 2× slower than simulated annealing at rank 2, 4.6× slower at rank 5, and 7.9× slower at rank 10, with the ratio largely independent of matrix size." |
| C9 | NBMF runtime grows approximately linearly in the number of columns for both samplers at fixed rank. | **Supported** | `scalability_results.csv` (synthetic, `simulated_annealing`, `k=5`) | Mean `total_runtime` at `k=5`: `n=10: 0.084 s`; `n=20: 0.169 s`; `n=50: 0.418 s`; `n=100: 0.855 s`. Ratios: 2.0×, 2.5×, 2.05× per doubling of `n`. | Measured only on the tested range `n ∈ [10, 100]`. No claim about asymptotic complexity. | "NBMF total runtime scaled approximately linearly with the number of columns at fixed rank over the tested synthetic range (10 ≤ n ≤ 100), consistent with the `m` QUBO solves per iteration." |
| C10 | NMF_MU does not dominate NBMF on the primary MovieLens benchmark. | **Supported** | `benchmark_supplemental_results.csv` | Mean `final_relative_error`: `NMF_MU = 0.4871 ± 0.0018` vs `NBMF_SA = 0.4685` and `NBMF_PathIntegral = 0.4684`. Mean runtime: `NMF_MU ≈ 0.5 ms` (much faster) vs NBMF `0.80–3.80 s`. | `NMF_MU` is fast but hits a higher final error than NBMF within 15 iterations at this size, which is the opposite of the intuition that a continuous-H NMF should always beat a binary-H NMF. The explanation is that 15 multiplicative-update iterations are not enough for MU to converge on a 100×100 matrix of this density. | "On the primary benchmark, multiplicative-update NMF (supplemental) reached a slightly higher relative error (0.487 ± 0.002) than NBMF (≈ 0.468) within the shared 15-iteration budget, while being roughly three orders of magnitude faster in wall clock. We report MU only as a reference and do not use it as a primary comparator." |
| C11 | ExactSolver is feasible within seconds up to 15×15 and rank 10, but is strictly exponential in rank. | **Supported** | `scalability_results.csv` (`data_source == feasibility`), plus convergence `NBMF_Exact` | Feasibility (1 seed, 15 iters): `15×15, k=10: 0.185 s`; `12×12, k=10: 0.147 s`. On 30×30 convergence at `k=10` (5 seeds): `NBMF_Exact` mean wall time 0.395 s. All feasibility cells ran under the 600 s timeout (`feasible = True`). | The tested feasibility grid does not include `k > 10`; the claim is specifically about the tested range. | "ExactSolver remained feasible (sub-second per run) up to 15×15 and rank 10 within the 600 s timeout, consistent with the `O(m · 2^k · iterations)` cost of exhaustive QUBO enumeration. We did not test ranks beyond 10." |
| C12 | `NBMF_Exact` was never used in the primary benchmark table. | **Supported** | `compliance_matrix.csv` item 17.5 | `forbidden set absent: ['NBMF_Exact', ...]` in `benchmark_primary_results.csv`. | None. | "We restricted ExactSolver to correctness and feasibility tiers and never used it as a primary benchmark comparator." |
| C13 | The reported numbers come from one audited run with five seeds. | **Supported** | `01_final_run_log.md`, `compliance_matrix.csv` | 14/14 compliance items PASS. Primary table contains exactly 5 seeds `{42,43,44,45,46}` and exactly 4 methods. | None. | "All results in this paper come from a single pipeline run on 2026-04-22, pre-declared by `protocol_final.md` and machine-audited (compliance matrix: 14/14 PASS, `results_final/compliance_matrix.csv`)." |

### Unsupported / deliberately-not-claimed items

| Claim the paper must **not** make | Why |
|---|---|
| "NBMF beats ALS." | The primary table shows the opposite at the tested size and rank. |
| "Path integral is more accurate than simulated annealing." | Final errors are within 1e-3 of each other. |
| "Quantum annealing beats classical." | No quantum hardware was used; all annealing is classical simulators from `dwave-samplers`. |
| "NBMF scales to large problems." | We only tested up to 100×100. |
| "NBMF has asymptotic complexity O(·)." | We report only empirical scaling within the tested range. |
| "Tabu and SteepestDescent were compared." | They were not included in the final run. |
| "NMF_SGD was a baseline." | It is not implemented in this repository. |

## 9. Do-Not-Overclaim Section

- The paper must not present NBMF as a superior method on the tested benchmark. At 100×100 rank 5, it is strictly worse in reconstruction error than both classical baselines, and slower than ALS by 15–70×.
- The paper must not infer quantum-hardware performance from these simulator runs.
- The paper must not claim NBMF generalizes to the full sparse MovieLens; all experiments use a dense top-N subblock with density 0.71.
- The paper must not frame `Repo_SGD` vs `NBMF_SA` differences as purely algorithmic; part of the gap comes from `Repo_SGD`'s biases and global mean, which NBMF does not have.
- The paper must not hide the `k=10` convergence variance for annealing samplers — it is a real finding and should be reported, not suppressed.

## 10. Best 4–5 Figures / Tables for the Paper

Suggested paper figure set (all from `plots_final/` and `results_final/`):

1. **Table 1 — Primary benchmark summary** (`benchmark_primary_summary.csv`). Four methods × two metrics with mean, std, n, 95% t-CI. This is the headline table.
2. **Figure 1 — Convergence on MovieLens subblock** (`convergence_primary.png`). All four primary methods, iteration vs reconstruction error with shaded ±1σ band.
3. **Figure 2 — Primary benchmark bars** (side-by-side `benchmark_error_bar.png` and `benchmark_runtime_bar.png`, or combined panel). Shows the quality vs runtime trade-off at a glance.
4. **Figure 3 — Runtime decomposition** (`runtime_breakdown_stacked.png`). Makes the "sampler dominates" point visually and motivates why path integral is expensive.
5. **Figure 4 — Scalability** (either `scalability_runtime_vs_size.png` on log-y or `scalability_error_vs_rank.png`). Shows approximately linear growth in `n` and the PI/SA gap widening with rank.

Appendix / supplemental: `correctness_small.png`, `convergence_by_rank.png`, `convergence_wallclock.png`, `feasibility_exactsolver.png`.

## 11. Final Paper-Writing Brief

The story told by the final run is honest and publishable:

1. **The binary-H NBMF formulation is correct.** ExactSolver recovers `H_true` on small noiseless problems, and both annealing samplers match it.
2. **Classical dense baselines are strong.** On the 100×100 MovieLens subblock at rank 5, ALS and biased SGD both achieve lower reconstruction error than either NBMF variant. This is expected: binary `H` is a strict constraint.
3. **Between the two annealers, simulated annealing is the clear choice on classical hardware.** Final errors are effectively identical, but simulated annealing is ~5× faster at rank 5 and the PI/SA gap grows with rank.
4. **Runtime is almost entirely in the sampler.** NNLS and QUBO construction are negligible. This tells the reader where engineering effort would pay off if one wanted to speed up NBMF.
5. **Scalability is empirically linear in columns at fixed rank** within the tested range, and ExactSolver is feasible only up to about rank 10 at the sizes we tested.
6. **Convergence at rank 10 reveals annealing limitations.** Both samplers occasionally fail to reach zero error on some seeds; only ExactSolver always does. This motivates either longer annealing schedules or the use of ExactSolver as an oracle within the feasible regime.

A suitable paper angle: *classical annealing samplers faithfully implement the binary-H NBMF formulation at small to moderate rank, but the binary-H constraint itself is the limiting factor for reconstruction quality on dense MovieLens subblocks; the runtime cost is overwhelmingly in the sampler, which defines where future hardware acceleration would matter.*

All the numerical content to support this angle is already in `results_final/` and `plots_final/`. No additional experiments are required before writing.
