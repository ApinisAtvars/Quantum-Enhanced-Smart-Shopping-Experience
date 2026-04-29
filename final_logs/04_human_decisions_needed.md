# Human Decisions Needed

This file lists the decisions that **genuinely require a human call** before paper submission. Everything else in this project has been resolved from evidence and is settled in `protocol_final.md` and `final_logs/03_final_claims_dossier.md`.

Each item has:
- the decision,
- the options,
- the evidence that informs the choice,
- a default recommendation.

---

## 1. Include `NMF_MU` (supplemental) in the main paper or keep it appendix-only?

**Options**

- (a) Main paper, in a footnote next to the primary benchmark table.
- (b) Appendix only.
- (c) Drop entirely.

**Evidence.** `benchmark_supplemental_results.csv` shows `NMF_MU` at `0.4871 ± 0.0018` relative error on the 100×100 MovieLens subblock — *higher* than both NBMF variants (≈ 0.468) at the same 15-iteration budget. Runtime is ≈ 0.5 ms vs ≈ 0.8–3.8 s for NBMF. The counter-intuitive result (continuous `H` losing to binary `H`) is an artifact of the 15-iteration budget being too short for the multiplicative updates to converge.

**Default recommendation.** (b) Appendix only, with one sentence stating that the 15-iteration budget is not enough for MU to converge and that a longer MU run outperforms NBMF. This avoids misleading the reader while keeping the supplemental comparison honest.

## 2. Report the historical reduced-run numbers at all?

**Options**

- (a) Mention in a reproducibility/acknowledgments note that earlier reduced-scale runs (50×50, 3 seeds) exist in `archive/` and explain the jump to the final 100×100 / 5-seed setting.
- (b) Do not mention them.

**Evidence.** The archive (`archive/finalization_backup_20260422T232612Z/legacy_run_20260407_163135/` and `advancement_logs_prev/`) contains results with different matrix sizes, different seed counts, and different density. Some of them even reach different conclusions (e.g., the 50×50 result showed `ALS` with far better error than anyone else at 0.3209; the 100×100 run flips that to `SGD` being best). These runs are not part of the final evidence.

**Default recommendation.** (b) Do not mention them in the paper body. The archive exists for auditability and for anyone who wants to inspect the full iteration history of the project, but the paper should stand on the single final run.

## 3. Put the ExactSolver feasibility figure in the main paper or appendix?

**Options**

- (a) Main paper (as part of the scalability discussion).
- (b) Appendix.
- (c) Drop it.

**Evidence.** `plots_final/feasibility_exactsolver.png` shows the `O(m · 2^k · iterations)` runtime growth on log-scale up to `15×15`, rank 10 (0.185 s). All feasibility cells ran under the 600 s timeout. The narrative value is that it bounds where ExactSolver is a valid oracle — an important but narrow point.

**Default recommendation.** (b) Appendix. The main paper can reference it in one sentence in the scalability section.

## 4. Do we want to add a longer-budget NMF_MU sanity check before submission?

**Options**

- (a) Leave the 15-iteration `NMF_MU` result as-is and explain it in the paper.
- (b) Rerun just the supplemental benchmark at a longer iteration budget (e.g., 200) to show that MU eventually catches up, then keep the 15-iteration number as the apples-to-apples point and the 200-iteration number as a sanity footnote.

**Evidence.** The protocol fixes iterations at 15 for apples-to-apples reasons (§10). A supplementary relaxation for MU would violate the apples-to-apples rule and would require a Section-21 amendment.

**Default recommendation.** (a). The current result is honest and the caveat is easy to write.

## 5. Figure set for the paper

**Options**

- (a) The 5-figure set recommended in `03_final_claims_dossier.md` §10.
- (b) A reduced 3-figure set: just Table 1 + `convergence_primary.png` + `runtime_breakdown_stacked.png`.
- (c) A different selection.

**Evidence.** All 10 figures in `plots_final/` were generated from the same audited run. There is no technical constraint on which to pick.

**Default recommendation.** (a) for a full paper, (b) for a short paper or poster.

---

## Items that **do not** need a human decision

These were fully resolved from evidence during finalization and are recorded in `protocol_final.md`:

- Whether to include `NBMF_Tabu` / `NBMF_SteepestDescent` → **excluded** (`protocol_final.md` §4.2).
- Whether to include `NMF_SGD` → **excluded** because it is not implemented (§4.3).
- Whether the primary benchmark rank is 5 → **yes** (§9).
- Whether to use 5 seeds → **yes** (§7), with 3 seeds for scalability and 1 for feasibility.
- Whether to put `NBMF_Exact` in the primary table → **no** (§3 rule 3).
- Whether the timeout is 600 s → **yes** (§12/§13), enforced in code (`experiments/scalability.py`) and re-checked by the compliance matrix.
- Whether to regenerate plots from the final CSVs only → **yes** (§15).
- Whether historical summaries count as evidence → **no** (§2 source-of-truth statement).
