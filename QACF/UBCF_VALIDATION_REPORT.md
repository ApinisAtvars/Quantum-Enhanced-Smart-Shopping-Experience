# UBCF Validation Report

**Date:** 2026-03-10  
**Source notebook:** `05_validation_ubcf.ipynb`  
**Artifacts reviewed:**
- `data/processed_32m_ubcf/ubcf_metrics.csv`
- `data/processed_32m_ubcf/hybrid_ubcf_stress.csv`
- `data/processed_32m_ubcf/research_addendum_ubcf.csv`
- `data/processed_32m_ubcf/privacy_log_ubcf.csv`
- `data/processed_32m_ubcf/runtime_complexity_ubcf.csv`

## 1) Executive Summary

The UBCF pipeline shows **strong baseline regression metrics** but **weak ranking quality** and **rapid degradation under extreme sparsity**. 

Key outcomes:
- Baseline UBCF: RMSE = **0.8296**, MAE = **0.6084**, but HR@10 = **0.004** and NDCG@10 = **0.00153**.
- In stress-routing experiments, ranking starts reasonable at low drop (HR@10 = **0.16** at 0% drop) but falls sharply at high drop (HR@10 = **0.02** at 95% drop).
- Bootstrap on sparse regime (drop ≥ 80%) confirms low expected ranking quality:
  - HR@10 mean **0.0511** (95% CI: **0.0344–0.0689**)
  - NDCG@10 mean **0.0254** (95% CI: **0.0131–0.0340**)
- Shot-noise simulation at 1024 shots causes negligible change in average kernel similarity.
- IQP-style transform produces very high average similarity (0.9218), suggesting **possible kernel saturation / reduced discriminability**.

## 2) Quantitative Findings

## 2.1 Baseline UBCF metrics

From `ubcf_metrics.csv`:
- RMSE: **0.829645**
- MAE: **0.608413**
- HR@10: **0.0040**
- NDCG@10: **0.001528**
- Precision@10: **0.0004**
- `k_clusters`: **16**

Interpretation: the model predicts ratings with moderate accuracy, but recommendation ranking quality is very low in this configuration.

## 2.2 Stress test vs sparsity drop

From `hybrid_ubcf_stress.csv`:
- 0% drop: HR@10 **0.1600**, NDCG@10 **0.1154**
- 80% drop: HR@10 **0.0633**, NDCG@10 **0.0305**
- 90% drop: HR@10 **0.0700**, NDCG@10 **0.0358**
- 95% drop: HR@10 **0.0200**, NDCG@10 **0.00988**

Relative retention from 0% → 95% drop:
- HR@10 retention: **12.5%** (87.5% loss)
- NDCG@10 retention: **8.6%** (91.4% loss)

Interpretation: ranking is not robust to severe sparsity, with most utility lost by 95% drop.

## 2.3 Routing behavior under stress

Routing proportions shift as sparsity increases:
- At 0% drop: classical route **97.3%**, quantum sparse **2.7%**, cold-content **0%**
- At 80% drop: classical **38.3%**, quantum sparse **57.7%**, cold-content **4.0%**
- At 95% drop: classical **10.0%**, quantum sparse **33.7%**, cold-content **56.3%**

Interpretation: at high sparsity, the system increasingly relies on fallback/cold routing paths, consistent with the drop in ranking quality.

## 2.4 Bootstrap confidence intervals (sparse regime)

From `research_addendum_ubcf.csv` (100 bootstrap iterations, sparse subset):
- HR@10: mean **0.05111**, CI **[0.03444, 0.06894]**
- NDCG@10: mean **0.02540**, CI **[0.01315, 0.03404]**

Interpretation: even with uncertainty accounted for, sparse-regime ranking remains low.

## 2.5 Kernel stress variants

From `research_addendum_ubcf.csv`:
- Original mean similarity: **0.052761** (std 0.107001)
- Shot noise 1024 mean similarity: **0.052747** (std 0.107170)
- IQP variant mean similarity: **0.921786** (std 0.188899)

Interpretation:
- Shot noise effect is minimal at this scale.
- IQP transform drastically raises average similarity; this can compress pairwise contrast and make ranking harder unless re-normalization or calibration is added.

## 2.6 Complexity/privacy accounting

From `privacy_log_ubcf.csv` and `research_addendum_ubcf.csv`:
- Queries per Lloyd iteration: **3006** (n=500, k=6)
- Iterations logged: **10**
- Total queries: **30,060**
- Per-iteration elapsed times are very small (~2e-5 to 7.45e-5 seconds in this micro-benchmark).

From `runtime_complexity_ubcf.csv`:
- Runtime rises from ~0.000234 sec (n=50) to a local peak ~0.000401 sec (n=200), then fluctuates.

Interpretation: observed runtime is broadly consistent with expected growth trends but noisy at this tiny timing scale.

## 3) Assessment

## Strengths
- Validation includes bootstrap uncertainty, stress routing analysis, and complexity/privacy logs.
- Sparsity stress behavior is transparent and reproducible through CSV outputs.
- Quantum shot-noise sensitivity is explicitly tested.

## Risks / Weaknesses
- Ranking metrics remain low in sparse regimes and can collapse at high drop.
- High reliance on cold-content fallback at 95% drop indicates recommendation quality may be driven by fallback heuristics rather than collaborative signal.
- IQP-style stress transformation appears to over-smooth kernel similarity and may require redesign before deployment use.

## 4) Recommendations

1. **Prioritize ranking optimization** for sparse regimes (target HR@10 and NDCG@10), not only RMSE/MAE.
2. **Calibrate fallback routing** to reduce excessive cold-content routing at very high sparsity.
3. **Rework IQP stress mapping** with post-transform normalization/temperature scaling and compare discriminability metrics (e.g., similarity histogram spread, nearest-neighbor margin).
4. **Increase bootstrap iterations** (e.g., 500+) for tighter CIs and more stable inference.
5. **Add segment-level evaluation** (cold users/items, heavy/light raters) to locate where ranking fails most.

## 5) Bottom Line

The current UBCF validation shows a pipeline that is analytically well-instrumented but ranking-fragile under extreme sparsity. The strongest immediate gain is likely to come from improving sparse-regime ranking and controlling fallback dominance, while keeping the existing logging/CI framework for rigorous tracking.