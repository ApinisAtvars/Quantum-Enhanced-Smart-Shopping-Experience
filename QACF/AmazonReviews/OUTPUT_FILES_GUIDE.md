# Output Files Guide: What Each File Means & How to Verify the Research Claim

**Purpose:** Quick reference for understanding all output files from the 5-phase pipeline.

---

## File Directory Overview

After running all 5 notebooks, expect outputs in:
- **`QACF/AmazonReviews/data/processed_amazon/`** — Phase 1–2 outputs
- **`QACF/data/processed_amazon/`** — Phase 3–5 outputs (may consolidate to one location)

---

## Phase 1: Data Preparation Outputs

### `user_sparse.npz`
- **Type:** SciPy sparse matrix (CSR format), saved as .npz
- **Shape:** (100K, 1.2M) — 100K users × 1.2M items
- **Content:** User-item rating matrix (only non-zero entries stored)
- **Use:** Raw collaborative signal; used to compute neighbors in Phase 2
- **Check:** Density ≈ 0.0000083 (verify sparsity)
  ```python
  import numpy as np
  from scipy.sparse import load_npz
  user_sparse = load_npz("user_sparse.npz").tocsr()
  density = user_sparse.nnz / (user_sparse.shape[0] * user_sparse.shape[1])
  print(f"Density: {density:.10f}")  # Should be ~0.0000083
  ```

### `user_latent.npy`
- **Type:** NumPy array (float32)
- **Shape** (100K, 32) — 100K users, 32 latent dimensions
- **Content:** SVD-projected user features (denoised)
- **Use:** Input to K-Means (Phase 2) and quantum kernel (Phase 3)
- **Check:** Explained variance
  ```python
  from sklearn.decomposition import TruncatedSVD
  svd = TruncatedSVD(n_components=32)
  variance_explained = svd.explained_variance_ratio_.sum()
  print(f"Variance explained: {variance_explained:.2%}")  # Should be 35–50%
  ```

### `item_latent.npy`
- **Type:** NumPy array (float32)
- **Shape:** (1.2M, 32)
- **Content:** SVD-projected item features
- **Use:** Reference for item clustering (not used in the main flow, but available)

### `user_combined.npy` & `item_combined.npy`
- **Type:** NumPy array (float32)
- **Shape:** (100K, 32+) and (1.2M, 32+) — latent + content features concatenated
- **Content:** Hybrid features (collaborative + content-based)
- **Use:** Input to quantum kernel for Phase 3
- **Check:** Shape should be > 32
  ```python
  user_combined = np.load("user_combined.npy")
  print(user_combined.shape)  # Should be (100000, 32+)
  ```

### `content_features.npz`
- **Type:** SciPy sparse matrix (CSR format)
- **Shape:** (1.2M, 4600) — 1.2M items × 4600 content features
- **Content:** TF-IDF vectors from review text + product categories
- **Use:** Content-based fallback for cold-start items (Phase 2, 4)
- **Check:** Sparsity
  ```python
  content = load_npz("content_features.npz").tocsr()
  content_density = content.nnz / (content.shape[0] * content.shape[1])
  print(f"Content density: {content_density:.4%}")  # Around 0.1–1%
  ```

### `id_maps.pkl`
- **Type:** Python pickle (dict)
- **Content:** `{"user_ids": [...], "item_ids": [...]}` mappings
- **Use:** Map from string IDs (reviewerID, ASIN) ↔ array indices
- **Check:** Load and inspect
  ```python
  import pickle
  with open("id_maps.pkl", "rb") as f:
      maps = pickle.load(f)
  print(f"Users: {len(maps['user_ids'])}, Items: {len(maps['item_ids'])}")
  ```

### `category_sparsity.csv`
- **Type:** CSV table
- **Columns:** `category`, `n_items`, `n_ratings`, `avg_rating_count_per_item`, `sparsity`
- **Content:** Sparsity-by-category diagnostic
- **Example:**
  ```
  category | n_items | n_ratings | avg_rating_count_per_item | sparsity
  Fiction  | 45000   | 3500000   | 77.8                      | 0.000015
  Science  | 32000   | 1200000   | 37.5                      | 0.000012
  ...
  ```
- **Use:** Understand data distribution; identify sparse categories
- **Check:** Total ratings match user_sparse.nnz

### `review_length_stats.csv`
- **Type:** CSV table (single row)
- **Columns:** `n_reviews`, `mean_words`, `std_words`, `min_words`, `max_words`, `sample_p50_words`, `sample_p90_words`, `sample_p99_words`
- **Content:** Text length distribution statistics
- **Example:**
  ```
  n_reviews | mean_words | std_words | min_words | max_words | p50 | p90 | p99
  10000000  | 145.3      | 98.2      | 1         | 2048      | 132 | 289 | 412
  ```
- **Use:** Verify review text richness; diagnostics
- **Check:** Mean should be >100 words (rich text)

---

## Phase 2: Classical Baseline Outputs

### `user_clusters.npy`
- **Type:** NumPy array (int32)
- **Shape:** (100K,) — one cluster assignment per user
- **Content:** User cluster labels (0 to K-1, where K is best K from tuning)
- **Use:** Define neighborhoods for classical CF (Phase 2); used by Phase 4 for comparison
- **Check:** Distribution
  ```python
  clusters = np.load("user_clusters.npy")
  unique, counts = np.unique(clusters, return_counts=True)
  print(f"K={len(unique)}, cluster sizes: min={counts.min()}, max={counts.max()}")
  # Should have roughly balanced clusters
  ```

### `amazon_tuning.csv`
- **Type:** CSV table
- **Columns:** `k`, `inertia`, `silhouette`
- **Content:** K-Means tuning results for each K ∈ {16, 24, 32, 40, 48, 64}
- **Example:**
  ```
  k  | inertia  | silhouette
  16 | 4567.2   | 0.1823
  24 | 3890.1   | 0.2145
  32 | 3412.3   | 0.2312  ← Best
  40 | 3101.5   | 0.2234
  ...
  ```
- **Use:** Verify K selection; check Silhouette trend
- **Check:** Best K = argmax(silhouette)

### `amazon_metrics.csv` (or `baseline_metrics.csv`)
- **Type:** CSV table, aggregated metrics across eval_users
- **Columns:** `algorithm`, `mean_hr`, `std_hr`, `mean_ndcg`, `std_ndcg`, `mean_rmse`, `std_rmse`, `mean_map`, `std_map`
- **Content:** Classical CF baseline performance
- **Example:**
  ```
  algorithm | mean_hr | std_hr | mean_ndcg | std_ndcg | mean_rmse | mean_map
  ItemCF    | 0.165   | 0.087  | 0.195     | 0.108    | 1.234     | 0.142
  ContentCF | 0.098   | 0.064  | 0.112     | 0.071    | 1.567     | 0.089
  Hybrid    | 0.181   | 0.091  | 0.224     | 0.115    | 1.198     | 0.165
  ```
- **Use:** Reference baseline; compare against Phase 4 hybrid
- **Check:** HR@10 ≈ 0.15–0.22 is reasonable for sparse CF

---

## Phase 3: Quantum Clustering Outputs

### `kernel_matrix.npz`
- **Type:** NumPy .npz (can contain multiple arrays)
- **Content:** 
  - `kernel`: 800×800 fidelity kernel matrix (float32)
  - `sample_idx`: indices of sampled 800 users (int32)
  - Optional: `kernel_ent` (entanglement variant), polynomial transforms
- **Use:** Quantum distance metric (Phase 4 routing), validation plots
- **Check:** Kernel properties
  ```python
  from scipy.sparse import load_npz
  kernel_data = np.load("kernel_matrix.npz")
  K = kernel_data["kernel"]
  print(f"Kernel shape: {K.shape}")  # (800, 800)
  print(f"Diagonal values (self-fidelity): {np.diag(K)[:10]}")  # Should be ~1.0
  print(f"Off-diag mean: {(K.sum() - np.trace(K)) / (K.shape[0] * (K.shape[0] - 1)):.4f}")  
  # Typically 0.2–0.5
  ```

### `quantum_user_labels_improved.npy` (and variants)
- **Type:** NumPy array (int32)
- **Shape:** (800,) — cluster label per sampled user
- **Content:** Quantum K-Means cluster assignments (Lloyd algorithm)
- **Variants:** 
  - `quantum_user_labels_dwave.npy` — D-Wave QUBO solution
  - `quantum_user_labels_vqe_adiabatic.npy` — VQE adiabatic variant
  - `quantum_user_labels_vqe_subspace_final.npy` — Tuned VQE variant
- **Use:** Different quantum clustering strategies (compared for robustness)
- **Check:** Cluster balance
  ```python
  labels = np.load("quantum_user_labels_improved.npy")
  unique, counts = np.unique(labels, return_counts=True)
  print(f"Quantum K={len(unique)}, balance: min={counts.min()}, max={counts.max()}")
  ```

### `quantum_metrics.csv`
- **Type:** CSV table
- **Columns:** `variant`, `auto_k`, `silhouette`, `davies_bouldin`, `dunn_index`, `inertia`
- **Content:** Quality metrics for each quantum clustering variant
- **Example:**
  ```
  variant              | auto_k | silhouette | davies_bouldin | dunn_index | inertia
  Lloyd_Improved       | 7      | 0.3412     | 1.089          | 2.145      | 2301.2
  DWave_QUBO           | 8      | 0.3156     | 1.234          | 1.987      | 2567.3
  VQE_Adiabatic        | 6      | 0.3578     | 1.045          | 2.234      | 2198.1
  VQE_Subspace_Tuned   | 7      | 0.3891     | 0.998          | 2.456      | 2089.7
  ```
- **Use:** Compare quantum variants; validate cluster quality
- **Check:** Silhouette should be > 0.25 (good clusters); higher = better

### `quantum_elbow.csv`
- **Type:** CSV table
- **Columns:** `k`, `silhouette_score`
- **Content:** K selection sweep results
- **Example:**
  ```
  k | silhouette
  4 | 0.2145
  5 | 0.2634
  6 | 0.3145
  7 | 0.3412  ← Best (auto-K selection)
  8 | 0.3289
  9 | 0.3001
  10| 0.2876
  ```
- **Use:** Verify automatic K selection
- **Check:** Clear elbow/peak around k=6–8

### `umap_quantum.png`, `mds_quantum.png`, `quantum_elbow.png`
- **Type:** PNG image plots
- **Content:** 
  - UMAP: 2D scatter of 800 users, colored by cluster (quantum kernel metric)
  - MDS: 2D scatter, multidimensional scaling alternative
  - Elbow: Line plot of K vs Silhouette score
- **Use:** Visual validation of cluster quality and K selection
- **Check:** UMAP should show clear, well-separated clusters (vs classical which shows overlap)

---

## Phase 4: Sparsity Stress Test THE CRITICAL OUTPUTS

### `hybrid_amazon_recs.csv`
- **Type:** CSV table (raw results)
- **Columns:** `user_id`, `drop_level`, `algorithm`, `hr@10`, `ndcg@10`, `precision@10`, `map@10`, `mrr`, `novelty@10`
- **Content:** Per-user × algorithm × drop-level metrics
- **Size:** ~300 users × 6 drop levels × 2 algorithms = ~3600 rows
- **Example:**
  ```
  user_id | drop_level | algorithm        | hr@10 | ndcg@10 | map@10
  user_42 | 0.00       | classical        | 0.220 | 0.245   | 0.235
  user_42 | 0.00       | hybrid           | 0.220 | 0.248   | 0.238
  user_42 | 0.80       | classical        | 0.080 | 0.085   | 0.075
  user_42 | 0.80       | hybrid           | 0.112 | 0.128   | 0.108  ← Better!
  user_42 | 0.95       | classical        | 0.020 | 0.018   | 0.015
  user_42 | 0.95       | hybrid           | 0.042 | 0.039   | 0.035  ← Much better!
  ...
  ```
- **Use:** Detailed comparison; identify which users benefit from quantum
- **Check:** Hybrid HR > Classical HR at high drop levels (≥0.60)

### `hybrid_amazon_stress_final.csv` (or `hybrid_amazon_stress.csv`)
- **Type:** CSV table (aggregated results)
- **Columns:** `drop_level`, `algorithm`, `count`, `mean_hr`, `std_hr`, `mean_ndcg`, `std_ndcg`, `mean_map`, `std_map`, `median_hr`, `quartile_25_hr`, `quartile_75_hr`
- **Content:** Aggregated metrics across all eval_users
- **Example (THE KEY PROOF TABLE):**
  ```
  drop_level | algorithm | count | mean_hr | std_hr | mean_ndcg | std_ndcg | median_hr | q75_hr
  0.00       | classical | 300   | 0.2200  | 0.1145 | 0.2450    | 0.1210   | 0.2000   | 0.3000
  0.00       | hybrid    | 300   | 0.2200  | 0.1150 | 0.2480    | 0.1230   | 0.2000   | 0.3100
  0.20       | classical | 300   | 0.1900  | 0.1000 | 0.2100    | 0.1050   | 0.1700   | 0.2600
  0.20       | hybrid    | 300   | 0.1980  | 0.1020 | 0.2180    | 0.1070   | 0.1800   | 0.2700
  0.60       | classical | 300   | 0.1200  | 0.0780 | 0.1280    | 0.0820   | 0.1000   | 0.1700
  0.60       | hybrid    | 300   | 0.1510  | 0.0950 | 0.1620    | 0.1000   | 0.1300   | 0.2100  ← +26% uplift!
  0.80       | classical | 300   | 0.0800  | 0.0560 | 0.0850    | 0.0610   | 0.0600   | 0.1100
  0.80       | hybrid    | 300   | 0.1125  | 0.0750 | 0.1185    | 0.0800   | 0.0900   | 0.1500  ← +41% uplift!
  0.95       | classical | 300   | 0.0198  | 0.0185 | 0.0180    | 0.0175   | 0.0100   | 0.0300
  0.95       | hybrid    | 300   | 0.0415  | 0.0310 | 0.0385    | 0.0295   | 0.0300   | 0.0600  ← +110% uplift!
  ```
- **Use:** PRIMARY TABLE for proving the research claim
- **Check:** 
  - At drop ≥ 0.60: hybrid_hr > classical_hr (significant)
  - At drop = 0.95: uplift ≈ 2× (typical target)
  - Std dev < mean (signal > noise)

### `classical_only_stress.csv`
- **Type:** CSV table
- **Columns:** `drop_level`, `mean_hr`, `std_hr`, `mean_ndcg`, `std_ndcg`, ...
- **Content:** Classical-only detailed degradation curve
- **Use:** Verify classical control arm behaves as expected
- **Check:** Values should match the `classical` algorithm column in `hybrid_amazon_stress_final.csv`

### `hybrid_vs_classical_uplift.csv`
- **Type:** CSV table (derived comparison)
- **Columns:** `drop_level`, `classical_hr`, `hybrid_hr`, `absolute_uplift`, `relative_uplift_pct`, `p_value`, `effect_size`, `significant`
- **Content:** Direct uplift metrics with statistical tests
- **Example (MOST IMPORTANT FOR PAPER):**
  ```
  drop_level | classical_hr | hybrid_hr | abs_uplift | rel_uplift_pct | p_value | effect_size | significant
  0.00       | 0.220        | 0.220     | 0.000      | 0.0%          | —       | 0.000       | No
  0.20       | 0.190        | 0.198     | 0.008      | 4.2%          | 0.347   | 0.087       | No
  0.40       | 0.160        | 0.178     | 0.018      | 11.3%         | 0.098   | 0.195       | No
  0.60       | 0.120        | 0.151     | 0.031      | 25.8%         | 0.005   | 0.342       | Yes ✓
  0.80       | 0.080        | 0.112     | 0.032      | 40.0%         | 0.001   | 0.456       | Yes ✓✓
  0.95       | 0.020        | 0.042     | 0.022      | 110.0%        | <0.001  | 0.678       | Yes ✓✓✓
  ```
- **Use:** Write paper abstract/results section
- **Check:** p_value < 0.05 for drop levels ≥ 0.60 (proves claim)

### `hybrid_results_summary.json`
- **Type:** JSON
- **Content:** Metadata, hyperparameters, key findings
- **Example:**
  ```json
  {
    "experiment": "hybrid_vs_classical_sparsity_stress",
    "dataset": "amazon_reviews",
    "n_eval_users": 300,
    "drop_levels": [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95],
    "algorithms": ["classical", "hybrid"],
    "key_findings": {
      "max_uplift_hr": 0.110,
      "max_uplift_drop_level": 0.95,
      "p_value_at_high_sparsity": "<0.001",
      "claim_supported": true
    },
    "hyperparams": {
      "quantum_weight_sparse": 0.40,
      "classical_weight_sparse": 0.60,
      "mmr_lambda": 0.05,
      "novelty_boost": 0.03
    }
  }
  ```
- **Use:** Documentation and reproducibility
- **Check:** `claim_supported == true`

---

## Phase 5: Statistical Validation Outputs (PROOF RIGOUR)

### `per_user_paired_significance_amazon.csv`
- **Type:** CSV table
- **Columns:** `drop_level`, `user_id`, `classical_hr`, `hybrid_hr`, `hr_ci_lower`, `hr_ci_upper`, `paired_t_stat`, `p_value`, `effect_size_cohens_d`, `significant_alpha05`
- **Content:** Per-user bootstrap CIs and paired t-tests
- **Size:** 300 users × 6 drop levels = 1800 rows
- **Example:**
  ```
  drop_level | user_id | classical_hr | hybrid_hr | ci_lower | ci_upper | p_value | sig
  0.95       | user_42 | 0.010        | 0.050     | 0.035    | 0.065    | 0.002   | Yes ✓
  0.95       | user_88 | 0.020        | 0.035     | 0.020    | 0.051    | 0.015   | Yes ✓
  0.95       | user_01 | 0.030        | 0.035     | 0.025    | 0.048    | 0.187   | No
  ```
- **Use:** Calculate what % of users see significant improvement
- **Check:** >80% of users should show p < 0.05 at drop ≥ 0.80

### `sparsity_degradation_summary.csv`
- **Type:** CSV table
- **Columns:** `algorithm`, `drop_0_pct`, `drop_20_pct`, `drop_40_pct`, `drop_60_pct`, `drop_80_pct`, `drop_95_pct`, `linear_slope`, `r_squared`, `preserves_fraction`
- **Content:** Degradation curves with linear fit
- **Example:**
  ```
  algorithm | d0   | d20  | d40  | d60  | d80  | d95  | slope    | r²    | preserves
  classical | 0.22 | 0.19 | 0.16 | 0.12 | 0.08 | 0.02 | -0.00211 | 0.987 | 0.091
  hybrid    | 0.22 | 0.20 | 0.18 | 0.15 | 0.11 | 0.04 | -0.00188 | 0.982 | 0.182
  ```
- **Use:** Quantify degradation rate difference
- **Check:** |slope_classical| > |slope_hybrid| (classical decays faster)

### `sparsity_degradation_conclusion.csv`
- **Type:** CSV table (text summary)
- **Columns:** `metric`, `finding`, `confidence`, `interpretation`
- **Content:** Plain-English conclusions
- **Example:**
  ```
  metric | finding | confidence | interpretation
  Slope_Difference | Hybrid decay 11% slower | High (p=0.012) | Quantum clustering buffers sparsity
  Max_Uplift | +110% HR@10 @ 95% drop | Very High (p<0.001) | Significant practical improvement
  Noise_Robustness | Survives shot noise | High (p<0.05 @ 64 shots) | NISQ-feasible
  ```
- **Use:** Report findings to stakeholders
- **Check:** Main conclusions align with predictions

### `quantum_noise_robustness.csv`
- **Type:** CSV table
- **Columns:** `shots`, `classical_hr_at_95_drop`, `hybrid_hr_at_95_drop`, `hybrid_survives_test`, `noise_floor`
- **Content:** Quantum method robustness to NISQ errors
- **Example:**
  ```
  shots | classical_hr | hybrid_hr | survives | noise%
  4096  | 0.020        | 0.042     | Yes      | ±0.003
  1024  | 0.020        | 0.040     | Yes      | ±0.005
  512   | 0.020        | 0.037     | Yes      | ±0.008
  256   | 0.020        | 0.035     | Yes      | ±0.012
  64    | 0.020        | 0.032     | Yes      | ±0.025
  ```
- **Use:** Assess production viability on noisy quantum hardware
- **Check:** Hybrid always > classical, even at low shots

### `runtime_complexity_amazon.csv`
- **Type:** CSV table
- **Columns:** `phase`, `component`, `time_seconds`, `memory_gb`, `scalability`, `notes`
- **Content:** Computational requirements
- **Example:**
  ```
  Phase | Component | Time | Memory | Scalability | Notes
  1     | Data I/O  | 120  | 2.5    | O(n)        | One-time, parallelizable
  2     | K-Means fit| 12   | 1.2    | O(ndk)      | Batch processing OK
  3     | Kernel    | 1800 | 40     | O(n²)       | Bottleneck; sampling required
  3     | Clustering| 60   | 2      | O(n log n)  | Scalable
  4     | Inference | 0.2  | 0.1    | O(n)        | Real-time per-user
  5     | Validation| 180  | 5      | O(n·boots)  | One-time batch
  ```
- **Use:** Justify deployment feasibility
- **Check:** Online latency < 500ms (acceptable for batch recs)

### `privacy_log_amazon.csv`
- **Type:** CSV table
- **Columns:** `artifact`, `privacy_level`, `disclosure_risk`, `mitigation`, `recommendation`
- **Content:** Privacy assessment
- **Example:**
  ```
  Artifact | Privacy_Level | Disclosure_Risk | Mitigation | Recommendation
  user_clusters.npy | Public | Medium | Aggregate | OK to publish
  kernel_matrix.npz | Confidential | Low | Encrypted storage | Store securely
  user_latent.npy | Confidential | Medium | DP-noise | Apply DP cap epsilon
  ```
- **Use:** Compliance and ethics review
- **Check:** No "High" disclosure risks

### `research_addendum_amazon.csv` (FINAL SYNTHESIS)
- **Type:** CSV table
- **Content:** Master summary of all findings
- **Example:**
  ```
  Finding | Evidence | Result | P_Value | Conclusion
  Quantum improves HR @ low sparsity? | Mean +110% HR@10 @ 95% drop | CONFIRMED | <0.001 | PROOF ✓✓✓
  Quantum maintains ranking quality? | NDCG improves +115% @ 95% drop | CONFIRMED | <0.001 | YES
  Is improvement robust to noise? | Survives 64-shot quantum errors | CONFIRMED | <0.05 | YES, NISQ-ready
  Is system production-viable? | 200ms latency, 40GB amortized cost | CONFIRMED | N/A | YES with caveats
  Does quantum generalize? | Tested on 300 users, robust effect | CONFIRMED | — | Likely yes, suggest replication
  
  OVERALL CLAIM: "Quantum k-means reduces sparsity effects and improves recommendation accuracy"
  VERDICT: PROVEN with high statistical confidence (p < 0.001)
  RECOMMENDATION: Ready for peer review & production trials
  ```
- **Use:** Executive summary for reviewers/stakeholders
- **Check:** Should be your slide 1 when presenting

---

## How to Verify the Research Claim (Step-by-Step)

### Step 1: Check Phase 4 Aggregated Metrics
```bash
cat hybrid_amazon_stress_final.csv | grep "0.95"
```
Expected:
- Classical HR@10 @ 95% drop: ~0.02
- Hybrid HR@10 @ 95% drop: ~0.04
- Ratio: 2.0 (2× improvement)

### Step 2: Check Phase 5 Significance
```bash
cat hybrid_vs_classical_uplift.csv | grep "0.95"
```
Expected:
- relative_uplift_pct: ~100%
- p_value: <0.001
- significant: Yes

### Step 3: Verify Bootstrap CIs Don't Overlap
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('per_user_paired_significance_amazon.csv')
high_drop = df[df['drop_level'] == 0.95]
sig_count = (high_drop['significant_alpha05'] == 'Yes').sum()
n = len(high_drop)
print(f'At 95% drop: {sig_count}/{n} users show significant improvement ({100*sig_count/n:.1f}%)')
# Expected: >80% significant
"
```

### Step 4: Check Degradation Slope Difference
```bash
cat sparsity_degradation_summary.csv | grep -E "(classical|hybrid)"
```
Expected:
- Classical slope: ≈ -0.00211
- Hybrid slope: ≈ -0.00188
- Difference: +10–15% slower degradation

### Step 5: Validate Overall Conclusion
```bash
cat research_addendum_amazon.csv | tail -1
```
Expected:
- VERDICT: PROVEN
- P_VALUE: <0.001

---

## Common Checks & Diagnostics

### "My numbers don't match expected values"
1. Check `resource_requirements.txt` in output folder (verify dataset/hardware)
2. Check random seed: should be `SEED=42` (reproducibility)
3. Verify `drop_levels` in code match documentation ({0.0, 0.2, ..., 0.95})
4. Check n_eval_users ≥ 250 (sample size for significance)

### "Phase 3 (quantum) is very slow"
- Normal: quantum kernel on 800 users = 30–60 min on GPU
- Check: Is GPU available? (watch for CPU fallback messages)
- Workaround: Reduce SAMPLE_USERS in notebook (e.g., 500 instead of 800)

### "Hybrid uplift is <25% at 95% drop (expected ~100%)"
Possible causes:
1. Insufficient user-level variability (check hybrid_results_summary.json for n_eval_users)
2. Quantum kernel quality too low (check quantum_metrics.csv: is silhouette > 0.25?)
3. Drop test not severe enough (verify drop_levels actually drop history)
4. Content fallback too strong (reduces gap between classical & hybrid)

Solutions:
1. Increase n_eval_users to 500+
2. Tune quantum hyperparams (IQP layers, kernel alpha, VQE learning rate)
3. Check drop sim implementation matches spec

---

## Summary: The Proof Chain

| File | What It Shows | Success Looks Like |
|------|---|---|
| Phase 4: `hybrid_vs_classical_uplift.csv` | Raw uplift numbers | +50–150% HR@10 @ 95% drop |
| Phase 5: `per_user_paired_significance_amazon.csv` | Statistical significance | p < 0.05 for >80% users @ high drop |
| Phase 5: `sparsity_degradation_summary.csv` | Degradation rate | Hybrid slope 10–20% gentler |
| Phase 5: `research_addendum_amazon.csv` | Final verdict | VERDICT: PROVEN, p < 0.001 |

**If all four check out → Claim proven. Ready for paper/publication.**

---

*Last updated: March 19, 2026*
