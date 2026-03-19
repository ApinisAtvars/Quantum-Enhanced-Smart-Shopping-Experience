# AmazonReviews Pipeline: Verified Implementation Audit + Why Each File Exists

## Objective
Demonstrate that **quantum k-means–based clustering reduces sparsity effects and improves recommendation accuracy vs classical clustering** on the Amazon Reviews data in `QACF/AmazonReviews/raw`.

---

## 1) Implementation check against `QACF/AmazonReviews/raw`

### Raw files present
- `raw/Books.jsonl` ✅
- `raw/meta_Books.jsonl` ✅

### Is the pipeline wired to these exact files?
- `01_data_prep_amazon.ipynb` ✅ **Direct ingestion**
  - Searches `QACF/AmazonReviews/raw`
  - Picks review candidates including `Books.jsonl`
  - Picks metadata candidates including `meta_Books.jsonl`
  - Builds processed artifacts used by all downstream stages

- `04_hybrid_integration_amazon.ipynb` ✅ **Direct + indirect usage**
  - Indirectly uses Stage-01 processed matrices/features
  - Also re-reads raw reviews (`Books.jsonl` candidate) for helpfulness-weight priors

- `02_classical_baselines_amazon.ipynb` ✅ **Indirect usage**
  - Uses Stage-01 outputs (no direct raw read)

- `03_quantum_prototype_amazon.ipynb` ✅ **Indirect usage**
  - Uses Stage-01/02 features and emits quantum artifacts

- `05_validation_amazon.ipynb` ✅ **Indirect usage**
  - Uses Stage-02/03/04 outputs for statistical validation

**Conclusion:** Yes, implementation is made for the data in `QACF/AmazonReviews/raw`. Raw data is directly consumed in Stage 01 (and partially in Stage 04), and all later stages consume derived artifacts.

---

## 2) What each file does and why

### `raw/Books.jsonl`
- **What:** Review-level interaction data (user, item, rating, text, helpfulness-like fields).
- **Why:** Primary collaborative signal and sparsity substrate.

### `raw/meta_Books.jsonl`
- **What:** Item metadata (titles/categories and related product context).
- **Why:** Content priors for cold/sparse fallback where CF is weak.

### `01_data_prep_amazon.ipynb` (Phase 1)
- **What it does:** Chunked raw ingestion, ID remapping, sparse matrix construction, latent/content feature generation.
- **Why it matters:** Creates a consistent feature space used by both classical and quantum branches.
- **Primary outputs:**
  - `user_sparse.npz`
  - `user_latent.npy`
  - `item_latent.npy`
  - `content_features.npz`
  - `user_combined.npy`, `item_combined.npy`
  - `id_maps.pkl`
  - `category_sparsity.csv`, `review_length_stats.csv`

### `02_classical_baselines_amazon.ipynb` (Phase 2)
- **What it does:** Classical MiniBatch K-Means clustering + baseline recommender metrics.
- **Why it matters:** Provides the control arm for quantum-vs-classical comparison.
- **Primary outputs:**
  - `user_clusters.npy`
  - `baseline_metrics.csv` (or `amazon_metrics.csv` compatibility)
  - `amazon_tuning.csv`

### `03_quantum_prototype_amazon.ipynb` (Phase 3)
- **What it does:** Quantum kernel construction + quantum clustering variants (improved, D-Wave hybrid, VQE variants), plus cluster-quality reporting.
- **Why it matters:** Produces quantum-derived neighborhood structure expected to be more robust under sparse signal.
- **Primary outputs (representative):**
  - `quantum_user_labels_improved.npy`
  - `quantum_user_labels_dwave.npy`
  - `quantum_user_labels_vqe_adiabatic.npy`
  - `quantum_user_labels_vqe_subspace_final.npy`
  - `kernel_matrix.npz`
  - `quantum_metrics.csv`, `quantum_metrics.json`
  - `quantum_clustering_sweep.csv`, `quantum_elbow.csv`

### `04_hybrid_integration_amazon.ipynb` (Phase 4)
- **What it does:** Hybrid routing by user sparsity profile:
  - active users → stronger classical neighborhood signal
  - sparse users → quantum-label-guided neighborhood/reranking
  - extreme-cold users → stronger content and priors
  - includes stress tests with controlled history dropping
- **Why it matters:** This is the direct test bench for “reduced sparsity degradation.”
- **Primary outputs:**
  - `hybrid_amazon_recs.csv`
  - `hybrid_amazon_stress.csv`
  - `hybrid_amazon_stress_final.csv`
  - `classical_only_stress.csv`
  - `hybrid_vs_classical_uplift.csv`
  - `drop0_eval_hybrid.csv`, `drop0_eval_classical.csv`
  - `hybrid_results_summary.json`

### `05_validation_amazon.ipynb` (Phase 5)
- **What it does:** Bootstrap/paired significance checks, degradation summaries, robustness/complexity logs.
- **Why it matters:** Converts observed improvements into defensible evidence with uncertainty and paired tests.
- **Primary outputs (representative):**
  - `research_addendum_amazon.csv`
  - `per_user_paired_rows_amazon.csv`
  - `per_user_paired_significance_amazon.csv`
  - `sparsity_degradation_summary.csv`
  - `sparsity_degradation_conclusion.csv`
  - `runtime_complexity_amazon.csv`
  - `privacy_log_amazon.csv`

### `data/processed_amazon/` and `QACF/data/processed_amazon(_5core)/`
- **What:** Stage handoff artifacts.
- **Why:** Separate expensive preprocessing/quantum generation from repeatable evaluation/validation.

---

## 3) Complete pipeline (claim-oriented)

1. **Raw ingestion and featurization (01)**
   - Input: `raw/Books.jsonl`, `raw/meta_Books.jsonl`
   - Output: sparse matrix + latent/content features + ID maps

2. **Classical baseline branch (02)**
   - Build classical clusters and compute baseline recommendation metrics

3. **Quantum clustering branch (03)**
   - Build quantum kernel(s), generate quantum labels, report cluster-quality diagnostics

4. **Hybrid sparsity routing experiment (04)**
   - Run hybrid and classical-only protocols across multiple sparsity-drop levels
   - Export direct uplift and stress-comparison tables

5. **Validation/statistics (05)**
   - Add bootstrap/paired significance and degradation conclusions

---

## 4) How to demonstrate the research claim with produced files

Use this file set together:
- `hybrid_vs_classical_uplift.csv`
- `hybrid_amazon_stress_final.csv` (fallback: `hybrid_amazon_stress.csv`)
- `classical_only_stress.csv`
- `baseline_metrics.csv` / `amazon_metrics.csv`
- `quantum_metrics.csv`
- `per_user_paired_significance_amazon.csv`
- `sparsity_degradation_summary.csv`

Evidence pattern to report:
1. At high drop rates (typically 0.80–0.95), hybrid/quantum-informed curves degrade slower than classical-only.
2. Ranking metrics (e.g., `hr@10`, `ndcg@10`) show positive uplift under sparse regimes.
3. Paired/significance outputs keep the uplift direction under uncertainty checks.

---

## 5) Important path reality (observed in current notebooks)

The notebooks use dynamic path detection and may read/write in different processed folders depending on launch location:
- `QACF/AmazonReviews/data/processed_amazon`
- `QACF/data/processed_amazon`
- `QACF/data/processed_amazon_5core`

In this repository snapshot, Phase-01/02 artifacts are visible under `QACF/AmazonReviews/data/processed_amazon`, while many Phase-03/04/05 outputs are visible under `QACF/data/processed_amazon` and `QACF/data/processed_amazon_5core`.

For reproducible end-to-end runs, use one consistent working directory and verify `IN_DIR` / `OUT_DIR` prints at the start of each notebook.
