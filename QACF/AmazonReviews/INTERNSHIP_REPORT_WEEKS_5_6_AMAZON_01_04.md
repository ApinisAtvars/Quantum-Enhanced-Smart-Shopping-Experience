# 2.3 Internship Report – Weeks 5 & 6 (Amazon Pipeline 01–04)

## Achievements

During this period, I completed and integrated the Amazon-focused pipeline from data engineering to hybrid quantum-classical stress testing (Notebooks 01–04). The work extended beyond a basic implementation by adding robust routing logic, multiple quantum clustering modes, and sparsity-conditioned evaluation tracks.

### A. Amazon Data Engineering & Sparse Feature Fabric (Notebook 01)

**The Challenge:** Amazon Books interactions are extremely sparse, and raw JSONL ingestion must remain scalable while preserving user-item structure for later quantum and classical comparison.

**The Solution:** I implemented a chunked ingestion and preprocessing workflow using `Books.jsonl` and `meta_Books.jsonl`, followed by ID remapping, sparse matrix construction, latent decomposition, and content feature extraction.

**Implementation Highlights:**
- Built reusable artifacts for downstream experiments (`user_sparse.npz`, `user_latent.npy`, `item_latent.npy`, `user_combined.npy`, `item_combined.npy`, `id_maps.pkl`).
- Structured data handoff so the same Phase 1 outputs are consumed consistently by classical (02), quantum (03), and hybrid (04) stages.
- Preserved direct linkage to raw data in `QACF/AmazonReviews/raw`, ensuring reproducibility and auditability.

**Impact:** This phase established a stable experiment substrate for controlled classical-vs-quantum comparisons under sparsity stress.

---

### B. Classical Baseline Control Arm with Tuned Clustering (Notebook 02)

**The Challenge:** A meaningful quantum comparison requires a strong and transparent classical baseline with tuned hyperparameters and clear performance diagnostics.

**The Solution:** I ran K-sweep tuning and selected the best silhouette regime before computing baseline recommendation metrics.

**Measured Results (from `amazon_tuning.csv` and `baseline_metrics.csv`):**
- Best observed silhouette in sweep: **0.2051 at k=16**.
- Classical item-CF baseline (`item_cf_cosine`):
  - **RMSE:** 1.1072
  - **MAE:** 0.6204
  - **HR@10:** 0.3975
  - **NDCG@10:** 0.2411
  - **Precision@10:** 0.03975
- Content-only baseline (`content_based_cosine`) remained substantially weaker on ranking metrics:
  - **HR@10:** 0.0675
  - **NDCG@10:** 0.0365

**Impact:** This produced a credible control arm and confirmed that collaborative signal remains dominant before aggressive sparsity stress is applied.

---

### C. Quantum Prototype Expansion with Multi-Mode Clustering (Notebook 03)

**The Core Challenge:** Notebook 03 exposed a difficult geometric issue: the quantum feature space was producing clusters that were detectable, but not consistently well-separated enough to beat the best classical silhouette.

I treated this as the central engineering problem of Weeks 5–6 and ran an iterative ablation cycle focused on silhouette stability.

**Silhouette Problem Statement:**
- Classical sweep in Notebook 02 achieved a strongest silhouette of **0.2051 (k=16)**.
- Initial and intermediate quantum variants often landed much lower (frequently in the **0.03–0.11** band).
- The best quantum configuration in this cycle reached **0.1270**, which is meaningful but still below the strongest classical reference.

**What we tried in Notebook 03 to improve silhouette:**

1. **Kernel family expansion (instead of single-kernel dependence)**
- Tested spectral and enhanced spectral families (`spectral_only`, `enhanced_vqe_spectral`).
- Added fused and trainable kernels to reduce single-encoding brittleness.
- Rationale: if one kernel over-compresses distances, a fused kernel can recover neighborhood contrast.

2. **Mode-level diversification (clustering algorithm layer)**
- Evaluated `quantum_lloyd`, `HDBSCAN`, `HDBSCAN_tuned`, `QUBO`, and ensemble variants.
- Introduced density-based tuning to avoid forcing equal-size partitions when user geometry is highly uneven.
- Rationale: poor silhouette can come from mismatch between geometry and clustering objective, not only from the kernel.

3. **Cluster-count sweeps and elbow-proxy diagnostics**
- Ran broad K sweeps (multiple modes with k from small to larger ranges).
- Used silhouette and Davies-Bouldin jointly to avoid selecting K that optimizes one metric while collapsing another.
- Rationale: we observed several local maxima that were not robust across metrics.

4. **Noise-aware checks for NISQ realism**
- Tracked noisy-kernel correlation and noisy MAE as guardrails.
- Rationale: an apparently better silhouette can be misleading if it is not robust under noise perturbations.

**What the evidence showed (Notebook 03 outputs):**
- `quantum_method_comparison_amazon.csv` ranked methods approximately as:
  - `HDBSCAN_tuned`: **0.1270**
  - `quantum_lloyd`: **0.1270**
  - `HDBSCAN`: **0.1138**
  - `kernel_fusion_trainable`: **0.1101**
  - `ensemble_hdb_qubo`: **0.0341**
  - `QUBO`: **-0.0099**
- `quantum_clustering_sweep.csv` showed many regimes with low or near-zero silhouette despite substantial tuning effort.
- `quantum_metrics.csv` confirmed the best viable run:
  - **Sample users:** 320
  - **Qubits:** 6
  - **Best kernel:** `trainable_fusion+ensemble`
  - **Best mode:** `HDBSCAN_tuned`
  - **Best k:** 22
  - **Silhouette:** **0.1270**
  - **Noisy correlation:** 0.3538
  - **Noisy MAE:** 0.3094

**Challenge Interpretation:**
The main bottleneck was not “no clustering signal,” but **insufficient margin between clusters** under sparse-user geometry. In other words, Notebook 03 produced workable quantum group structure, but not yet a consistently high-separation manifold.

**Impact:** Notebook 03 delivered a robust experimentation framework and identified the most stable quantum operating point for downstream hybrid routing, even though the silhouette ceiling remained below the strongest classical baseline.

---

### D. Hybrid Sparsity-Aware Routing & Stress Protocol (Notebook 04)

**The Core Challenge:** In Notebook 04, the hardest problem was translating moderate clustering quality into stable top-N recommendation gains under extreme sparsity. A good silhouette does not automatically guarantee higher HR@10 once ranking, neighbor scarcity, and candidate filtering interact.

**What we tried in Notebook 04 to address this:**

1. **Sparsity-conditional routing instead of one global policy**
- Implemented active/sparse/cold routing logic.
- Increased quantum-path reliance as user history was progressively dropped.
- Objective: preserve recall when classical neighborhoods become too thin.

2. **Routing refinement layer on quantum labels**
- `hybrid_results_summary.json` shows raw quantum routing groups and a refined grouping stage.
- Raw imported label structure had unstable geometry (`raw_silhouette` reported as **-0.5605**), then routing refinement produced 2 routing groups with very high internal separation (`refined_silhouette` **0.9412**).
- Practical interpretation: refinement improved *routing separability*, but this did not fully translate into monotonic ranking gains at every drop level.

3. **Hybrid scoring controls**
- Tuned candidate pool and neighborhood scan (`candidate_pool=3500`, `max_neighbor_scan=600`).
- Applied blend controls (`base_quantum_weight=0.45`, shrinkage) and post-ranking controls (diversity lambda and novelty boost).
- Objective: reduce overfitting to sparse noisy neighbors while preserving recommendation diversity.

4. **Stress protocol at multiple drop levels**
- Evaluated across drop levels from dense to severe sparse.
- Tracked both ranking metrics and route composition to understand *why* a specific drop level improved or failed.

**Observed routing behavior (from `hybrid_amazon_stress_final.csv`):**
- **Drop 0.0:** classical-active 68 users (22.67%), quantum-sparse 232 users (77.33%).
- **Drop 0.8:** classical-active 3 users (1%), quantum-sparse 297 users (99%).
- **Drop 0.9 / 0.95:** 100% quantum-sparse routing.

**Observed stress outcomes (Hybrid vs Classical):**
- **Drop 0.0:** HR@10 = 0.00667 vs 0.00000 (hybrid advantage).
- **Drop 0.6:** HR@10 = 0.00000 vs 0.00333 (hybrid underperformed).
- **Drop 0.9:** HR@10 = 0.01000 vs 0.00000 (hybrid advantage).
- **Drop 0.95:** HR@10 = 0.00333 vs 0.00000 (hybrid advantage).

**Additional stress evidence (from `sparsity_stress_test_results.csv`):**
- At some severe drops, hybrid improved RMSE while HR@10 stayed flat.
- At other drops, RMSE and HR@10 diverged in direction, confirming that score calibration and top-N ranking quality were not perfectly aligned.

**Silhouette-to-ranking challenge in practice:**
Notebook 04 confirmed a key difficulty: **cluster geometry quality and recommendation ranking quality are related but not equivalent objectives**. We improved routing structure and saw sparse-regime wins at key points, but performance remained non-monotonic across all stress levels.

**Early significance signal:**
- Available p-values in this stage remained above 0.05 in populated rows (e.g., ~0.4226), so the Week 5–6 outcome is best interpreted as *promising engineering progress with mixed statistical strength*.

**Impact:** Notebook 04 successfully operationalized the hybrid protocol and exposed exactly where the next optimization cycle is needed: routing thresholds, hybrid weight schedule, and ranking calibration under extreme history loss.

---

## Validation of Project Goals (Weeks 5–6 Status)

The Week 5–6 objective for Amazon 01–04 was to build a complete, testable mechanism that can answer whether quantum-assisted routing mitigates sparsity degradation.

**Status:** **Engineering objective achieved; statistical claim still in-progress.**

- ✅ **Achieved:** End-to-end Amazon pipeline implementation from raw ingestion to hybrid stress outputs.
- ✅ **Achieved:** Dynamic routing architecture that increases quantum path usage as sparsity increases.
- ✅ **Achieved:** Exported benchmark tables required for formal statistical validation.
- ⚠️ **Pending stronger evidence:** In this run, high-sparsity improvements are present at some drops (e.g., 0.9, 0.95) but are not uniformly dominant across all drop levels.
- ⚠️ **Pending significance:** Available p-values in the Week 5–6 stage outputs are not yet below conventional significance threshold.

This means the research platform is now fully operational, and the next milestone is parameter/seed stabilization to convert directional uplift into consistently significant uplift.

---

## Stress-Test Interpretation (Long-Tail Framing)

**What it is:** I simulated long-tail sparsity by progressively dropping user history from 0% to 95%, then compared hybrid routing against a classical-only protocol.

**The result:** The hybrid model produced positive HR@10 at several severe-sparsity points where classical remained at zero (notably 0.9 and 0.95), but showed non-monotonic behavior (e.g., drop 0.6 underperformed).

**What it means:** The mechanism for sparsity-aware quantum routing is functioning and can recover wins in very sparse slices, but requires further calibration to make gains stable and statistically decisive across the entire degradation curve.

---

## Research Conducted (Weeks 5–6)

### 1) Sparse Data Engineering for Quantum-Ready Inputs

**What it is:** Designing feature pipelines so classical and quantum branches consume aligned user representations.

**Why I used it:** Misalignment between preprocessing stages can create false uplift/loss signals and invalidate comparisons.

**Contribution:** I standardized shared artifacts (`user_latent`, `user_combined`, sparse matrix maps) and ensured deterministic phase handoff from Notebook 01 into 02/03/04.

---

### 2) Classical Baseline Tuning Under Cluster-Count Sensitivity

**What it is:** K-means parameter sweep and metric-grounded baseline selection.

**Why I used it:** Quantum claims are only credible when the classical baseline is well-tuned and transparently reported.

**Contribution:** I profiled inertia/silhouette across candidate K values, selected a stable baseline regime, and established control metrics for direct hybrid comparison.

---

### 3) Quantum Kernel/Mode Exploration for Robustness

**What it is:** Comparative testing of quantum kernels and clustering modes (including fusion/ensemble and tuned density-based clustering).

**Why I used it:** Single-kernel conclusions are fragile in NISQ-like settings where noise and mode choice can dominate outcomes.

**Contribution:** I implemented a multi-mode evaluation scaffold and captured noisy-kernel diagnostics to quantify robustness, not only point performance.

---

### 4) Hybrid Routing Policy Under Controlled Sparsity Degradation

**What it is:** Rule-based protocol that reallocates users from classical to quantum neighborhoods as interaction history vanishes.

**Why I used it:** The practical target is sparse-user resilience, not replacing classical methods for all users.

**Contribution:** I operationalized a progressive routing policy and verified its behavior empirically (classical-active share falling toward 0 as drop severity increases), creating a production-relevant hybrid control logic.

---

### 5) Early-Stage Statistical Readiness

**What it is:** Preparing experiment outputs for paired significance and confidence analysis.

**Why I used it:** Stochastic recommenders can show random gains; formal tests are needed before claiming effect.

**Contribution:** I exported uplift and paired-comparison tables from Phase 04 in a validation-ready format, enabling immediate downstream significance testing.

---

## Technical Reflection

Weeks 5–6 produced a complete and reproducible Amazon 01–04 hybrid experimentation stack. The strongest outcome is architectural: the system now supports rigorous sparsity-conditioned comparisons with explicit routing diagnostics and reproducible outputs. Empirical results show promising sparse-slice gains but mixed consistency, indicating that the next optimization cycle should target seed stability, routing thresholds, and kernel-mode calibration to secure robust statistically significant uplift.

---

## Deep-Dive: Silhouette Challenges and What We Learned (03 & 04)

### Why silhouette was difficult in this dataset

1. **Ultra-sparse behavior manifold**
- Sparse user histories reduced overlap structure, making cluster boundaries weak and unstable.

2. **Metric mismatch risk**
- Optimizing silhouette alone did not guarantee better recommendation HR@10, especially after candidate filtering and reranking.

3. **Method sensitivity**
- Quantum clustering quality varied substantially by mode (from negative silhouette in QUBO variants to moderate positive values in tuned HDBSCAN/Lloyd).

4. **Routing transfer gap**
- Even when routing groups looked clean (high refined silhouette), downstream ranking gains could still fluctuate by sparsity level.

### What we tried specifically to overcome it

- Kernel fusion and ensemble kernels to improve geometric contrast.
- HDBSCAN tuning and multiple K sweeps to avoid rigid centroid assumptions.
- Routing refinement and dynamic path assignment in Notebook 04.
- Score blending, shrinkage, and diversity/novelty controls to stabilize top-N outputs.
- Multi-table diagnostics (silhouette + Davies-Bouldin + HR@10 + NDCG@10 + route composition), rather than trusting one metric.

### Practical conclusion for Weeks 5–6

The work validated the engineering hypothesis that hybrid quantum routing can recover wins in severe sparse slices, but also revealed the research reality that silhouette gains must be paired with ranking-calibration improvements to produce consistent, statistically significant end-metric uplift.
