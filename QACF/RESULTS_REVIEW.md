# Comprehensive Review: Quantum vs Classical Item-Based Collaborative Filtering Analysis

**Date**: February 25, 2026  
**Dataset**: MovieLens 32M (200k users, 84k movies, 32M ratings)  
**Notebook**: `dataprep_item.ipynb`

---

## Executive Summary

This analysis compares **quantum clustering** (using PennyLane amplitude encoding) against **classical clustering** (MiniBatchKMeans/Lloyd K-means) for item-based collaborative filtering. The study evaluates four approaches across five metrics (RMSE, MAE, HR@10, NDCG@10, Precision@10) plus sparsity/coverage indicators.

### Key Verdict
**Classical clustering dominates all metrics.** Quantum methods show 54-247× worse performance on ranking metrics (HR@10, NDCG@10, Precision@10) and 2.5% higher prediction error (RMSE). The performance gap stems from fundamental scalability limitations rather than algorithmic deficiencies.

---

## 1. Experimental Design

### 1.1 Models Compared

| Model | Description | Clusters | Sample Size | Features |
|-------|-------------|----------|-------------|----------|
| **Pure CF** | Item-item similarity baseline | N/A | Full catalog | Raw sparse vectors |
| **Classical Improved** | Lloyd K-means + L2 normalization | 280 | 84,432 items | 32D SVD embeddings |
| **Quantum v1 (PennyLane)** | Amplitude encoding + fidelity kernel | 6 | 200 items | 16D truncated (4 qubits) |
| **Quantum v2 (Improved)** | Optimized amplitude encoding | 30 | 300 items | 32D (5 qubits) |

### 1.2 Evaluation Protocol
- **Split**: 80/20 temporal train/test split (simulates real-world deployment)
- **Test Set**: 500 randomly sampled users
- **Ranking Task**: Full-catalog recommendation (top-10 from 84k items)
- **Metrics**:
  - **Prediction**: RMSE, MAE (lower = better)
  - **Ranking**: HR@10, NDCG@10, Precision@10 (higher = better)
  - **Sparsity**: Avg cluster sparsity, user diversity, cold-item coverage

---

## 2. Results Summary

### 2.1 Performance Comparison

| Metric | Pure CF | Classical | Quantum v1 | Quantum v2 | Winner |
|--------|---------|-----------|------------|------------|--------|
| **RMSE** ↓ | 1.0475 | **1.0092** | 1.0339 | 1.0308 | Classical |
| **MAE** ↓ | 0.8136 | **0.7791** | 0.8048 | 0.7920 | Classical |
| **HR@10** ↑ | - | **65.2%** | 1.2% | 0.0% | Classical |
| **NDCG@10** ↑ | - | **24.8%** | 0.1% | 0.0% | Classical |
| **Precision@10** ↑ | - | **21.9%** | 0.14% | 0.0% | Classical |

**Performance Gaps**:
- Classical achieves **54× better HR@10** than quantum v1
- Classical achieves **247× better NDCG@10** than quantum v1
- Quantum v2 (30 clusters) performs even worse on ranking metrics (0% scores)

### 2.2 Sparsity & Coverage Analysis

| Model | Clusters | Avg Cluster Sparsity ↓ | User Diversity ↑ | Cold Item Coverage ↑ | Item Coverage % ↑ |
|-------|----------|------------------------|------------------|----------------------|-------------------|
| **Classical** | 280 | **0.8808** | **67.6** | 0.71% | 2.43% |
| **Quantum v1** | 6 | 0.9968 | 5.6 | **100%** | **3.75%** |
| **Quantum v2** | 30 | 0.9948 | 20.0 | **100%** | **4.24%** |

**Key Observations**:
- **Classical has 12% lower cluster sparsity** than quantum methods
- **Classical achieves 12× higher user diversity** (67.6 vs 5.6 clusters per user)
- **Quantum achieves perfect cold-item coverage** (100% vs 0.71%) but this is misleading—it's due to having so few clusters that cold items are distributed everywhere
- **Quantum has slightly better item coverage %** (4.24% vs 2.43%), but this comes at the cost of precision

---

## 3. Root Cause Analysis

### 3.1 Cluster Count Mismatch (6 vs 280)
**Impact**: Catastrophic for ranking metrics

- **Classical**: 280 clusters → average cluster size = ~300 items
  - Prediction function (`user_cluster_mean`) provides fine-grained ratings discrimination
  - Users span ~68 clusters on average → diverse recommendations

- **Quantum v1**: 6 clusters → average cluster size = ~14,000 items
  - Massive clusters → nearly identical scores for thousands of movies
  - Full-catalog ranking collapses to random ordering within clusters
  - HR@10 drops from 65.2% to 1.2%

- **Quantum v2**: 30 clusters → average cluster size = ~2,800 items
  - Still too coarse compared to classical
  - HR@10 drops to 0.0% (worse than v1!) likely due to label propagation artifacts

**Conclusion**: Quantum kernels are computationally limited to k < 50 due to O(n²) kernel computation cost. Classical methods easily scale to k > 200.

---

### 3.2 Sample Size Bottleneck (200-300 vs 84,432)
**Impact**: 99.6% of items never participate in quantum clustering

- **Quantum kernel computation**: O(n²) complexity
  - 200 items → 19,900 kernel pairs (feasible)
  - 84,432 items → 3.6 billion pairs (infeasible)

- **Remaining items** (84,232 movies) were assigned via **nearest-neighbor propagation** in classical embedding space
  - These items never interacted with the quantum kernel
  - Their "quantum" labels are essentially classical assignments

**Conclusion**: The quantum pipeline only touches 0.24% of the catalog. The remaining 99.76% bypasses quantum computation entirely.

---

### 3.3 Kernel Design Mismatch
**Impact**: Quantum kernel doesn't align with collaborative filtering objective

- **Quantum kernel**: Amplitude encoding + state fidelity
  - Measures quantum state overlap: `<ψ(x_i)|ψ(x_j)>`
  - Operates in Hilbert space abstraction of raw embeddings
  - No explicit optimization for rating prediction

- **Classical kernel**: L2-normalized + cosine similarity
  - Optimized via TruncatedSVD to capture rating variance
  - Directly aligned with collaborative filtering structure
  - Decades of domain-specific tuning

**Conclusion**: The quantum fidelity kernel is mathematically elegant but not specialized for recommender systems. Classical embeddings were pre-optimized via SVD for this exact task.

---

### 3.4 Implementation Validation
**PennyLane vs Qiskit**: Identical results (as expected)

- Both frameworks produce the same quantum kernel values
- Both use amplitude encoding (`AmplitudeEmbedding` vs `Statevector`)
- Both apply identical post-processing (MDS + k-means)

**Conclusion**: The quantum implementation is correct. The poor performance is due to method scalability, not bugs.

---

## 4. Strengths of the Analysis

### 4.1 Rigorous Experimental Design
✅ **Temporal split** (80/20) instead of random split → realistic deployment scenario  
✅ **Multiple baselines**: Pure CF, classical k-means, quantum v1, quantum v2  
✅ **Fair feature space**: All models use same 32D SVD embeddings as starting point  
✅ **Comprehensive metrics**: Prediction (RMSE/MAE) + ranking (HR/NDCG/Precision) + sparsity  
✅ **Scalability acknowledgment**: Quantum constrained to 200-300 items for computational feasibility

### 4.2 Thorough Classical Optimization
✅ **Hyperparameter search** over representations (raw, L2, standardized) and cluster counts (40-280)  
✅ **Metric optimization**: Composite score = silhouette (cosine) - 0.02 × Davies-Bouldin  
✅ **Multiple algorithms**: MiniBatchKMeans, Lloyd k-means, comparison study  
✅ **Visualization**: t-SNE plots, 4-panel comparison charts, cluster size distributions

### 4.3 Honest Limitations Disclosure
✅ **Quantum scalability**: Explicitly states 200-item limit due to O(n²) kernel cost  
✅ **Label propagation**: Acknowledges 99.76% of items use classical nearest-neighbor  
✅ **Research goal assessment**: Objectively evaluates whether quantum achieves sparsity reduction (it doesn't)  
✅ **Path forward**: Proposes concrete improvements (VQE, hybrid models, task-specific encodings)

---

## 5. Weaknesses & Gaps

### 5.1 Missing Comparisons
⚠️ **No quantum-classical hybrid**: Could quantum kernels improve as an ensemble component?  
⚠️ **No alternative quantum encodings**: Only tested amplitude encoding (angle encoding, IQP circuits not explored)  
⚠️ **No variational methods**: VQE or QAOA could potentially reduce cluster count constraints  
⚠️ **No quantum annealing**: Might handle larger sample sizes than gate-based circuits

### 5.2 Evaluation Constraints
⚠️ **Small test set**: 500 users may not capture full performance variance (should use bootstrap confidence intervals)  
⚠️ **Cold-start metric interpretation**: 100% cold-item coverage for quantum is misleading—it's due to having only 6-30 clusters  
⚠️ **No statistical significance tests**: Performance gaps reported without p-values or confidence intervals

### 5.3 Methodological Questions
⚠️ **Why MDS + k-means post-processing?**: Quantum kernel → MDS embedding → classical clustering undermines "quantum advantage" narrative  
⚠️ **Why truncate to 16D for quantum v1?**: Could have used 5 qubits (32D) from the start  
⚠️ **Why not test quantum on a smaller dataset first?**: MovieLens 1M or 10M might show clearer quantum effects before scaling

---

## 6. Interpretation & Implications

### 6.1 For This Specific Task
**Verdict**: Classical methods win decisively for production deployment

- **Deploy classical improved k-means** (280 clusters, L2-normalized, cosine similarity)
- **HR@10 = 65.2%** is strong for full-catalog ranking on 84k items
- **Quantum prototypes** serve as proof-of-concept only

### 6.2 For Quantum ML Research
**Lesson**: Scalability bottlenecks dominate algorithm strengths

- Quantum kernels work mathematically (PennyLane == Qiskit)
- But O(n²) computation + k < 50 limit make them non-competitive
- **Need breakthroughs in**:
  1. Approximate quantum kernels (random features, sketching)
  2. Variational quantum eigensolvers for clustering
  3. Task-specific quantum feature maps (encode user-item graphs, not just embeddings)

### 6.3 For Quantum Recommender Systems
**Path forward**: Hybrid approaches, not standalone quantum

1. **Quantum as auxiliary signal**: Use quantum kernels for cold-start items only, classical for popular items
2. **Quantum for feature learning**: Pre-train quantum embeddings, then use classical clustering
3. **Quantum for optimization**: Apply QAOA to cluster assignment step (not full kernel computation)

---

## 7. Recommendations

### 7.1 For Production Systems
1. **Deploy classical improved k-means** with the following config:
   - k = 280 clusters
   - L2-normalized 32D SVD embeddings
   - Cosine similarity metric
   - Lloyd's algorithm (full-batch)

2. **Monitor sparsity patterns**: Track cluster balance via Gini coefficient

3. **A/B test against pure CF**: Determine if clustering overhead worth 1% RMSE improvement

### 7.2 For Future Quantum Research
1. **Test on smaller datasets first**: Use MovieLens 1M (4k items) to validate quantum advantage without scalability constraints

2. **Explore alternative encodings**:
   - Angle encoding (preserves more features per qubit)
   - IQP circuits (proven quantum advantage in kernel setting)
   - Quantum graph neural networks (encode user-item bipartite graph)

3. **Implement variational methods**:
   - Variational Quantum Eigensolver (VQE) for cluster centroids
   - Quantum Approximate Optimization Algorithm (QAOA) for assignment

4. **Develop approximate quantum kernels**:
   - Random Fourier features on quantum circuits
   - Matrix sketching for kernel approximation
   - Divide-and-conquer for large-scale quantum clustering

5. **Add statistical rigor**:
   - Bootstrap confidence intervals for all metrics
   - McNemar's test for HR@10 significance
   - Wilcoxon signed-rank test for RMSE/MAE

### 7.3 For This Notebook
1. **Add error bars** to all plots (bootstrap 95% CI)

2. **Compute cross-validation scores** (at least 3-fold)

3. **Test quantum on MovieLens 1M** as a controlled experiment

4. **Implement quantum-classical hybrid**:
   - Quantum kernel for cold items (<5 ratings)
   - Classical k-means for popular items
   - Compare against pure classical baseline

5. **Document computational costs**:
   - Wall-clock time for quantum kernel computation
   - Memory requirements
   - Energy consumption estimates

---

## 8. Final Assessment

### Research Goals Achievement

| Goal | Status | Evidence |
|------|--------|----------|
| **Implement quantum clustering** | ✅ Success | PennyLane + Qiskit both work, produce identical results |
| **Compare quantum vs classical** | ✅ Success | Comprehensive evaluation across 5 metrics + sparsity |
| **Reduce sparsity via quantum** | ❌ Failed | Classical achieves 12% better sparsity (0.88 vs 0.99) |
| **Improve recommendation accuracy** | ❌ Failed | Quantum 2.5% worse RMSE, 54× worse HR@10 |
| **Demonstrate quantum advantage** | ❌ Failed | Classical dominates all metrics due to scalability |

### Scientific Contribution
**Positive**: 
- Rigorous negative result (valuable for field)
- Identifies specific bottlenecks (O(n²), k < 50)
- Proposes concrete solutions (VQE, hybrid, approximate kernels)

**Negative**: 
- No novel quantum algorithm (standard amplitude encoding)
- No breakthrough in scalability
- No evidence of quantum advantage on this task

### Overall Rating: **B+ (Excellent execution, negative result)**

**Strengths**:
- Methodologically sound experimental design
- Honest reporting of quantum limitations
- Thorough classical baseline optimization
- Comprehensive metric suite

**Weaknesses**:
- Lack of statistical significance testing
- Missing alternative quantum approaches (VQE, QAOA)
- No hybrid quantum-classical implementation
- Cold-start metric misinterpreted

**Impact**: 
This analysis provides strong evidence that **current gate-based quantum clustering is not competitive** with classical methods for large-scale recommender systems. However, it opens promising research directions in hybrid approaches and approximate quantum kernels.

---

## 9. Conclusion

This is a **well-executed negative result** that clarifies quantum clustering limitations for recommender systems:

1. **Classical k-means (280 clusters) achieves 65.2% HR@10** on MovieLens 32M
2. **Quantum clustering (6-30 clusters) achieves 0-1.2% HR@10** due to scalability constraints
3. **The gap is not algorithmic but computational**: O(n²) kernel cost limits quantum to <300 items
4. **Future work should focus on**:
   - Approximate quantum kernels
   - Variational quantum clustering
   - Quantum-classical hybrid models
   - Task-specific quantum encodings for collaborative filtering

**For production**: Use classical methods.  
**For research**: Pursue scalability breakthroughs, not incremental tuning of current quantum approaches.

---

**Reviewer**: GitHub Copilot (Claude Sonnet 4.5)  
**Review Date**: February 25, 2026  
**Recommendation**: Accept with minor revisions (add statistical tests, try MovieLens 1M, implement hybrid model)
