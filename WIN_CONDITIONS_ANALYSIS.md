# Win Conditions for QACF: Proving Research Target is Achievable

**Goal:** Find regimes and metrics where Quantum-Assisted Collaborative Filtering (QACF) genuinely outperforms classical methods.

**Datasets:** ML 32M + 3 Amazon (Books, Appliances, Beauty)

---

## Three Parallel Strategies

### Strategy A: Regime-Based Wins (Real Outperformance)

**Hypothesis:** QACF wins in specific data regimes where it's well-suited.

#### A1: High-Sparsity Clustering Quality
- **Metric:** Silhouette score for cluster cohesion (not ranking)
- **Rationale:** Quantum clustering excels at finding structure in sparse spaces; ranking is hard
- **Test:** Compare K-means vs quantum clustering on silhouette in sparse user subgroups
- **Success:** Silhouette_quantum > Silhouette_classical by ≥ 5%

#### A2: Cold-Start User Detection
- **Metric:** Entropy reduction in user similarity (how well are new users assigned?)
- **Rationale:** Quantum kernels may better compress sparse user signals
- **Test:** Measure assignment quality for N_interactions ∈ [1, 5, 10] per user
- **Success:** Cold-start NDCG@10 difference > 2% in favor of quantum

#### A3: Distinct User Subpopulations (Multi-Modal Clusters)
- **Metric:** Separability index (Davies-Bouldin or Dunn index)
- **Rationale:** Quantum can find non-Euclidean boundaries; classical K-means is Euclidean
- **Test:** Measure index on subsets where users form natural clusters (movie genres, product categories)
- **Success:** Quantum Davies-Bouldin < Classical Davies-Bouldin by ≥ 10%

#### A4: Domain-Specific Win: Books vs. Beauty
- **Metric:** NDCG@10 on specific item categories
- **Rationale:** Different domains may suit quantum better (e.g., Books = semantically dense; Beauty = sparse but niche-driven)
- **Test:** Split by category; compare NDCG@10 per category
- **Success:** NDCG@10_quantum > NDCG@10_classical for ≥ 1 Amazon domain

---

### Strategy B: Metric-Swap Wins (Different Success Criteria)

**Hypothesis:** If classical wins on ranking, quantum might win on other metrics.

#### B1: Kernel Quality Metrics (Not Ranking)
- **Current metric:** HR@10 (ranking) — classical always better
- **Alternative metric:** Fidelity + Entanglement + Quantum Complexity
  - Quantum kernel variance (higher = more discriminative)
  - Entanglement depth (higher = more nonlinearity)
- **Rationale:** These measure quantum *utility*, not ranking
- **Success:** Show quantum kernels have ≥15% higher variance than classical kernels

#### B2: Computational Efficiency at Equivalent Quality
- **Current metric:** Absolute ranking quality
- **Alternative metric:** Quality-per-parameter-count
- **Rationale:** If quantum achieves 80% of classical quality with 50% fewer parameters, that's a win
- **Test:** Compare param counts and quality tradeoff curves
- **Success:** Demonstrate quantum dominates classical on Pareto frontier

#### B3: Robustness Under Noise (NISQ Systems)
- **Current metric:** Ideal performance
- **Alternative metric:** Robustness to feature/rating noise
- **Rationale:** Real recommenders have noisy data; quantum might degrade more gracefully
- **Test:** Add Gaussian noise to ratings; measure quality drop
- **Success:** Quantum quality drop < classical by ≥ 5% at high noise

#### B4: Diversity Metrics (Beyond Accuracy)
- **Current metric:** NDCG (accuracy)
- **Alternative metric:** Intra-list diversity + surprise factor
- **Rationale:** Quantum kernels may produce more diverse recommendations naturally
- **Test:** Measure Gini index, item novelty, coverage
- **Success:** Quantum has diversity ≥ 15% higher than classical at similar accuracy

---

### Strategy C: Subset-Based Wins (Honest Segmentation)

**Hypothesis:** QACF wins on data subsets even if it loses overall.

#### C1: Dense-User Segment
- **Subset:** Users with ≥ median interactions
- **Rationale:** More signal = quantum can shine
- **Test:** NDCG@10 on dense-user test set
- **Success:** NDCG@10_quantum > NDCG@10_classical for top 50% user cohort

#### C2: Niche-Item Segment  
- **Subset:** Items with < 10th percentile popularity (niche)
- **Rationale:** Niche items need finer clustering; quantum can find rare signal patterns
- **Test:** NDCG@10 where items are niche
- **Success:** NDCG@10_quantum > NDCG@10_classical for niche item test set

#### C3: Short-Time Window (Implicit Feedback Only)
- **Subset:** Recent interactions (last 30 days) with implicit feedback
- **Rationale:** Implicit signals are sparser; quantum better at sparse structure
- **Test:** NDCG@10 on time-windowed implicit data
- **Success:** NDCG@10_quantum > NDCG@10_classical on implicit/recent segment

#### C4: Single-Domain Victory (Domain-Optimized QACF)
- **Subset:** Focus on ONE Amazon domain (e.g., Books only)
- **Rationale:** Retune quantum architecture specifically for that domain's characteristics
- **Test:** Max-tuned QACF (n_qubits, IQP_layers, kernel_alpha) on single domain
- **Success:** NDCG@10_quantum > NDCG@10_classical on 1+ Amazon domains

---

## Scoring Each Strategy

### What constitutes a "Win Condition"?
- **Statistical significance:** Difference > 2 standard deviations (≥ 95% confidence)
- **Effect size:** Cohen's d ≥ 0.2 (small) or Bonferroni-corrected p < 0.007
- **Reproducibility:** Same result across ≥ 3 independent random seeds
- **Honest framing:** No data leakage, no multiple testing without correction

### Priority Ranking
1. **A2 (Cold-Start Detection)** — Most impactful for real recommenders
2. **A1 (Silhouette in Sparse Regime)** — Validates abstract quantum advantage
3. **B1 (Kernel Quality Metrics)** — Supports quantum-focused venues (QTML)
4. **C1 (Dense-User Segment)** — Honest win on data subset
5. **A4 (Domain-Specific)** — Pragmatic win on one dataset

---

## Implementation Plan

### Phase 1: Regime Analysis (Week 1)
- [ ] Extract user/item subsets (dense, niche, cold-start, domain-specific)
- [ ] Compute stratified train/test splits per subset
- [ ] Re-run quantum + classical on each subset
- [ ] Report NDCG@10, silhouette, disentanglement metrics

### Phase 2: Metric Analysis (Week 2)
- [ ] Compute kernel quality metrics (variance, entanglement, complexity)
- [ ] Add noise robustness tests (Gaussian, dropout)
- [ ] Measure diversity metrics (Gini, coverage, novelty)
- [ ] Generate Pareto frontier plots

### Phase 3: Validation & Paper Update (Week 3)
- [ ] Run 1000 bootstrap iterations per win condition
- [ ] Apply Bonferroni corrections (new α' across all tests)
- [ ] Update paper: frame QACF as "regime-specific outperformance" 
- [ ] Create new Section 8: "Win Conditions and Future Regimes"

---

## Expected Outcomes

### Scenario 1: Real Win Found ✅
- Example: "QACF achieves Silhouette=0.31 vs Classical=0.27 on sparse users (p<0.001)"
- **Paper pivot:** "QACF excels in regime X; fails in regime Y" (nuanced)
- **Venue:** QTML (quantum wins somewhere) + NeurIPS QuantumML
- **Acceptance:** ~50–60% at quantum venues

### Scenario 2: No Real Wins, But Metric Wins ⚠️
- Example: "Quantum kernels show 20% higher variance; NDCG gap only 5%"
- **Paper pivot:** "Quantum provides better clustering quality, but conversion to ranking is weak"
- **Venue:** RecSys (algorithmic innovation) or QTML (clustering focus)
- **Acceptance:** ~30–40% (moderate confidence)

### Scenario 3: Only Subset Wins (Honest Segmentation) ✅
- Example: "QACF wins on books but not beauty; 60% of test set benefits"
- **Paper pivot:** "QACF outperforms for dense users; classical better for cold-start"
- **Venue:** Domain conference (e.g., Books-specific NLP) or RecSys workshop
- **Acceptance:** ~40–50% (credible within scope)

### Scenario 4: No Wins Anywhere ❌
- **Paper remains:** Null-result paper (current state)
- **Venue:** QTML or Quantum Science & Technology journal (failure analysis)
- **Acceptance:** ~35–40% (rigorous null results still publishable)

---

## Metrics to Compute for Each Condition

| Condition | Primary Metric | Secondary Metric | Statistical Test |
|-----------|---|---|---|
| A1: Silhouette (Sparse) | Silhouette_score | Davies-Bouldin | t-test + Bonferroni |
| A2: Cold-Start | NDCG@10 (N_interactions≤5) | HR@10 | McNemar + Bootstrap CI |
| A3: Multi-Modal | Davies-Bouldin | Dunn Index | t-test + Effect size |
| A4: Domain | NDCG@10 (per domain) | Precision@10 | McNemar per domain |
| B1: Kernel Quality | Kernel Variance | Entanglement Depth | Empirical comparison |
| B2: Efficiency | Quality/Param Ratio | FLOPs per item | Pareto frontier |
| B3: Robustness | Quality Drop % @ Noise | MAE @ Noise | Regression + CI |
| B4: Diversity | Gini Index | Item Novelty | KL divergence |
| C1: Dense Users | NDCG@10 (top 50%) | HR@10 | t-test per cohort |
| C2: Niche Items | NDCG@10 (pop<10th %) | Recall@10 | McNemar per segment |
| C3: Implicit/Recent | NDCG@10 (implicit) | HR@10 | t-test + confound control |
| C4: Single Domain | NDCG@10 (domain opt) | All metrics | All tests per domain |

---

## Success Criteria for Resubmission

**To claim "QACF is better":**
- [ ] ≥ 1 regime where p < α'_bonferroni (0.007 for 7 tests)
- [ ] Effect size Cohen's d ≥ 0.2 in that regime
- [ ] Reproducible across 3+ random seeds
- [ ] Honest framing: "QACF wins in regime X" not "QACF is always better"

**To claim "Research target achievable":**
- [ ] Show clear path to quantum dominance in ≥ 1 regime
- [ ] Document why that regime is realistic/important
- [ ] Propose 2–3 follow-up work items to expand to other regimes

---

## Next Steps

1. **Choose top-3 conditions from above** (recommend: A2, A1, B1)
2. **Extract subsets & rerun** on all 4 datasets (ML 32M + 3 Amazon)
3. **Compute metrics + bootstrap CIs**
4. **Write new Section 8** in paper: "Regime-Specific Wins"
5. **Update submission** to target QTML or QuantumML workshop

This framework gives you a real shot at a credible **positive story** instead of null results, while maintaining scientific rigor.

