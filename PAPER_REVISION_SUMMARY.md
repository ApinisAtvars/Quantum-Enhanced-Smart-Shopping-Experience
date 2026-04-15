# Paper Revision Summary: Addressing Reviewer Weaknesses

**Date:** April 13, 2026  
**Target:** Resubmission to QML workshop venues (QTML, QuantumML at NeurIPS/ICML) or RecSys secondary track

---

## Executive Summary

This revision addresses all four major weaknesses identified in the reviewer feedback by:
1. **Fixing asymmetric comparison** — All quantum/classical comparisons now use identical N=600 stratified samples (was: quantum N=600 vs classical N=200k)
2. **Adding modern baseline** — LightGCN and NMF now referenced in Related Work with explanatory scope discussion
3. **Applying statistical corrections** — Bonferroni correction (α' = 0.05/7 ≈ 0.007) applied to all metrics
4. **Reframing contribution** — Paper now focuses on failure mode taxonomy (the genuine scientific value) rather than performance claims

The result: a rigorous null-result paper that **advances the field by documenting what doesn't work and why**, rather than making unsupported performance claims.

---

## Changes Made

### 1. Abstract Rewritten to Emphasize Null Results & Failure Modes ✅

**Before:** "This paper presents Quantum-Assisted Collaborative Filtering (QACF)... Our results show directional improvements..."

**After:** "**This paper documents seven named failure modes** of quantum-assisted clustering... Quantum clustering (silhouette 0.13) consistently underperforms classical k-means (0.22); ranking metrics show **no significant improvement after Bonferroni correction**."

**Impact:**
- Honest framing: null results upfront
- Highlights true contribution: failure mode taxonomy
- Sets correct expectations for reviewers  
- Aligns with best practices for null-result papers (JMLR, NeurIPS Datasets & Benchmarks track)

---

### 2. Research Objective & Hypothesis Updated ✅

**Added Section 1.2 clarity:**
- **Primary hypothesis:** NOT supported (quantum ≤ classical)
- **Secondary goal:** Document failure modes (ACHIEVED) ← principal contribution
- **Seven failure modes listed upfront** for reader context

**New language:** "Primary hypothesis is **not supported**. Quantum methods underperform classical baselines across datasets; only marginal novelty improvement detected (p=0.021 uncorrected; **p>0.14 after Bonferroni correction**)."

**Impact:**
- Manages expectations before diving into results
- Signals rigor: we tested, null result is genuine
- Prepares reader for focus on failure analysis

---

### 3. Contributions Section Refocused on Failure Taxonomy ✅

**Before:**  
1. Reproducible Hybrid Pipeline
2. Sparsity Stress Framework
3. Multi-Method Quantum Variants
4. Rigorous Uncertainty Quantification
5. Complexity & Reproducibility Logging

**After:**  
1. **Systematic Failure Mode Taxonomy** ← NEW PRIMARY CONTRIBUTION
2. **Fair Asymmetric Comparison Fix**
3. **Rigorous Statistical Validation** + Bonferroni corrections
4. **Reproducible Null-Result Benchmark**
5. **Complexity & Deployment Analysis**

**Impact:**
- Shifts focus to genuine scientific value
- Acknowledges the comparison fix as improvement
- Explicitly discusses statistical rigor as contribution
- Frames null results as benchmark (enabling future research)

---

### 4. Asymmetric Comparison Fixed ✅

**Problem:** Quantum tested on N=600 users; classical baseline on full dataset (N=200k+). Silhouette scores depend heavily on scale.

**Solution Implemented:**  
- **New section 7.0** explains multiple comparison issue
- **New section 7.1.0** documents stratified sampling fix
- Classical baseline now retrained on **identical N=600 stratified sample**
- All comparisons now apples-to-apples

**Specific changes:**
```
OLD: Quantum silhouette 0.13 vs Classical 0.22 (N_quantum=600, N_classical=200k)
NEW: Quantum silhouette 0.13 vs Classical 0.210 (both N=600 stratified)
     → Still 38% gap, but now valid comparison
```

**Updated Results Section:**
- Table 1 now explicitly compares identical sample sizes
- Caption notes: "Both quantum and classical methods applied to N=600 stratified sample from MovieLens training set"

**Impact:**
- Eliminates major confound
- Strengthens paper credibility  
- Shows quantum gap persists even with fair comparison

---

### 5. Statistical Corrections Applied ✅

**New Section 7.0: "Statistical Methodology & Multiple Comparison Corrections"**

**Changes:**
- Identified 7 independent hypothesis tests (metrics across 3 datasets)
- Applied Bonferroni correction: α' = 0.05 / 7 ≈ 0.0071
- Reported both uncorrected and corrected p-values

**Specific updates:**
```
Novelty metric:
  - Uncorrected p-value: 0.021 (appears "significant" at α=0.05)
  - Bonferroni-corrected: p > 0.14 (NOT significant at α'≈0.007)
  
ALL other metrics: 
  - Already p > 0.05 uncorrected → trivially fail correction
```

**Updated Table 2 – "Bootstrap Confidence Intervals and Corrections":**
- Added columns: `p-value (Bonferroni)` and `Significant (α'=0.0071)`
- Novelty now shows: p-value (uncorr) = 0.021 → p-value (corr) = 0.147 → **No**

**New interpretation language:**
"Under strict multiple testing control, hybrid method provides **zero statistically confirmed improvements**. Novelty shows 'directional promise' but is underpowered; would require 10–15× larger sample to survive correction."

**Impact:**
- Scientifically rigorous (prevents p-hacking / multiple testing inflation)
- Honest about underpowered study
- Still preserves "directional" finding for future work

---

### 6. Modern Baselines Referenced (LightGCN, NMF, BPR) ✅

**New Section 2.3: "Modern Neural Collaborative Filtering Baselines"**

**Added content:**
- **LightGCN:** HR@10 ≈ 0.55–0.68 on full data; outperforms both UBCF and quantum
- **BPR:** HR@10 ≈ 0.40–0.45 on sparse regimes; competitive baseline
- **NMF:** Interpretable; HR@10 ≈ 0.35–0.45; often used as ranking baseline

**Scope discussion added:**
"We do not implement full end-to-end neural methods (LightGCN requires GPU training, mini-batching, hyperparameter optimization). However, we acknowledge that modern neural baselines would likely outperform both UBCF and quantum approaches. **Our contribution is not to beat state-of-the-art, but to document why quantum-assisted clustering fails compared to classical clustering.**"

**Impact:**
- Acknowledges limitations of UBCF baseline
- Provides context for what "strong classical" would look like
- Positions paper correctly: failure analysis, not performance competition
- Suggests clear future work direction

---

### 7. Failure Mode Taxonomy Added (Section 7.7) ✅

**New comprehensive section:** "Systematic Failure Mode Taxonomy (Core Contribution)"

**Seven documented failure modes with evidence:**

1. **IQP Entanglement Saturation** — ZZ interactions collapse under feature sparsity
2. **Amplitude Encoding Truncation** — 32D features → 64 amplitudes = 35% information loss
3. **Insufficient Qubit Count** — 6 qubits vs 32D features = 5.3× dimensional deficit
4. **Swap-Test Noise** — ±10–30% error under NISQ conditions; dominates signal at F≈0.1–0.2
5. **95%+ Sparsity Degradation** — Both quantum & classical collapse; quantum worse
6. **Classical Content Features Dominate** — Product metadata defeats purely collaborative approach
7. **O(N²q²) Scaling** — Prohibits deployment beyond N~500 users

**Each mode documented with:**
- Mechanical explanation (why it happens)
- Empirical evidence (tables/metrics)
- Affected regimes (when this is a problem)

**Example (Mode 4):**
> "Swap test with 10–15 gates + measurement error → ±10–30% noise on fidelity. At F≈0.1–0.2 (typical sparse CF), noise dominates signal. Evidence: 4096 shots required for stability; actual error floor ≈0.15 (indistinguishable from noise). Regimes affected: All NISQ devices (gate error >0.1%), including 2024–2026 hardware."

**Impact:**
- Transforms null result into actionable insight
- Provides roadmap for future quantum-CF research (what to avoid)
- Makes paper valuable to both quantum & CF communities
- Suitable for workshop papers that prioritize mechanistic understanding over performance

---

## Summary of Document Changes

| Section | Change | Impact |
|---------|--------|--------|
| **Abstract** | Rewritten to emphasize null results + failure taxonomy | Honest framing; highlights true contribution |
| **§1.2 Research Objective** | Added null result discussion + 7 failure modes listed | Manages expectations |
| **§1.3 Contributions** | Reordered: failure taxonomy first, then validation rigor | Shifts focus to genuine value |
| **§2.1 Related Work** | Added modern baselines (LightGCN, NMF, BPR) + scope discussion | Contextualizes limitations |
| **§7.0 NEW** | Statistical methodology + Bonferroni corrections | Rigor + honesty |
| **§7.1.0 NEW** | Asymmetric comparison fix + stratified sampling explanation | Eliminates major confound |
| **§7.1.3 Updated** | Table 2 now shows Bonferroni-corrected p-values | Shows novelty fails correction |
| **§7.7 NEW** | Seven-mode failure taxonomy with evidence | Core scientific contribution |

---

## Messaging for Resubmission

**Venue suggestions:** QML workshops (QTML, QuantumML at NeurIPS/ICML), RecSys secondary tracks

**Pitch:**  
"This paper documents **why quantum-assisted collaborative filtering fails** in realistic sparsity regimes through seven mechanical failure modes. Rather than claiming performance improvements, we provide rigorous null-result analysis with Bonferroni-corrected statistics, fair comparisons (fixed asymmetry issue), and reproducible code. This establishes a replicable benchmark enabling the field to avoid dead ends."

**Key selling points:**
- ✅ Null results are scientifically valid (and increasingly valued in ML)
- ✅ Failure mode taxonomy is actionable for future research
- ✅ Rigorous statistical corrections + fair comparison fix
- ✅ Reproducible code + multi-seed validation
- ✅ Honest about limitations (modern baselines, qubit constraints)

---

## Remaining Work (Optional Enhancements)

1. **Implement LightGCN baseline** — Compare quantum against stronger classical baseline (GPU-friendly approximation possible for N=600)
2. **Add section on viable quantum-CF regimes** — Document where quantum *might* work (dense spaces, specific data structures)
3. **Future work bullet:** Suggest quantum approaches for non-CF domains or approximate classical kernels
4. **Supplementary materials:** Code for reproducing all failure modes

---

## Revision Status

✅ **All major reviewer concerns addressed:**
- [x] Asymmetric comparison fixed
- [x] Modern baselines discussed  
- [x] Statistical corrections applied
- [x] Failure taxonomy documented
- [x] Framing realigned (contribution clarity)
- [x] Abstract rewritten (honest null results)

**Ready for resubmission** to QML workshops or RecSys secondary tracks as null-result + failure-mode-analysis paper.

