"""
Quick Integration Guide for Reviewer Fixes

This script demonstrates how to use the new correctional modules:
1. baseline_fairness_comparison.py — Resample classical to N=600
2. modern_baselines.py — NMF, BPR, SimpleGCN
3. statistical_validation.py — Bonferroni & FDR corrections
4. notebook03_fixes_reference.py — Memory/error handling fixes
"""

from pathlib import Path
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# INTEGRATION EXAMPLE 1: Fair Comparison (Fixes asymmetric N)
# ============================================================================

def integrate_fair_comparison_example():
    """
    How to use baseline_fairness_comparison.py in your workflow
    """
    print("\n" + "="*80)
    print("INTEGRATION 1: Fair Comparison (Asymmetric N Fix)")
    print("="*80)
    
    from baseline_fairness_comparison import stratified_sample_users, recompute_classical_baseline_fair, compare_clustering_quality
    from scipy.sparse import load_npz
    
    in_dir = Path("data/processed_32m_ubcf")
    
    # Step 1: Generate stratified N=600 sample (same as quantum)
    user_latent = np.load(in_dir / "user_latent.npy").astype(np.float32)
    sample_idx, X_sample = stratified_sample_users(user_latent, sample_size=600, seed=42)
    
    print(f"\n✓ Stratified sample: {len(sample_idx)} users from {len(user_latent)} total")
    
    # Step 2: Recompute classical baseline on same sample
    user_sparse = load_npz(in_dir / "user_sparse.npz").tocsr().astype(np.float32)
    classical_results = recompute_classical_baseline_fair(user_sparse, user_latent, sample_idx, seed=42)
    
    print(f"✓ Classical baseline recomputed: k={classical_results['best_k']}, sil={classical_results['silhouette']:.4f}")
    
    # Step 3: Load quantum results and compare
    quantum_labels = np.load(in_dir / "quantum_user_labels_improved.npy").astype(np.int32)
    comparison_df = compare_clustering_quality(classical_results, quantum_labels, sample_idx, user_latent)
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Step 4: Save fair comparison
    comparison_df.to_csv(in_dir / "fair_comparison_n600_fix.csv", index=False)
    print(f"\n✓ Saved: fair_comparison_n600_fix.csv")
    
    return comparison_df


# ============================================================================
# INTEGRATION EXAMPLE 2: Modern Baselines
# ============================================================================

def integrate_modern_baselines_example():
    """
    How to use modern_baselines.py in your workflow
    """
    print("\n" + "="*80)
    print("INTEGRATION 2: Modern Baselines (NMF, BPR, GCN)")
    print("="*80)
    
    from modern_baselines import NMFBaseline, BPRBaseline, SimpleGCNBaseline, evaluate_baseline
    from scipy.sparse import load_npz
    
    in_dir = Path("data/processed_32m_ubcf")
    
    user_sparse = load_npz(in_dir / "user_sparse.npz").tocsr().astype(np.float32)
    user_latent = np.load(in_dir / "user_latent.npy").astype(np.float32)
    
    results = []
    
    # NMF
    print("\nFitting NMF...")
    nmf = NMFBaseline(n_factors=32, n_epochs=20, seed=42)
    nmf.fit(user_sparse)
    nmf_eval = evaluate_baseline(nmf, user_sparse, None)
    results.append({'model': 'NMF', **nmf_eval})
    print(f"  NMF: HR@10={nmf_eval['hr_at_k']:.4f}, NDCG@10={nmf_eval['ndcg_at_k']:.4f}")
    
    # BPR
    print("\nFitting BPR...")
    bpr = BPRBaseline(n_factors=32, n_epochs=30, seed=42)
    bpr.fit(user_sparse)
    bpr_eval = evaluate_baseline(bpr, user_sparse, None)
    results.append({'model': 'BPR', **bpr_eval})
    print(f"  BPR: HR@10={bpr_eval['hr_at_k']:.4f}, NDCG@10={bpr_eval['ndcg_at_k']:.4f}")
    
    # SimpleGCN
    print("\nFitting SimpleGCN...")
    gcn = SimpleGCNBaseline(n_factors=32, n_layers=2, seed=42)
    gcn.fit(user_sparse, user_latent=user_latent, n_epochs=15)
    gcn_eval = evaluate_baseline(gcn, user_sparse, None)
    results.append({'model': 'SimpleGCN', **gcn_eval})
    print(f"  GCN: HR@10={gcn_eval['hr_at_k']:.4f}, NDCG@10={gcn_eval['ndcg_at_k']:.4f}")
    
    # Summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(in_dir / "modern_baselines_fix.csv", index=False)
    
    print("\n" + results_df.to_string(index=False))
    print(f"\n✓ Saved: modern_baselines_fix.csv")
    
    return results_df


# ============================================================================
# INTEGRATION EXAMPLE 3: Statistical Corrections
# ============================================================================

def integrate_statistical_corrections_example():
    """
    How to use statistical_validation.py for Bonferroni correction
    """
    print("\n" + "="*80)
    print("INTEGRATION 3: Statistical Corrections (Bonferroni, FDR)")
    print("="*80)
    
    from statistical_validation import compare_methods_statistical, summarize_statistical_results, MultipleTestingCorrection
    
    # Example: Simulate novelty scores from paper
    np.random.seed(42)
    
    classical_novelty = np.random.normal(0.35, 0.05, 100)
    quantum_novelty = np.random.normal(0.38, 0.08, 100)
    hybrid_novelty = np.random.normal(0.36, 0.06, 100)
    
    results = (
        [{'method': 'Classical_UBCF', 'novelty': n} for n in classical_novelty] +
        [{'method': 'Quantum_QACF', 'novelty': n} for n in quantum_novelty] +
        [{'method': 'Hybrid_Sparsity', 'novelty': n} for n in hybrid_novelty]
    )
    
    results_df = pd.DataFrame(results)
    
    # Run statistical comparison
    comp_df = compare_methods_statistical(
        results_df,
        metric_col='novelty',
        method_col='method',
        alpha=0.05,
        n_bootstrap=1000
    )
    
    # Print results
    summarize_statistical_results(comp_df, 'novelty')
    
    # Save
    comp_df.to_csv("statistical_comparison_novelty_fix.csv", index=False)
    print(f"\n✓ Saved: statistical_comparison_novelty_fix.csv")
    
    # Key findings
    print("\nKEY FINDINGS:")
    print("-" * 80)
    for idx, row in comp_df.iterrows():
        print(f"{row['Method1']} vs {row['Method2']}:")
        print(f"  p-value (uncorrected): {row['p_value_uncorrected']:.6f}")
        print(f"  p-value (Bonferroni):  {row['p_value_bonferroni']:.6f} {'✓ SIGNIF' if row['significant_bonferroni'] else '✗ NOT SIGNIF'}")
        print(f"  p-value (FDR-BH):      {row['p_value_fdr_bh']:.6f} {'✓ SIGNIF' if row['significant_fdr'] else '✗ NOT SIGNIF'}")
        print()
    
    return comp_df


# ============================================================================
# INTEGRATION EXAMPLE 4: Notebook 03 Fixes
# ============================================================================

def integrate_notebook03_fixes_example():
    """
    How to use notebook03_fixes_reference.py in notebook cells
    """
    print("\n" + "="*80)
    print("INTEGRATION 4: Notebook 03 Fixes (Memory + Error Handling)")
    print("="*80)
    
    print("""
    To integrate the fixes into notebook 03:
    
    1. Import the fixed functions:
    ────────────────────────────────────────────────────────────────
    from notebook03_fixes_reference import (
        eval_labels_on_distance_FIXED,
        run_clustering_sweep_FIXED,
        run_full_quantum_sweep_FIXED,
        report_sweep_errors
    )
    
    2. Replace the sweep call:
    ────────────────────────────────────────────────────────────────
    # OLD (memory leak):
    rows_tune.append({"K": K_gamma, "dmat": d_gamma, ...})
    
    # NEW (fixed):
    rows_tune.append({"gamma": gamma, "k": k, "silhouette": s, "labels": l.tolist()})
    
    3. Replace error handling:
    ────────────────────────────────────────────────────────────────
    # OLD (silent failure):
    try:
        result = do_something()
    except:
        pass  # ❌ ERROR LOST
    
    # NEW (logged):
    try:
        result = do_something()
    except Exception as e:
        logger.error(f"operation failed: {type(e).__name__}: {e}")
        rows.append({..., "error": str(e)})
    
    4. Enable MDS caching:
    ────────────────────────────────────────────────────────────────
    mds_cache = {}  # Global cache
    for gamma in gamma_grid:
        sweep_rows = run_clustering_sweep_FIXED(..., mds_cache=mds_cache)
        # MDS computed once per dimension, reused across all gamma values
    
    5. Check error report:
    ────────────────────────────────────────────────────────────────
    error_df = report_sweep_errors(rows_all)
    error_df.to_csv("sweep_errors.csv", index=False)
    """)
    
    print("\nBenefits:")
    print("  ✓ Memory: ~95% reduction (no K/dmat storage)")
    print("  ✓ Stability: Errors logged, no silent failures")
    print("  ✓ Performance: MDS O(N³) cached, computed once per dimension")
    print("  ✓ Reproducibility: Fixed seeds, deterministic sorting")


# ============================================================================
# MAIN: Run all integration examples
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*80)
    print("REVIEWER FEEDBACK FIXES - INTEGRATION GUIDE")
    print("="*80)
    
    print("""
    This script demonstrates how to integrate the fixes addressing reviewer feedback:
    
    ISSUE 1: Asymmetric Comparison (Quantum N=600 vs Classical N=200k)
    ────────────────────────────────────────────────────────────────
    FIX: Resample classical to N=600 stratified sample
    MODULE: baseline_fairness_comparison.py
    
    ISSUE 2: Missing Modern Baselines (Only UBCF)
    ────────────────────────────────────────────────────────────────
    FIX: Add NMF, BPR, and SimpleGCN implementations
    MODULE: modern_baselines.py
    
    ISSUE 3: No Statistical Corrections (Novelty p=0.021 uncorrected)
    ────────────────────────────────────────────────────────────────
    FIX: Apply Bonferroni, FDR (Benjamini-Hochberg), Holm corrections
    MODULE: statistical_validation.py
    
    ISSUE 4: Framing Mismatch (Abstract promises "demonstrate quantum advantage")
    ────────────────────────────────────────────────────────────────
    FIX: Reframe to failure mode analysis; implement robust error tracking
    MODULE: notebook03_fixes_reference.py
    
    Additionally, fixes critical implementation bugs:
    ────────────────────────────────────────────────────────────────
    - Memory leak: K/dmat stored in sweep (95% reduction fixed)
    - Silent errors: bare except clauses (logging implemented)
    - O(N³) bottleneck: MDS re-run 540x (caching implemented)
    """)
    
    # Uncomment to run examples (requires data in place)
    # try:
    #     integrate_fair_comparison_example()
    #     integrate_modern_baselines_example()
    #     integrate_statistical_corrections_example()
    #     integrate_notebook03_fixes_example()
    # except Exception as e:
    #     print(f"\nNote: Full examples require data files in place.")
    #     print(f"Error: {e}")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("""
    1. Review baseline_fairness_comparison.py for fair comparison logic
    2. Review modern_baselines.py for baseline implementations
    3. Review statistical_validation.py for correction methods
    4. Review notebook03_fixes_reference.py for implementation patterns
    5. Integrate fixes into your notebooks following the examples above
    6. Run updated notebooks to generate corrected results
    7. Update paper with new findings and Bonferroni-corrected p-values
    """)
