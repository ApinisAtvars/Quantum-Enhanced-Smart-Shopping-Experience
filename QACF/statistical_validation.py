"""
Statistical Validation with Multiple Testing Corrections

Implements:
1. Bonferroni correction (strict, controls FWER)
2. FDR correction (Benjamini-Hochberg, less conservative)
3. Bootstrap confidence intervals
4. Effect size reporting (Cohen's d, etc.)
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MultipleTestingCorrection:
    """
    Statistical corrections for multiple hypothesis tests.
    """
    
    @staticmethod
    def bonferroni(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Bonferroni correction: multiply each p-value by number of tests.
        Controls Family-Wise Error Rate (FWER).
        
        Args:
            p_values: Array of p-values
            alpha: Significance level (default 0.05)
        
        Returns:
            (corrected_p_values, reject_null, corrected_alpha)
        """
        n_tests = len(p_values)
        corrected_alpha = alpha / n_tests
        corrected_p = np.minimum(p_values * n_tests, 1.0)
        reject = corrected_p < alpha
        
        return corrected_p, reject, corrected_alpha
    
    @staticmethod
    def fdr_bh(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Benjamini-Hochberg False Discovery Rate correction.
        Less conservative than Bonferroni; controls FDR instead of FWER.
        
        Args:
            p_values: Array of p-values
            alpha: Significance level (default 0.05)
        
        Returns:
            (adjusted_p_values, reject_null, effective_alpha)
        """
        n_tests = len(p_values)
        order = np.argsort(p_values)
        sorted_p = p_values[order]
        
        # Find largest i such that P(i) <= (i/m) * alpha
        thresholds = (np.arange(1, n_tests + 1) / n_tests) * alpha
        valid_idx = np.where(sorted_p <= thresholds)[0]
        
        if len(valid_idx) > 0:
            i_max = valid_idx[-1]
            threshold = thresholds[i_max]
            reject_sorted = sorted_p <= threshold
        else:
            threshold = 0.0
            reject_sorted = np.zeros(n_tests, dtype=bool)
        
        # Invert to original order
        reject = np.empty(n_tests, dtype=bool)
        reject[order] = reject_sorted
        
        # Adjusted p-values
        adjusted_p = np.ones(n_tests)
        for i, idx in enumerate(order):
            adjusted_p[idx] = min(
                sorted_p[i] * n_tests / (i + 1),  # BH adjustment
                1.0
            )
        
        return adjusted_p, reject, threshold
    
    @staticmethod
    def holm(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Holm-Bonferroni method: step-down procedure (less conservative than standard Bonferroni).
        """
        n_tests = len(p_values)
        order = np.argsort(p_values)
        sorted_p = p_values[order]
        
        reject_sorted = np.zeros(n_tests, dtype=bool)
        for i in range(n_tests):
            threshold = alpha / (n_tests - i)
            if sorted_p[i] < threshold:
                reject_sorted[i:] = True
                break
        
        reject = np.empty(n_tests, dtype=bool)
        reject[order] = reject_sorted
        
        adjusted_p = np.ones(n_tests)
        for i, idx in enumerate(order):
            adjusted_p[idx] = min(
                sorted_p[i] * (n_tests - i),
                1.0
            )
        
        return adjusted_p, reject


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95, 
                  statistic_fn = None, seed: int = 42) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for any statistic.
    
    Args:
        data: Input data array
        n_bootstrap: Number of bootstrap resamples
        ci: Confidence interval (0.95 = 95%)
        statistic_fn: Function to compute statistic (default: mean)
        seed: Random seed
    
    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    if statistic_fn is None:
        statistic_fn = np.mean
    
    rng = np.random.default_rng(seed)
    point_est = statistic_fn(data)
    
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        resample = rng.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic_fn(resample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    alpha = 1.0 - ci
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return point_est, ci_lower, ci_upper


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Cohen's d effect size: standardized mean difference.
    
    d = (mean1 - mean2) / pooled_std
    
    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def compare_methods_statistical(results_df: pd.DataFrame, metric_col: str, 
                                method_col: str = 'method', alpha: float = 0.05,
                                n_bootstrap: int = 1000) -> pd.DataFrame:
    """
    Comprehensive statistical comparison of methods on a single metric.
    
    Args:
        results_df: DataFrame with columns [method, metric, ...]
        metric_col: Column name of the metric
        method_col: Column name of method labels
        alpha: Significance level
        n_bootstrap: Bootstrap iterations
    
    Returns:
        comparison_df: Statistical test results with corrections
    """
    methods = results_df[method_col].unique()
    n_comparisons = len(methods) * (len(methods) - 1) // 2
    
    comparison_rows = []
    p_values_raw = []
    
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1:]:
            data1 = results_df[results_df[method_col] == m1][metric_col].values
            data2 = results_df[results_df[method_col] == m2][metric_col].values
            
            # Remove NaN
            data1 = data1[~np.isnan(data1)]
            data2 = data2[~np.isnan(data2)]
            
            if len(data1) == 0 or len(data2) == 0:
                continue
            
            # T-test
            t_stat, p_val = stats.ttest_ind(data1, data2)
            p_values_raw.append(p_val)
            
            # Effect size
            d = cohens_d(data1, data2)
            
            # Bootstrap CI for difference in means
            diff_data = data1.mean() - data2.mean()
            _, ci_lower, ci_upper = bootstrap_ci(
                np.concatenate([data1 - data1.mean(), data2 - data2.mean()]),
                n_bootstrap=n_bootstrap
            )
            
            comparison_rows.append({
                'Method1': m1,
                'Method2': m2,
                f'{metric_col}_mean_1': float(data1.mean()),
                f'{metric_col}_std_1': float(data1.std()),
                f'{metric_col}_mean_2': float(data2.mean()),
                f'{metric_col}_std_2': float(data2.std()),
                'mean_diff': float(diff_data),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'cohens_d': float(d),
                't_statistic': float(t_stat),
                'p_value_uncorrected': float(p_val),
                'n_samples_1': len(data1),
                'n_samples_2': len(data2)
            })
    
    comp_df = pd.DataFrame(comparison_rows)
    
    if len(comp_df) == 0:
        logger.warning("No valid comparisons found")
        return comp_df
    
    # Apply corrections
    p_arr = comp_df['p_value_uncorrected'].values
    
    bonf_p, bonf_reject, bonf_alpha = MultipleTestingCorrection.bonferroni(p_arr, alpha)
    fdr_p, fdr_reject, _ = MultipleTestingCorrection.fdr_bh(p_arr, alpha)
    holm_p, holm_reject = MultipleTestingCorrection.holm(p_arr, alpha)
    
    comp_df['p_value_bonferroni'] = bonf_p
    comp_df['p_value_fdr_bh'] = fdr_p
    comp_df['p_value_holm'] = holm_p
    comp_df['significant_bonferroni'] = bonf_reject
    comp_df['significant_fdr'] = fdr_reject
    comp_df['significant_holm'] = holm_reject
    comp_df['corrected_alpha_bonf'] = bonf_alpha
    
    return comp_df


def summarize_statistical_results(comp_df: pd.DataFrame, metric_col: str) -> None:
    """Print formatted statistical comparison summary."""
    logger.info("\n" + "="*100)
    logger.info(f"STATISTICAL COMPARISON: {metric_col.upper()}")
    logger.info("="*100)
    
    for idx, row in comp_df.iterrows():
        logger.info(f"\n{row['Method1']} vs {row['Method2']}:")
        logger.info(f"  Mean diff: {row['mean_diff']:.6f} [{row['ci_lower']:.6f}, {row['ci_upper']:.6f}]")
        logger.info(f"  Cohen's d: {row['cohens_d']:.4f}", end="")
        if abs(row['cohens_d']) < 0.2:
            logger.info(" (negligible)")
        elif abs(row['cohens_d']) < 0.5:
            logger.info(" (small)")
        elif abs(row['cohens_d']) < 0.8:
            logger.info(" (medium)")
        else:
            logger.info(" (large)")
        
        logger.info(f"  p-value (uncorrected): {row['p_value_uncorrected']:.6f}", end="")
        if row['p_value_uncorrected'] < 0.05:
            logger.info(" ✓ SIGNIFICANT")
        else:
            logger.info(" ✗ NOT SIGNIFICANT")
        
        logger.info(f"  p-value (Bonferroni): {row['p_value_bonferroni']:.6f}", end="")
        if row['significant_bonferroni']:
            logger.info(" ✓")
        else:
            logger.info(" ✗")
        
        logger.info(f"  p-value (FDR-BH): {row['p_value_fdr_bh']:.6f}", end="")
        if row['significant_fdr']:
            logger.info(" ✓")
        else:
            logger.info(" ✗")
    
    logger.info("\n" + "="*100)


if __name__ == "__main__":
    # Example: Testing novelty across 3 methods
    np.random.seed(42)
    
    # Simulate novelty scores (lower is better)
    classical_novelty = np.random.normal(0.35, 0.05, 1000)
    quantum_novelty = np.random.normal(0.38, 0.08, 1000)
    hybrid_novelty = np.random.normal(0.36, 0.06, 1000)
    
    results = (
        [{'method': 'Classical_UBCF', 'novelty': n} for n in classical_novelty] +
        [{'method': 'Quantum_QACF', 'novelty': n} for n in quantum_novelty] +
        [{'method': 'Hybrid_Sparsity', 'novelty': n} for n in hybrid_novelty]
    )
    
    results_df = pd.DataFrame(results)
    
    # Compare
    comp_df = compare_methods_statistical(
        results_df,
        metric_col='novelty',
        method_col='method',
        alpha=0.05,
        n_bootstrap=1000
    )
    
    summarize_statistical_results(comp_df, 'novelty')
    
    logger.info("\nComparison DataFrame:")
    logger.info(comp_df.to_string())
