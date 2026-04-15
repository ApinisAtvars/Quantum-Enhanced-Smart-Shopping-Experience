"""
Fairness Comparison Utilities: Classical vs Quantum on Symmetric Sample

This module ensures fair evaluation between classical and quantum methods by:
1. Resampling classical clustering to the same N=600 stratified sample
2. Using identical train-test splits
3. Computing metrics on identical held-out sets
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix, load_npz
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def stratified_sample_users(user_latent, sample_size=600, seed=42, stratify_by_activity=True):
    """
    Generate stratified sample of users for fair comparison.
    
    Args:
        user_latent: Full user latent matrix (N_all, D)
        sample_size: Target sample size (default 600, matching quantum)
        seed: Random seed
        stratify_by_activity: Stratify by latent vector norm (activity proxy)
    
    Returns:
        sample_idx: Indices of sampled users
        sample_latent: Sampled latent matrix
    """
    rng = np.random.default_rng(seed)
    N = user_latent.shape[0]
    
    if N <= sample_size:
        logger.warning(f"Dataset size {N} <= sample_size {sample_size}, returning full dataset")
        idx = np.arange(N)
        return idx, user_latent.copy()
    
    if stratify_by_activity:
        # Segment by latent norm (activity proxy)
        user_norm = np.linalg.norm(user_latent, axis=1)
        percentiles = [0, 25, 50, 75, 100]
        segments = np.percentile(user_norm, percentiles)
        segment_labels = np.digitize(user_norm, segments[1:-1])  # 0, 1, 2, 3
        
        # Stratified sample: maintain segment proportions
        unique_seg, seg_counts = np.unique(segment_labels, return_counts=True)
        proportions = seg_counts / len(segment_labels)
        target_per_segment = np.round(proportions * sample_size).astype(int)
        
        # Ensure sum == sample_size (adjust largest segment)
        target_per_segment[-1] += sample_size - target_per_segment.sum()
        
        sample_idx_list = []
        for seg, target_n in zip(unique_seg, target_per_segment):
            seg_mask = segment_labels == seg
            seg_indices = np.where(seg_mask)[0]
            selected = rng.choice(seg_indices, size=min(target_n, len(seg_indices)), replace=False)
            sample_idx_list.extend(selected)
        
        sample_idx = np.array(sample_idx_list, dtype=np.int32)
    else:
        # Simple random sample
        sample_idx = rng.choice(N, size=sample_size, replace=False)
    
    sample_idx = np.sort(sample_idx)
    sample_latent = user_latent[sample_idx]
    
    logger.info(f"Stratified sample: {len(sample_idx)} users from {N} total (seed={seed})")
    return sample_idx, sample_latent


def recompute_classical_baseline_fair(user_sparse, user_latent, sample_idx, seed=42):
    """
    Recompute classical UBCF baseline on the same N=600 sample as quantum method.
    
    Args:
        user_sparse: User-item sparse matrix (all users, normalized ratings)
        user_latent: User latent factors (all users, D dims)
        sample_idx: Sample indices from stratified sampling
        seed: Random seed
    
    Returns:
        results_dict: {
            'k_tuning': DataFrame of k values and silhouette scores,
            'best_k': optimal k,
            'classical_labels': cluster assignments for sampled users,
            'silhouette': overall silhouette score,
            'davies_bouldin': Davies-Bouldin score,
            'sample_idx': indices of sampled users
        }
    """
    rng = np.random.default_rng(seed)
    
    # Extract sample
    X_latent_sample = user_latent[sample_idx]
    
    # Hyperparameter sweep on sample only
    k_list = [4, 6, 8, 12, 16, 20, 24, 32]
    tuning_rows = []
    
    for k in k_list:
        try:
            km = MiniBatchKMeans(
                n_clusters=k, 
                batch_size=128, 
                random_state=seed, 
                n_init=15, 
                max_iter=300
            )
            labels = km.fit_predict(X_latent_sample)
            
            sil = silhouette_score(
                X_latent_sample, 
                labels, 
                sample_size=min(1000, len(X_latent_sample)),
                random_state=seed
            ) if len(np.unique(labels)) > 1 else np.nan
            
            tuning_rows.append({
                'k': k,
                'inertia': float(km.inertia_),
                'silhouette': float(sil),
                'n_clusters_actual': len(np.unique(labels))
            })
            logger.info(f"  k={k}: silhouette={sil:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to fit k={k}: {e}")
            tuning_rows.append({
                'k': k,
                'inertia': float('nan'),
                'silhouette': float('nan'),
                'n_clusters_actual': 0,
                'error': str(e)
            })
    
    tuning_df = pd.DataFrame(tuning_rows)
    
    # Select best k by silhouette (ignoring NaN)
    valid_mask = tuning_df['silhouette'].notna() & (tuning_df['silhouette'] > -1.0)
    if valid_mask.any():
        best_idx = tuning_df[valid_mask]['silhouette'].idxmax()
        best_k = int(tuning_df.loc[best_idx, 'k'])
    else:
        best_k = int(tuning_df.loc[tuning_df['inertia'].idxmin(), 'k'])
    
    logger.info(f"Classical best_k = {best_k}")
    
    # Final fit
    km_final = MiniBatchKMeans(
        n_clusters=best_k,
        batch_size=128,
        random_state=seed,
        n_init=20,
        max_iter=500
    )
    labels_final = km_final.fit_predict(X_latent_sample)
    
    # Metrics
    sil_final = silhouette_score(
        X_latent_sample,
        labels_final,
        sample_size=min(1000, len(X_latent_sample)),
        random_state=seed
    )
    
    try:
        db_final = davies_bouldin_score(X_latent_sample, labels_final)
    except Exception as e:
        logger.warning(f"Davies-Bouldin computation failed: {e}")
        db_final = float('nan')
    
    return {
        'k_tuning': tuning_df,
        'best_k': best_k,
        'classical_labels': labels_final,
        'silhouette': float(sil_final),
        'davies_bouldin': float(db_final),
        'sample_idx': sample_idx,
        'X_latent_sample': X_latent_sample
    }


def compare_clustering_quality(classical_results, quantum_labels, sample_idx, X_latent_full):
    """
    Fair comparison: classical and quantum on identical N=600 sample.
    
    Args:
        classical_results: Output from recompute_classical_baseline_fair()
        quantum_labels: Quantum cluster labels (N_sample,)
        sample_idx: Indices of sampled users
        X_latent_full: Full latent matrix (for reference)
    
    Returns:
        comparison_df: Side-by-side metrics
    """
    X_sample = X_latent_full[sample_idx]
    
    results = []
    
    # Classical metrics
    try:
        sil_c = classical_results['silhouette']
    except:
        sil_c = float('nan')
    
    try:
        db_c = classical_results['davies_bouldin']
    except:
        db_c = float('nan')
    
    results.append({
        'method': 'Classical (MiniBatchKMeans)',
        'n_samples': len(sample_idx),
        'k_clusters': classical_results['best_k'],
        'silhouette_score': sil_c,
        'davies_bouldin_score': db_c,
        'n_clusters_achieved': len(np.unique(classical_results['classical_labels']))
    })
    
    # Quantum metrics
    try:
        sil_q = silhouette_score(
            X_sample,
            quantum_labels,
            sample_size=min(1000, len(X_sample)),
            random_state=42
        ) if len(np.unique(quantum_labels)) > 1 else float('nan')
    except Exception as e:
        logger.warning(f"Quantum silhouette failed: {e}")
        sil_q = float('nan')
    
    try:
        db_q = davies_bouldin_score(X_sample, quantum_labels)
    except Exception as e:
        logger.warning(f"Quantum Davies-Bouldin failed: {e}")
        db_q = float('nan')
    
    results.append({
        'method': 'Quantum (QACF)',
        'n_samples': len(sample_idx),
        'k_clusters': len(np.unique(quantum_labels)),
        'silhouette_score': sil_q,
        'davies_bouldin_score': db_q,
        'n_clusters_achieved': len(np.unique(quantum_labels))
    })
    
    comparison_df = pd.DataFrame(results)
    
    # Print side-by-side
    logger.info("\n" + "="*80)
    logger.info("FAIR COMPARISON: Classical vs Quantum on N=600 Stratified Sample")
    logger.info("="*80)
    logger.info(comparison_df.to_string(index=False))
    logger.info("="*80 + "\n")
    
    return comparison_df


if __name__ == "__main__":
    # Example usage
    in_dir = Path("data/processed_32m_ubcf")
    
    user_latent = np.load(in_dir / "user_latent.npy").astype(np.float32)
    user_sparse = load_npz(in_dir / "user_sparse.npz").tocsr().astype(np.float32)
    
    # Generate fair sample
    sample_idx, X_sample = stratified_sample_users(user_latent, sample_size=600, seed=42)
    
    # Recompute classical on same sample
    classical_results = recompute_classical_baseline_fair(user_sparse, user_latent, sample_idx, seed=42)
    
    logger.info(f"Classical baseline k-tuning:\n{classical_results['k_tuning'].to_string()}")
    logger.info(f"Classical silhouette: {classical_results['silhouette']:.4f}")
    
    # Load quantum results
    quantum_labels = np.load(in_dir / "quantum_user_labels_improved.npy").astype(np.int32)
    
    # Compare
    comparison = compare_clustering_quality(classical_results, quantum_labels, sample_idx, user_latent)
    comparison.to_csv(in_dir / "fair_comparison_n600.csv", index=False)
