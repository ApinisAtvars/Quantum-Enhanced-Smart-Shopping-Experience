"""
Fixed Notebook 03: Quantum Clustering Implementation
Reference implementation for embedding into notebook cells

This module provides the CORRECTED versions of functions from notebook 03 with:
1. Memory leak fixes (no K/dmat storage in sweep results)
2. Error handling (proper logging instead of bare except)
3. MDS caching (avoid O(N³) repeated calculations)
4. Type hints for clarity
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score, davies_bouldin_score
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# MEMORY FIX #1: Corrected Evaluation Function
# ============================================================================

def eval_labels_on_distance_FIXED(
    dmat_eval: np.ndarray, 
    labels: np.ndarray,
    seed: int = 42,
    mds_cache: Dict[str, Tuple[float, float]] = None
) -> Tuple[float, float]:
    """
    FIXED: Evaluate clustering quality using precomputed distance matrix.
    
    Memory: No K or dmat stored (only returned scores)
    Error handling: Proper exception logging instead of silent failures
    
    Args:
        dmat_eval: Precomputed distance matrix (N×N)
        labels: Cluster labels (N,)
        seed: Random seed for MDS reproducibility
        mds_cache: Optional MDS embedding cache {hash: (emb, sil, db)}
    
    Returns:
        (silhouette_score, davies_bouldin_score)
    """
    try:
        labels = np.asarray(labels, dtype=np.int32)
        unique_labels = np.unique(labels)
        
        # Validation: require at least 2 clusters
        if len(unique_labels) < 2:
            logger.warning(f"Invalid clustering: only {len(unique_labels)} unique label(s), expected >= 2")
            return -1.0, np.inf
        
        # Check for all-noise (HDBSCAN specific)
        if len(unique_labels) == 1 and unique_labels[0] == -1:
            logger.warning("Clustering returned all noise labels (-1), treating as invalid")
            return -1.0, np.inf
        
        # Silhouette score (efficient, no MDS needed)
        sil = float(silhouette_score(dmat_eval, labels, metric="precomputed"))
        
        # Davies-Bouldin (requires embedding)
        # Use cache if available to avoid repeated MDS O(N³) computation
        cache_key = f"mds_{len(dmat_eval)}_{seed}"
        if mds_cache is not None and cache_key in mds_cache:
            emb2 = mds_cache[cache_key]
            logger.debug(f"Using cached MDS embedding (key={cache_key})")
        else:
            emb2 = MDS(n_components=2, dissimilarity="precomputed", 
                       random_state=seed, n_init=4).fit_transform(dmat_eval)
            if mds_cache is not None:
                mds_cache[cache_key] = emb2
                logger.debug(f"Cached MDS embedding (key={cache_key})")
        
        db = float(davies_bouldin_score(emb2, labels))
        
        return sil, db
        
    except Exception as e:
        logger.error(f"eval_labels_on_distance failed: {type(e).__name__}: {e}")
        return float('nan'), float('nan')


# ============================================================================
# MEMORY FIX #2: Clustering Sweep with Proper Error Handling
# ============================================================================

def run_clustering_sweep_FIXED(
    K: np.ndarray,
    dmat: np.ndarray,
    gamma: float,
    k_range: range,
    seed: int = 42,
    mds_cache: Dict[str, np.ndarray] = None
) -> List[Dict[str, Any]]:
    """
    FIXED: Run clustering sweep across k values and methods.
    
    Memory: DROPS K and dmat from results — store only metrics + labels
    Errors: All exceptions properly logged, not silently swallowed
    
    Args:
        K: Precomputed kernel matrix (N×N)
        dmat: Precomputed distance matrix (N×N)
        gamma: Current gamma parameter value
        k_range: Range of k values to sweep
        seed: Random seed
        mds_cache: Cache dict for MDS embeddings
    
    Returns:
        List of result dicts:
        {
            'method': str,
            'k': int,
            'gamma': float,
            'silhouette': float,
            'db': float,
            'labels': list,  # Store as list, not array
            'error': str (optional)  # If exception occurred
        }
    """
    if mds_cache is None:
        mds_cache = {}
    
    rows = []
    
    for k_try in k_range:
        # ============================================================
        # Method 1: QLloyd with mean update
        # ============================================================
        try:
            # Placeholder for actual quantum Lloyd implementation
            # In real notebook: l1, _, _ = quantum_lloyd_kernel(K, k=k_try, ...)
            l1 = np.random.randint(0, k_try, len(K))  # REPLACE with actual
            
            s1, d1 = eval_labels_on_distance_FIXED(dmat, l1, seed=seed, mds_cache=mds_cache)
            
            rows.append({
                "method": "qlloyd_mean",
                "k": int(k_try),
                "gamma": float(gamma),
                "silhouette": float(s1),
                "db": float(d1),
                "labels": l1.tolist(),  # IMPORTANT: store as list, not array
            })
            
        except Exception as e:
            logger.error(f"QLloyd mean failed at k={k_try}, gamma={gamma}: {type(e).__name__}: {e}")
            rows.append({
                "method": "qlloyd_mean",
                "k": int(k_try),
                "gamma": float(gamma),
                "silhouette": float("nan"),
                "db": float("nan"),
                "labels": [],
                "error": str(e),
            })
        
        # ============================================================
        # Method 2: QLloyd with medoid update
        # ============================================================
        try:
            # l2, _, _ = quantum_lloyd_kernel(K, k=k_try, update_mode="medoid", ...)
            l2 = np.random.randint(0, k_try, len(K))  # REPLACE with actual
            
            s2, d2 = eval_labels_on_distance_FIXED(dmat, l2, seed=seed, mds_cache=mds_cache)
            
            rows.append({
                "method": "qlloyd_medoid",
                "k": int(k_try),
                "gamma": float(gamma),
                "silhouette": float(s2),
                "db": float(d2),
                "labels": l2.tolist(),
            })
            
        except Exception as e:
            logger.error(f"QLloyd medoid failed at k={k_try}, gamma={gamma}: {type(e).__name__}: {e}")
            rows.append({
                "method": "qlloyd_medoid",
                "k": int(k_try),
                "gamma": float(gamma),
                "silhouette": float("nan"),
                "db": float("nan"),
                "labels": [],
                "error": str(e),
            })
        
        # ============================================================
        # Method 3: Agglomerative Clustering
        # ============================================================
        try:
            l3 = AgglomerativeClustering(
                n_clusters=k_try,
                metric="precomputed",
                linkage="average"
            ).fit_predict(dmat)
            
            s3, d3 = eval_labels_on_distance_FIXED(dmat, l3, seed=seed, mds_cache=mds_cache)
            
            rows.append({
                "method": "agglo_average",
                "k": int(k_try),
                "gamma": float(gamma),
                "silhouette": float(s3),
                "db": float(d3),
                "labels": l3.tolist(),
            })
            
        except Exception as e:
            logger.error(f"Agglomerative failed at k={k_try}, gamma={gamma}: {type(e).__name__}: {e}")
            rows.append({
                "method": "agglo_average",
                "k": int(k_try),
                "gamma": float(gamma),
                "silhouette": float("nan"),
                "db": float("nan"),
                "labels": [],
                "error": str(e),
            })
        
        # ============================================================
        # Method 4: Spectral Clustering
        # ============================================================
        try:
            l4 = SpectralClustering(
                n_clusters=k_try,
                affinity="precomputed",
                random_state=seed,
                n_init=8
            ).fit_predict(K)
            
            s4, d4 = eval_labels_on_distance_FIXED(dmat, l4, seed=seed, mds_cache=mds_cache)
            
            rows.append({
                "method": "spectral_precomputed",
                "k": int(k_try),
                "gamma": float(gamma),
                "silhouette": float(s4),
                "db": float(d4),
                "labels": l4.tolist(),
            })
            
        except Exception as e:
            logger.error(f"Spectral failed at k={k_try}, gamma={gamma}: {type(e).__name__}: {e}")
            rows.append({
                "method": "spectral_precomputed",
                "k": int(k_try),
                "gamma": float(gamma),
                "silhouette": float("nan"),
                "db": float("nan"),
                "labels": [],
                "error": str(e),
            })
    
    return rows


# ============================================================================
# MEMORY FIX #3: Main Sweep with Checkpointing
# ============================================================================

def run_full_quantum_sweep_FIXED(
    X_amp: np.ndarray,
    user_latent: np.ndarray,
    out_dir: Path,
    seed: int = 42,
    gamma_points: int = 15,
    k_range_tuple: Tuple[int, int] = (4, 13)
) -> Dict[str, Any]:
    """
    FIXED: Full quantum clustering sweep with checkpointing and proper error handling.
    
    Memory improvements:
    - K and dmat never stored in results (only computed for each gamma, then discarded)
    - MDS embeddings cached by dimension (reused across methods)
    - Labels stored as lists, not numpy arrays
    - Checkpoint saves after each gamma iteration (resumable)
    
    Args:
        X_amp: Amplitude-encoded user vectors (N, D)
        user_latent: Original latent factors (for reference)
        out_dir: Output directory for results
        seed: Random seed
        gamma_points: Number of gamma values to sweep
        k_range_tuple: (k_min, k_max) for sweep
    
    Returns:
        {
            'best_result': dict with best clustering,
            'all_results': list of all sweep results,
            'tuning_df': DataFrame with metrics,
            'mds_cache': MDS embedding cache (reuse for eval)
        }
    """
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate log-spaced gamma grid
    gamma_grid = np.logspace(np.log10(0.5), np.log10(8.0), gamma_points)
    logger.info(f"Gamma grid ({len(gamma_grid)} points): {gamma_grid[0]:.4f} to {gamma_grid[-1]:.4f}")
    
    rows_all = []
    mds_cache = {}  # Global cache for MDS embeddings
    
    # ========================================================================
    # Main sweep loop
    # ========================================================================
    logger.info(f"Starting clustering sweep over {len(gamma_grid)} gamma values and k={k_range_tuple[0]}..{k_range_tuple[1]-1}")
    
    for gamma_idx, gamma_try in enumerate(tqdm(gamma_grid, desc="Gamma sweep", position=0)):
        try:
            # Compute kernel and distance for this gamma
            # PLACEHOLDER: Use your actual kernel computation
            # K_gamma = center_psd_normalize(np.clip((gamma_try * kernel_subspace) ** 3, 0.0, 1.0))
            K_gamma = np.random.randn(len(X_amp), len(X_amp))  # REPLACE with actual
            K_gamma = (K_gamma + K_gamma.T) / 2  # Symmetrize
            K_gamma = np.clip(K_gamma, 0, 1)  # Ensure [0,1]
            
            # Distance matrix from kernel
            diag = np.clip(np.diag(K_gamma), 0.0, None)
            d2 = diag[:, None] + diag[None, :] - 2.0 * K_gamma
            d2 = np.clip(d2, 0.0, None)
            dmat_gamma = np.sqrt(d2)
            
            # Run sweep for this gamma
            sweep_rows = run_clustering_sweep_FIXED(
                K_gamma, dmat_gamma, float(gamma_try),
                range(*k_range_tuple),
                seed=seed,
                mds_cache=mds_cache
            )
            rows_all.extend(sweep_rows)
            
            # Checkpoint every 1/5 of sweep
            if (gamma_idx + 1) % max(1, len(gamma_grid) // 5) == 0:
                checkpoint_df = pd.DataFrame([
                    r for r in rows_all
                    if 'error' not in r or not r.get('error')
                ])
                checkpoint_file = out_dir / f"quantum_tuning_checkpoint_g{gamma_idx:02d}.csv"
                checkpoint_df.to_csv(checkpoint_file, index=False)
                logger.info(f"Checkpoint saved: {checkpoint_file.name} ({len(checkpoint_df)} valid rows)")
        
        except Exception as e:
            logger.error(f"Gamma {gamma_idx} ({gamma_try:.4f}) failed: {type(e).__name__}: {e}")
            # Continue to next gamma instead of crashing
    
    # ========================================================================
    # Identify best result
    # ========================================================================
    valid_results = [r for r in rows_all if 'error' not in r or not r.get('error')]
    
    if not valid_results:
        logger.error("No valid clustering results found!")
        return {'best_result': None, 'all_results': rows_all, 'tuning_df': pd.DataFrame([]), 'mds_cache': mds_cache}
    
    # Sort by silhouette (primary), then Davies-Bouldin (secondary)
    sorted_results = sorted(
        valid_results,
        key=lambda r: (float('nan') if np.isnan(r['silhouette']) else r['silhouette'], 
                      float('inf') if np.isnan(r['db']) else -r['db']),
        reverse=True
    )
    
    best_result = sorted_results[0]
    logger.info(f"Best result: method={best_result['method']}, k={best_result['k']}, "
                f"gamma={best_result['gamma']:.4f}, sil={best_result['silhouette']:.4f}")
    
    # Save all results
    tuning_df = pd.DataFrame(valid_results)
    tuning_df.to_csv(out_dir / "quantum_tuning_sweep_all.csv", index=False)
    logger.info(f"Saved full sweep results ({len(tuning_df)} rows)")
    
    return {
        'best_result': best_result,
        'all_results': rows_all,
        'tuning_df': tuning_df,
        'mds_cache': mds_cache
    }


# ============================================================================
# UTILITY: Error Summary Report
# ============================================================================

def report_sweep_errors(all_results: List[Dict]) -> pd.DataFrame:
    """Generate error summary from sweep results."""
    error_rows = [r for r in all_results if 'error' in r and r.get('error')]
    
    if not error_rows:
        logger.info("No errors found in sweep")
        return pd.DataFrame()
    
    error_df = pd.DataFrame([{
        'method': r.get('method'),
        'k': r.get('k'),
        'gamma': r.get('gamma'),
        'error_type': type(r.get('error')).__name__,
        'error_message': str(r.get('error'))[:100],
    } for r in error_rows])
    
    logger.warning(f"Errors found: {len(error_df)}")
    logger.warning(error_df.to_string())
    
    return error_df


if __name__ == "__main__":
    logger.info("Reference implementation loaded. Use these functions in notebook 03 cells.")
