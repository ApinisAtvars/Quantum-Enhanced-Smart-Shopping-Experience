"""
Shared Fixes Module for All Datasets (UBCF, Amazon, Fashion, Appliances)

This module is imported by all notebooks across all datasets.
It contains the corrected implementations for:
1. Fair comparison (baseline_fairness_comparison)
2. Modern baselines (baseline implementations)
3. Statistical validation (Bonferroni, FDR corrections)
4. Memory-efficient notebook 03 functions
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, MiniBatchKMeans
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.sparse import csr_matrix
import warnings

warnings.filterwarnings('ignore')

# Quick logging config for all datasets
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# FIX #1: Fair Comparison — Stratified Sampling
# ============================================================================

def get_stratified_sample(data_matrix, sample_size=600, seed=42):
    """
    Generate stratified sample for fair comparison across datasets.
    Works for numpy arrays or sparse matrices.
    """
    rng = np.random.default_rng(seed)
    
    if hasattr(data_matrix, 'shape'):
        N = data_matrix.shape[0]
    else:
        N = len(data_matrix)
    
    if N <= sample_size:
        logger.info(f"Dataset size {N} <= sample_size {sample_size}, using full dataset")
        return np.arange(N)
    
    # Stratify by activity (sum of values or norm)
    if hasattr(data_matrix, 'sum'):
        activity = np.asarray(data_matrix.sum(axis=1)).ravel()
    else:
        activity = np.linalg.norm(data_matrix, axis=1)
    
    segments = np.digitize(activity, np.percentile(activity, [25, 50, 75]))
    
    # Stratified sample
    sample_idx_list = []
    for seg in np.unique(segments):
        seg_mask = (segments == seg)
        seg_indices = np.where(seg_mask)[0]
        n_target = int(np.ceil(sample_size * np.mean(seg_mask)))
        selected = rng.choice(seg_indices, size=min(n_target, len(seg_indices)), replace=False)
        sample_idx_list.extend(selected)
    
    sample_idx = np.array(sample_idx_list[:sample_size], dtype=np.int32)
    logger.info(f"Stratified sample: {len(sample_idx)} users")
    return sample_idx


# ============================================================================
# FIX #2: Modern Baselines — Core Functions
# ============================================================================

class SimplePredictor:
    """Base predictor interface."""
    def fit(self, user_item_matrix):
        raise NotImplementedError
    def predict(self, user_ids):
        raise NotImplementedError


class NMFPredictor(SimplePredictor):
    """Quick NMF for all datasets."""
    def __init__(self, n_factors=32, n_epochs=15, seed=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.seed = seed
        self.rng = np.random.default_rng(seed)
    
    def fit(self, user_item_matrix):
        n_users, n_items = user_item_matrix.shape
        self.W = self.rng.uniform(0.1, 1.0, (n_users, self.n_factors)).astype(np.float32)
        self.H = self.rng.uniform(0.1, 1.0, (self.n_factors, n_items)).astype(np.float32)
        
        for epoch in range(self.n_epochs):
            # Update H
            for i in range(n_items):
                col = user_item_matrix.getcol(i)
                active_users = col.nonzero()[0]
                if len(active_users) > 0:
                    W_active = self.W[active_users]
                    y = col.data
                    denom = np.dot(W_active.T, np.dot(W_active, np.ones(self.n_factors))) + 1e-8
                    self.H[:, i] = np.maximum(self.H[:, i] * (np.dot(W_active.T, y) / denom), 1e-8)
            
            # Update W
            for u in range(n_users):
                row = user_item_matrix[u]
                active_items = row.nonzero()[1]
                if len(active_items) > 0:
                    H_active = self.H[:, active_items].T
                    y = row.data
                    denom = np.dot(H_active.T, np.dot(H_active, np.ones(self.n_factors))) + 1e-8
                    self.W[u] = np.maximum(self.W[u] * (np.dot(H_active.T, y) / denom), 1e-8)
        
        return self
    
    def predict(self, user_ids):
        return np.dot(self.W[user_ids], self.H)


class BPRPredictor(SimplePredictor):
    """Quick BPR for all datasets."""
    def __init__(self, n_factors=32, n_epochs=20, seed=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.seed = seed
        self.rng = np.random.default_rng(seed)
    
    def fit(self, user_item_matrix):
        n_users, n_items = user_item_matrix.shape
        self.U = self.rng.normal(0, 0.01, (n_users, self.n_factors)).astype(np.float32)
        self.V = self.rng.normal(0, 0.01, (n_items, self.n_factors)).astype(np.float32)
        
        for epoch in range(self.n_epochs):
            for u in range(n_users):
                pos_idx = user_item_matrix[u].nonzero()[1]
                if len(pos_idx) == 0:
                    continue
                
                pos_i = self.rng.choice(pos_idx)
                neg_j = self.rng.integers(0, n_items)
                while neg_j in pos_idx:
                    neg_j = self.rng.integers(0, n_items)
                
                x_ui = np.dot(self.U[u], self.V[pos_i])
                x_uj = np.dot(self.U[u], self.V[neg_j])
                x_diff = x_ui - x_uj
                
                sigmoid_val = 1.0 / (1.0 + np.exp(-np.clip(x_diff, -500, 500)))
                delta = 1.0 - sigmoid_val
                
                self.U[u] += 0.01 * (delta * (self.V[pos_i] - self.V[neg_j]) - 0.001 * self.U[u])
                self.V[pos_i] += 0.01 * (delta * self.U[u] - 0.001 * self.V[pos_i])
                self.V[neg_j] += 0.01 * (-delta * self.U[u] - 0.001 * self.V[neg_j])
        
        return self
    
    def predict(self, user_ids):
        return np.dot(self.U[user_ids], self.V.T)


# ============================================================================
# FIX #3: Statistical Corrections — Bonferroni & FDR
# ============================================================================

def apply_bonferroni(p_values, alpha=0.05):
    """Apply Bonferroni correction: multiply each p-value by number of tests."""
    corrected = np.minimum(np.array(p_values) * len(p_values), 1.0)
    corrected_alpha = alpha / len(p_values)
    return corrected, corrected >= alpha, corrected_alpha


def apply_fdr_bh(p_values, alpha=0.05):
    """Apply Benjamini-Hochberg FDR correction."""
    p_arr = np.asarray(p_values)
    n_tests = len(p_arr)
    
    order = np.argsort(p_arr)
    sorted_p = p_arr[order]
    
    thresholds = (np.arange(1, n_tests + 1) / n_tests) * alpha
    valid_idx = np.where(sorted_p <= thresholds)[0]
    
    if len(valid_idx) > 0:
        i_max = valid_idx[-1]
        threshold = thresholds[i_max]
    else:
        threshold = 0.0
    
    reject = np.empty(n_tests, dtype=bool)
    reject[order] = sorted_p <= threshold
    
    return sorted_p, reject, threshold


# ============================================================================
# FIX #4: Memory-Efficient Notebook 03 Functions
# ============================================================================

def eval_labels_safe(dmat, labels, seed=42, mds_cache=None):
    """
    Safe evaluation with optional MDS caching (avoids O(N³) repeats).
    Returns (silhouette, davies_bouldin) or (nan, nan) on error.
    """
    try:
        labels = np.asarray(labels, dtype=np.int32)
        
        if len(np.unique(labels)) < 2:
            logger.warning(f"Invalid labels: only {len(np.unique(labels))} clusters")
            return float('nan'), float('nan')
        
        sil = float(silhouette_score(dmat, labels, metric="precomputed"))
        
        # Cache MDS if provided
        cache_key = f"mds_{len(dmat)}_{seed}"
        if mds_cache is not None and cache_key in mds_cache:
            emb = mds_cache[cache_key]
        else:
            emb = MDS(n_components=2, dissimilarity="precomputed", random_state=seed, n_init=2).fit_transform(dmat)
            if mds_cache is not None:
                mds_cache[cache_key] = emb
        
        db = float(davies_bouldin_score(emb, labels))
        return sil, db
    
    except Exception as e:
        logger.error(f"eval_labels_safe failed: {type(e).__name__}: {e}")
        return float('nan'), float('nan')


def kernel_to_distance_safe(kernel):
    """Convert kernel matrix to distance matrix safely."""
    try:
        k = np.asarray(kernel, dtype=np.float64)
        diag = np.clip(np.diag(k), 0.0, None)
        d2 = diag[:, None] + diag[None, :] - 2.0 * k
        d2 = np.clip(d2, 0.0, None)
        return np.sqrt(d2)
    except Exception as e:
        logger.error(f"kernel_to_distance_safe failed: {e}")
        return None


# ============================================================================
# Utility: Quick Metrics
# ============================================================================

def compute_quick_metrics(predictions, ground_truth, topk=10):
    """Compute HR@k, NDCG@k quickly."""
    hits = 0
    ndcg = 0.0
    
    for pred, truth in zip(predictions, ground_truth):
        if truth in pred[:topk]:
            hits += 1
            rank = list(pred[:topk]).index(truth) + 1
            ndcg += 1.0 / np.log2(rank + 1)
    
    n = len(ground_truth)
    return {
        'hr_at_k': float(hits / n) if n > 0 else 0.0,
        'ndcg_at_k': float(ndcg / n) if n > 0 else 0.0
    }


# ============================================================================
# Data Loading Utility (handles multiple dataset formats)
# ============================================================================

def load_dataset_pair(data_dir, dataset_type='ubcf'):
    """
    Load user_latent, user_sparse for any dataset.
    Handles: ubcf, amazon, Fashion, appliances
    """
    data_dir = Path(data_dir)
    
    try:
        from scipy.sparse import load_npz
    except:
        logger.error("scipy.sparse not available")
        return None, None
    
    try:
        user_latent = np.load(data_dir / "user_latent.npy").astype(np.float32)
        user_sparse = load_npz(data_dir / "user_sparse.npz").tocsr().astype(np.float32)
        logger.info(f"Loaded {dataset_type}: {user_sparse.shape[0]} users, {user_sparse.shape[1]} items")
        return user_latent, user_sparse
    except Exception as e:
        logger.error(f"Failed to load dataset from {data_dir}: {e}")
        return None, None


if __name__ == "__main__":
    logger.info("Shared fixes module loaded successfully")
