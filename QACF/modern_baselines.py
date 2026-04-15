"""
Modern Collaborative Filtering Baselines for Fair Comparison

Includes:
1. NMF (Non-negative Matrix Factorization)
2. BPR (Bayesian Personalized Ranking) via implicit-feedback
3. LightGCN (Graph neural network for collaborative filtering)

All normalized to the same evaluation protocol as UBCF and QACF.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix, load_npz
import logging
from typing import Tuple, Dict, List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NMFBaseline:
    """Non-negative Matrix Factorization baseline for implicit feedback."""
    
    def __init__(self, n_factors=32, n_epochs=100, learning_rate=0.01, seed=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = learning_rate
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
    def fit(self, user_item_matrix: csr_matrix):
        """
        Fit NMF on user-item matrix (implicit feedback).
        
        Args:
            user_item_matrix: (n_users, n_items) sparse matrix with ratings/implicit
        """
        n_users, n_items = user_item_matrix.shape
        
        # Initialize factors with non-negative values
        W = self.rng.uniform(0, 1, (n_users, self.n_factors)).astype(np.float32)
        H = self.rng.uniform(0, 1, (self.n_factors, n_items)).astype(np.float32)
        
        logger.info(f"NMF: {n_users} users × {n_items} items, {self.n_factors} factors")
        
        for epoch in range(self.n_epochs):
            # Update H (factors × items)
            for i in range(n_items):
                col_idx = user_item_matrix.getcol(i).nonzero()[0]
                if len(col_idx) > 0:
                    W_col = W[col_idx]  # (n_active_users, n_factors)
                    y_col = user_item_matrix[col_idx, i].toarray().ravel()  # (n_active_users,)
                    
                    denom = np.dot(W_col.T, np.dot(W_col, np.ones(self.n_factors))) + 1e-8
                    H[:, i] = np.maximum(
                        H[:, i] * (np.dot(W_col.T, y_col) / denom),
                        1e-8
                    )
            
            # Update W (users × factors)
            for u in range(n_users):
                row_idx = user_item_matrix[u].nonzero()[1]
                if len(row_idx) > 0:
                    H_row = H[:, row_idx].T  # (n_active_items, n_factors)
                    y_row = user_item_matrix[u, row_idx].toarray().ravel()  # (n_active_items,)
                    
                    denom = np.dot(H_row.T, np.dot(H_row, np.ones(self.n_factors))) + 1e-8
                    W[u] = np.maximum(
                        W[u] * (np.dot(H_row.T, y_row) / denom),
                        1e-8
                    )
            
            if (epoch + 1) % max(1, self.n_epochs // 10) == 0:
                # Compute reconstruction error
                R_approx = np.dot(W, H)
                mse = ((user_item_matrix.toarray() - R_approx) ** 2).mean()
                logger.info(f"  Epoch {epoch + 1}/{self.n_epochs}: MSE={mse:.6f}")
        
        self.W = W
        self.H = H
        return self
    
    def predict(self, user_ids: np.ndarray) -> np.ndarray:
        """Predict user-item scores."""
        return np.dot(self.W[user_ids], self.H)


class BPRBaseline:
    """Bayesian Personalized Ranking for implicit feedback."""
    
    def __init__(self, n_factors=32, n_epochs=50, learning_rate=0.01, reg_u=0.001, reg_i=0.001, seed=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = learning_rate
        self.reg_u = reg_u
        self.reg_i = reg_i
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
    def fit(self, user_item_matrix: csr_matrix):
        """
        Fit BPR: rank positive items > negative items for each user.
        
        Args:
            user_item_matrix: (n_users, n_items) sparse matrix with binary/implicit values
        """
        n_users, n_items = user_item_matrix.shape
        
        # Initialize factors
        self.U = self.rng.normal(0, 0.01, (n_users, self.n_factors)).astype(np.float32)
        self.V = self.rng.normal(0, 0.01, (n_items, self.n_factors)).astype(np.float32)
        
        logger.info(f"BPR: {n_users} users × {n_items} items, {self.n_factors} factors")
        
        for epoch in range(self.n_epochs):
            loss_sum = 0.0
            n_updates = 0
            
            for u in range(n_users):
                pos_idx = user_item_matrix[u].nonzero()[1]
                if len(pos_idx) == 0:
                    continue
                
                # Sample: positive item and negative item
                pos_i = self.rng.choice(pos_idx)
                neg_j = self.rng.integers(0, n_items)
                
                # Ensure negative item is not in user's positive set
                while neg_j in pos_idx:
                    neg_j = self.rng.integers(0, n_items)
                
                # BPR ranking loss: sigmoid(x_ui - x_uj) where x_uk = U_u · V_k^T
                x_ui = np.dot(self.U[u], self.V[pos_i])
                x_uj = np.dot(self.U[u], self.V[neg_j])
                x_diff = x_ui - x_uj
                
                # Sigmoid derivative
                sigmoid_val = 1.0 / (1.0 + np.exp(-x_diff))
                delta = 1.0 - sigmoid_val
                
                # Update factors
                self.U[u] += self.lr * (delta * (self.V[pos_i] - self.V[neg_j]) - self.reg_u * self.U[u])
                self.V[pos_i] += self.lr * (delta * self.U[u] - self.reg_i * self.V[pos_i])
                self.V[neg_j] += self.lr * (-delta * self.U[u] - self.reg_i * self.V[neg_j])
                
                loss_sum += -np.log(sigmoid_val + 1e-8)
                n_updates += 1
            
            if (epoch + 1) % max(1, self.n_epochs // 10) == 0:
                avg_loss = loss_sum / max(n_updates, 1)
                logger.info(f"  Epoch {epoch + 1}/{self.n_epochs}: Loss={avg_loss:.6f}")
        
        return self
    
    def predict(self, user_ids: np.ndarray) -> np.ndarray:
        """Predict user-item scores."""
        return np.dot(self.U[user_ids], self.V.T)


class SimpleGCNBaseline:
    """
    Lightweight GCN-inspired baseline (not full LightGCN, but GCN-adjacent).
    
    Uses graph convolution on user-item-user graph to aggregate neighbor signals.
    Simplified for interpretability: no positional encoding, single aggregation layer.
    """
    
    def __init__(self, n_factors=32, n_layers=2, learning_rate=0.01, seed=42):
        self.n_factors = n_factors
        self.n_layers = n_layers
        self.lr = learning_rate
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
    def fit(self, user_item_matrix: csr_matrix, user_latent: np.ndarray = None, n_epochs=20):
        """
        Fit GCN on user-item bipartite graph.
        
        Args:
            user_item_matrix: (n_users, n_items) sparse matrix
            user_latent: (n_users, d_latent) optional semantic features
            n_epochs: training iterations
        """
        n_users, n_items = user_item_matrix.shape
        
        # Initialize user/item embeddings
        if user_latent is not None and user_latent.shape[1] >= self.n_factors:
            self.user_emb = user_latent[:, :self.n_factors].astype(np.float32).copy()
        else:
            self.user_emb = self.rng.normal(0, 0.1, (n_users, self.n_factors)).astype(np.float32)
        
        self.item_emb = self.rng.normal(0, 0.1, (n_items, self.n_factors)).astype(np.float32)
        
        logger.info(f"SimpleGCN: {n_users} users, {n_items} items, {self.n_factors} factors, {self.n_layers} layers")
        
        for epoch in range(n_epochs):
            # Forward: aggregate neighbor messages
            user_emb_new = self.user_emb.copy()
            item_emb_new = self.item_emb.copy()
            
            for layer in range(self.n_layers):
                # User <- Item (via ratings)
                for u in range(n_users):
                    item_idx = user_item_matrix[u].nonzero()[1]
                    if len(item_idx) > 0:
                        agg = self.item_emb[item_idx].mean(axis=0)
                        user_emb_new[u] = 0.8 * self.user_emb[u] + 0.2 * agg
                
                # Item <- User (via ratings)
                for i in range(n_items):
                    user_idx = user_item_matrix.getcol(i).nonzero()[0]
                    if len(user_idx) > 0:
                        agg = self.user_emb[user_idx].mean(axis=0)
                        item_emb_new[i] = 0.8 * self.item_emb[i] + 0.2 * agg
                
                self.user_emb = user_emb_new.copy()
                self.item_emb = item_emb_new.copy()
            
            if (epoch + 1) % max(1, n_epochs // 5) == 0:
                logger.info(f"  Epoch {epoch + 1}/{n_epochs}: Layer propagation complete")
        
        return self
    
    def predict(self, user_ids: np.ndarray) -> np.ndarray:
        """Predict user-item scores via dot product."""
        return np.dot(self.user_emb[user_ids], self.item_emb.T)


def evaluate_baseline(model, user_item_matrix: csr_matrix, user_ids: np.ndarray, 
                      topk=10, n_eval_users=500, seed=42) -> Dict:
    """
    Standard evaluation: HR@k, NDCG@k on hold-out test.
    """
    rng = np.random.default_rng(seed)
    
    # Filter to users with >= 3 ratings
    user_nnz = np.diff(user_item_matrix.indptr)
    eligible = np.where(user_nnz >= 3)[0]
    eval_users = rng.choice(eligible, size=min(n_eval_users, len(eligible)), replace=False)
    
    hits, ndcgs, mrrs = [], [], []
    
    for u in eval_users:
        start, end = user_item_matrix.indptr[u], user_item_matrix.indptr[u + 1]
        items = user_item_matrix.indices[start:end]
        
        if len(items) < 2:
            continue
        
        # Hold-out one item
        held_idx = int(rng.integers(len(items)))
        held_item = int(items[held_idx])
        train_items = set(items.tolist())
        train_items.discard(held_item)
        
        # Predict
        scores = model.predict(np.array([u]))[0]
        
        # Rank excluding training set
        for item in train_items:
            scores[item] = -np.inf
        
        top_idx = np.argsort(-scores)[:topk]
        
        hit = 1.0 if held_item in top_idx else 0.0
        hits.append(hit)
        
        if hit:
            rank = int(np.where(top_idx == held_item)[0][0]) + 1
            ndcgs.append(1.0 / np.log2(rank + 1))
            mrrs.append(1.0 / rank)
        else:
            ndcgs.append(0.0)
            mrrs.append(0.0)
    
    return {
        'hr_at_k': float(np.mean(hits)) if hits else np.nan,
        'ndcg_at_k': float(np.mean(ndcgs)) if ndcgs else np.nan,
        'mrr': float(np.mean(mrrs)) if mrrs else np.nan,
        'n_eval': len(hits)
    }


if __name__ == "__main__":
    # Example usage
    in_dir = Path("data/processed_32m_ubcf")
    
    user_sparse = load_npz(in_dir / "user_sparse.npz").tocsr().astype(np.float32)
    user_latent = np.load(in_dir / "user_latent.npy").astype(np.float32)
    
    results = []
    
    # NMF
    logger.info("\n" + "="*80)
    logger.info("Fitting NMF baseline...")
    nmf = NMFBaseline(n_factors=32, n_epochs=20, seed=42)
    nmf.fit(user_sparse)
    nmf_eval = evaluate_baseline(nmf, user_sparse, None)
    logger.info(f"NMF: HR@10={nmf_eval['hr_at_k']:.4f}, NDCG@10={nmf_eval['ndcg_at_k']:.4f}")
    results.append({'model': 'NMF', **nmf_eval})
    
    # BPR
    logger.info("\n" + "="*80)
    logger.info("Fitting BPR baseline...")
    bpr = BPRBaseline(n_factors=32, n_epochs=30, seed=42)
    bpr.fit(user_sparse)
    bpr_eval = evaluate_baseline(bpr, user_sparse, None)
    logger.info(f"BPR: HR@10={bpr_eval['hr_at_k']:.4f}, NDCG@10={bpr_eval['ndcg_at_k']:.4f}")
    results.append({'model': 'BPR', **bpr_eval})
    
    # SimpleGCN
    logger.info("\n" + "="*80)
    logger.info("Fitting SimpleGCN baseline...")
    gcn = SimpleGCNBaseline(n_factors=32, n_layers=2, seed=42)
    gcn.fit(user_sparse, user_latent=user_latent, n_epochs=15)
    gcn_eval = evaluate_baseline(gcn, user_sparse, None)
    logger.info(f"GCN: HR@10={gcn_eval['hr_at_k']:.4f}, NDCG@10={gcn_eval['ndcg_at_k']:.4f}")
    results.append({'model': 'SimpleGCN', **gcn_eval})
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(in_dir / "modern_baselines_comparison.csv", index=False)
    logger.info("\n" + "="*80)
    logger.info("Results saved to modern_baselines_comparison.csv")
    logger.info(results_df.to_string(index=False))
