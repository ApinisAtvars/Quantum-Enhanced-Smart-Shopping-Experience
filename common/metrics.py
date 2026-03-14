import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple

def get_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def get_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def evaluate_ranking(
    model, 
    test_df: pd.DataFrame, 
    train_df: pd.DataFrame, 
    n_items: int, 
    k: int = 10, 
    threshold: float = 4.0
) -> Dict[str, float]:
    """
    Evaluates Top-K ranking metrics.
    test_df: The test or validation set.
    train_df: The training set (used to exclude seen items).
    model: An object implementing an `predict_for_user(u, items)` method.
    threshold: Rating threshold to consider an item relevant.
    """
    # Create dict of seen items for fast lookup
    train_groups = train_df.groupby('user_idx')['item_idx'].apply(set).to_dict()
    test_groups = test_df.groupby('user_idx')
    
    hr, ndcg, prec, rec = 0.0, 0.0, 0.0, 0.0
    n_users = 0
    all_items = np.arange(n_items)
    
    for user_idx, group in test_groups:
        relevant_items = set(group[group['rating'] >= threshold]['item_idx'].tolist())
        if not relevant_items:
            continue
            
        seen_items = train_groups.get(user_idx, set())
        unseen_items = np.array(list(set(all_items) - seen_items))
        
        # We need predictions for all unseen items
        preds = model.predict_for_user(user_idx, unseen_items)
        
        # Get top-k item indices (relative to the unseen_items array)
        top_k_rel_idx = np.argsort(preds)[-k:][::-1]
        top_k_items = unseen_items[top_k_rel_idx]
        
        # Calculate metrics
        hits = [1 if item in relevant_items else 0 for item in top_k_items]
        
        if sum(hits) > 0:
            hr += 1.0
            
        # NDCG
        dcg = sum([hits[i] / np.log2(i + 2) for i in range(len(hits))])
        idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(relevant_items), k))])
        ndcg += dcg / idcg if idcg > 0 else 0
        
        prec += sum(hits) / k
        rec += sum(hits) / len(relevant_items)
        
        n_users += 1
        
    if n_users == 0:
        return {'HR@10': 0.0, 'NDCG@10': 0.0, 'Precision@10': 0.0, 'Recall@10': 0.0}
        
    return {
        f'HR@{k}': hr / n_users,
        f'NDCG@{k}': ndcg / n_users,
        f'Precision@{k}': prec / n_users,
        f'Recall@{k}': rec / n_users
    }
