import json
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

def save_json(data: Dict[str, Any], path: str):
    """Saves dictionary to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(path: str) -> Dict[str, Any]:
    """Loads JSON file to dictionary."""
    with open(path, 'r') as f:
        return json.load(f)

def save_qubo_bridge(
    out_dir: str, 
    ratings_matrix: np.ndarray, 
    mask_matrix: np.ndarray, 
    meta: Dict[str, Any], 
    prefix: str = "qubo"
):
    """
    Exports a small dense matrix-factorization subproblem (ratings and mask)
    along with metadata, serving as a bridge to QUBO formulation.
    """
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f'{prefix}_ratings.npy'), ratings_matrix)
    np.save(os.path.join(out_dir, f'{prefix}_mask.npy'), mask_matrix)
    save_json(meta, os.path.join(out_dir, f'{prefix}_meta.json'))


def extract_dense_subblock(
    data_path: str,
    n_users: int,
    n_items: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Extract a dense subblock from MovieLens data by selecting
    the top-N most active users and top-M most active items.

    Returns (rating_matrix, mask_matrix, meta) where:
        rating_matrix: (n_users, n_items) dense matrix (0 for unobserved)
        mask_matrix:   (n_users, n_items) binary mask (1 = observed)
        meta:          dict with original IDs and stats
    """
    sep = '\t' if data_path.endswith('.data') else '::'
    df = pd.read_csv(
        data_path, sep=sep, engine='python', header=None,
        names=['userId', 'itemId', 'rating', 'timestamp']
    )

    user_counts = df['userId'].value_counts()
    item_counts = df['itemId'].value_counts()

    actual_n_users = min(n_users, len(user_counts))
    actual_n_items = min(n_items, len(item_counts))

    top_users = user_counts.head(actual_n_users).index.values
    top_items = item_counts.head(actual_n_items).index.values

    sub_df = df[df['userId'].isin(top_users) & df['itemId'].isin(top_items)]

    u_map = {u: i for i, u in enumerate(top_users)}
    i_map = {it: j for j, it in enumerate(top_items)}

    rating_matrix = np.zeros((actual_n_users, actual_n_items))
    mask_matrix = np.zeros((actual_n_users, actual_n_items))

    for _, row in sub_df.iterrows():
        ui = u_map[row['userId']]
        ij = i_map[row['itemId']]
        rating_matrix[ui, ij] = row['rating']
        mask_matrix[ui, ij] = 1.0

    density = mask_matrix.sum() / (actual_n_users * actual_n_items)

    meta = {
        'n_users': actual_n_users,
        'n_items': actual_n_items,
        'n_observed': int(mask_matrix.sum()),
        'density': float(density),
        'original_user_ids': [int(u) for u in top_users],
        'original_item_ids': [int(i) for i in top_items],
        'data_path': data_path,
    }

    return rating_matrix, mask_matrix, meta
