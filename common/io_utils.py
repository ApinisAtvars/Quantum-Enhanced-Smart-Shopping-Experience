import json
import os
import numpy as np
from typing import Dict, Any

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
