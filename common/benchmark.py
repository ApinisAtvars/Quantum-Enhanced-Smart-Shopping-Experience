import pandas as pd
import json
import os
from typing import Dict, Any

from .logging_utils import get_logger

logger = get_logger("Benchmark")

def summarize_benchmark(
    out_dir: str, 
    model_name: str, 
    metrics: Dict[str, float], 
    history: pd.DataFrame, 
    meta: Dict[str, Any]
):
    """
    Creates a benchmark summary combining model metrics, runtime, and split stats.
    """
    summary = {
        'model': model_name,
        'final_val_rmse': float(metrics.get('val_rmse', history['val_rmse'].iloc[-1])),
        'final_val_mae': float(metrics.get('val_mae', history['val_mae'].iloc[-1])),
        'n_users': meta['n_users'],
        'n_items': meta['n_items'],
        'interactions': meta['n_interactions'],
        'total_runtime_sec': float(history['epoch_runtime'].sum()),
        'avg_epoch_sec': float(history['epoch_runtime'].mean()),
    }
    
    # Add ranking metrics if evaluated
    for key in ['HR@10', 'Precision@10', 'Recall@10', 'NDCG@10']:
        if key in metrics:
            summary[key] = float(metrics[key])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_name.lower()}_benchmark_summary.json")
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=4)
        
    logger.info(f"Saved benchmark summary to {out_path}")
    return summary
