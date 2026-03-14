import argparse
import time
import os
import json
import numpy as np
import pandas as pd

from ALS.model import ALS
from common.data import load_data, preprocess_and_split
from common.metrics import get_rmse, get_mae, evaluate_ranking
from common.logging_utils import get_logger
from common.io_utils import save_json, save_qubo_bridge
from common.benchmark import summarize_benchmark

logger = get_logger("ALS_Train")

def parse_args():
    parser = argparse.ArgumentParser(description="Train ALS Matrix Factorization")
    parser.add_argument('--data', type=str, required=True, help="Path to ratings dataset")
    parser.add_argument('--out', type=str, required=True, help="Output directory")
    parser.add_argument('--config', type=str, default='', help="Path to config JSON")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # default
    config = {
        'factors': 50,
        'reg': 0.1,
        'iterations': 15,
        'seed': 42
    }
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config.update(json.load(f))
            
    os.makedirs(args.out, exist_ok=True)
    save_json(config, os.path.join(args.out, 'config_used.json'))

    logger.info(f"Loading data from {args.data}")
    df = load_data(args.data)
    train, val, test, meta = preprocess_and_split(df, seed=config['seed'])
    save_json(meta, os.path.join(args.out, 'meta.json'))
    
    n_users, n_items = meta['n_users'], meta['n_items']
    logger.info(f"Dataset stats: {n_users} users, {n_items} items, {meta['n_interactions']} interactions")
    
    logger.info("preparing data dictionaries for ALS...")
    train_user_items = train.groupby('user_idx')[['item_idx', 'rating']].apply(lambda x: list(zip(x.item_idx, x.rating))).to_dict()
    train_item_users = train.groupby('item_idx')[['user_idx', 'rating']].apply(lambda x: list(zip(x.user_idx, x.rating))).to_dict()
    
    val_u = val['user_idx'].values
    val_i = val['item_idx'].values
    val_r = val['rating'].values

    train_u = train['user_idx'].values
    train_i = train['item_idx'].values
    train_r = train['rating'].values

    model = ALS(
        n_users=n_users, 
        n_items=n_items, 
        factors=config['factors'], 
        reg=config['reg'], 
        seed=config['seed']
    )
    
    history = []
    logger.info("Starting training...")
    
    for epoch in range(1, config['iterations'] + 1):
        start_time = time.time()
        
        # ALS Update
        model.fit_step(train_user_items, train_item_users)
        
        # Predictions
        train_preds = model.predict(train_u, train_i)
        train_preds = np.clip(train_preds, meta['rating_min'], meta['rating_max'])
        
        val_preds = model.predict(val_u, val_i)
        val_preds = np.clip(val_preds, meta['rating_min'], meta['rating_max'])
        
        train_rmse = get_rmse(train_r, train_preds)
        train_mae = get_mae(train_r, train_preds)
        val_rmse = get_rmse(val_r, val_preds)
        val_mae = get_mae(val_r, val_preds)
        
        obj = model.calculate_objective(train_user_items)
        epoch_time = time.time() - start_time
        
        logger.info(f"Epoch {epoch:02d} | Time: {epoch_time:.2f}s | Obj: {obj:.2f} | "
                    f"Train RMSE: {train_rmse:.4f} MAE: {train_mae:.4f} | "
                    f"Val RMSE: {val_rmse:.4f} MAE: {val_mae:.4f}")
                    
        history.append({
            'epoch': epoch,
            'epoch_runtime': epoch_time,
            'objective': obj,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'val_rmse': val_rmse,
            'val_mae': val_mae
        })

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(args.out, 'history.csv'), index=False)

    logger.info("Evaluating Ranking Metrics on Test set...")
    ranking_metrics = evaluate_ranking(model, test, train, n_items=n_items, k=10, threshold=4.0)
    logger.info(f"Ranking Metrics: {ranking_metrics}")

    # Combine final metrics
    final_metrics = {
        'val_rmse': float(history[-1]['val_rmse']),
        'val_mae': float(history[-1]['val_mae']),
        **ranking_metrics
    }
    save_json(final_metrics, os.path.join(args.out, 'metrics.json'))
    
    # Save factors
    np.save(os.path.join(args.out, 'P.npy'), model.P)
    np.save(os.path.join(args.out, 'Q.npy'), model.Q)
    
    # Benchmark Prep
    summarize_benchmark(args.out, "ALS", final_metrics, history_df, meta)
    
    # QUBO Bridge Export
    # Select top active users and items for a dense subproblem
    logger.info("Exporting QUBO Bridge Subproblem...")
    user_counts = train['user_idx'].value_counts()
    item_counts = train['item_idx'].value_counts()
    
    top_users = user_counts.head(50).index.values
    top_items = item_counts.head(50).index.values
    
    sub_P = model.P[top_users, :]
    sub_Q = model.Q[top_items, :]
    
    # Reconstruct ratings block
    rating_block = sub_P @ sub_Q.T
    rating_block = np.clip(rating_block, meta['rating_min'], meta['rating_max'])
    
    # Mask block (1 if observed in train, 0 otherwise)
    mask_block = np.zeros((len(top_users), len(top_items)))
    sub_train = train[train['user_idx'].isin(top_users) & train['item_idx'].isin(top_items)]
    
    # map to local sub-indices
    u_map = {u: i for i, u in enumerate(top_users)}
    i_map = {i_id: j for j, i_id in enumerate(top_items)}
    
    for u, i in zip(sub_train['user_idx'], sub_train['item_idx']):
        mask_block[u_map[u], i_map[i]] = 1.0

    qubo_meta = {
        'n_users_sub': len(top_users),
        'n_items_sub': len(top_items),
        'original_user_ids': [int(u) for u in top_users],
        'original_item_ids': [int(i) for i in top_items],
        'description': "Dense 50x50 block of highly active users/items and their ALS predicted ratings. Mask indicates observed training interactions."
    }
    
    save_qubo_bridge(args.out, rating_block, mask_block, qubo_meta, prefix="als_qubo")
    logger.info("Done.")

if __name__ == "__main__":
    main()
