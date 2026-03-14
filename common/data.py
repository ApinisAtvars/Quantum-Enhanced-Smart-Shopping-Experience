import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

def load_data(filepath: str) -> pd.DataFrame:
    """Loads ratings data, normalizing columns to userId, itemId, rating, timestamp."""
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            # Rough column normalization, assuming standard or known setups
            cols = [c.lower() for c in df.columns]
            rename_map = {}
            for original, lower in zip(df.columns, cols):
                if 'user' in lower: rename_map[original] = 'userId'
                elif 'item' in lower or 'movie' in lower: rename_map[original] = 'itemId'
                elif 'rating' in lower or 'score' in lower: rename_map[original] = 'rating'
                elif 'time' in lower or 'date' in lower: rename_map[original] = 'timestamp'
            df.rename(columns=rename_map, inplace=True)

        elif filepath.endswith('.data') or filepath.endswith('.dat'):
            # MovieLens 100k or 1M
            sep = '\t' if filepath.endswith('.data') else '::'
            df = pd.read_csv(filepath, sep=sep, engine='python', header=None,
                             names=['userId', 'itemId', 'rating', 'timestamp'])
        else:
            raise ValueError("Unsupported file format. Use .csv, .data, or .dat")

        required = ['userId', 'itemId', 'rating']
        if not all(col in df.columns for col in required):
            raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}")

        return df

    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from {filepath}: {e}")


def preprocess_and_split(df: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Encodes IDs to contiguous integers and splits into train/val/test.
    Splitting uses time if available (last for test, 2nd-to-last for val).
    Otherwise, deterministic random split.
    """
    # 1. Encode user and item IDs
    user_ids = df['userId'].unique()
    item_ids = df['itemId'].unique()
    
    user2idx = {o: i for i, o in enumerate(user_ids)}
    item2idx = {o: i for i, o in enumerate(item_ids)}
    
    df['user_idx'] = df['userId'].map(user2idx)
    df['item_idx'] = df['itemId'].map(item2idx)
    
    n_users = len(user_ids)
    n_items = len(item_ids)

    # 2. Split Strategy
    if 'timestamp' in df.columns:
        # Sort by timestamp
        df = df.sort_values(['user_idx', 'timestamp'])
        
        # We need groups that have at least 3 interactions to have train, val, and test
        counts = df.groupby('user_idx').size()
        valid_users = counts[counts >= 3].index
        
        # Keep users with < 3 interactions only in train
        train_only = df[~df['user_idx'].isin(valid_users)]
        splittable = df[df['user_idx'].isin(valid_users)]
        
        test = splittable.groupby('user_idx').tail(1)
        splittable_no_test = splittable.drop(test.index)
        val = splittable_no_test.groupby('user_idx').tail(1)
        train_main = splittable_no_test.drop(val.index)
        
        train = pd.concat([train_main, train_only]).sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        # Deterministic fallback split (80-10-10 roughly)
        df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        counts = df_shuffled.groupby('user_idx').size()
        valid_users = counts[counts >= 3].index
        
        train_only = df_shuffled[~df_shuffled['user_idx'].isin(valid_users)]
        splittable = df_shuffled[df_shuffled['user_idx'].isin(valid_users)]
        
        test = splittable.groupby('user_idx').tail(1)
        splittable_no_test = splittable.drop(test.index)
        val = splittable_no_test.groupby('user_idx').tail(1)
        train_main = splittable_no_test.drop(val.index)
        
        train = pd.concat([train_main, train_only]).sample(frac=1, random_state=seed).reset_index(drop=True)

    # 3. Filter val and test to known users/items from train
    train_users = set(train['user_idx'].unique())
    train_items = set(train['item_idx'].unique())
    
    val = val[val['user_idx'].isin(train_users) & val['item_idx'].isin(train_items)]
    test = test[test['user_idx'].isin(train_users) & test['item_idx'].isin(train_items)]

    meta = {
        'n_users': n_users,
        'n_items': n_items,
        'n_interactions': len(df),
        'train_size': len(train),
        'val_size': len(val),
        'test_size': len(test),
        'rating_min': float(df['rating'].min()),
        'rating_max': float(df['rating'].max()),
        'sparsity': 1.0 - (len(df) / (n_users * n_items))
    }
    
    return train, val, test, meta
