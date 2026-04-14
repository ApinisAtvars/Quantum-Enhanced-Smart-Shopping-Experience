import numpy as np
from typing import Tuple, Optional, Dict, Any


def generate_synthetic(
    n: int,
    m: int,
    k: int,
    noise_std: float = 0.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic nonneg/binary matrix factorization problem.

    Returns V, W_true, H_true, V_noisy where:
        W_true ∈ R+^{n×k}   (nonneg continuous)
        H_true ∈ {0,1}^{k×m} (binary)
        V      = W_true @ H_true
        V_noisy = V + noise  (clipped to >= 0)
    """
    rng = np.random.RandomState(seed)
    W_true = rng.rand(n, k)
    H_true = rng.randint(0, 2, size=(k, m)).astype(np.float64)
    V = W_true @ H_true

    if noise_std > 0:
        noise = rng.randn(n, m) * noise_std
        V_noisy = np.maximum(V + noise, 0.0)
    else:
        V_noisy = V.copy()

    return V, W_true, H_true, V_noisy


def synthetic_meta(n: int, m: int, k: int, noise_std: float, seed: int) -> Dict[str, Any]:
    return {
        'n': n,
        'm': m,
        'rank': k,
        'noise_std': noise_std,
        'seed': seed,
        'data_type': 'synthetic',
    }
