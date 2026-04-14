"""
Wrappers that adapt the repository's ALS and BiasedSGD models (originally designed
for sparse explicit-feedback collaborative filtering) to operate on a dense matrix V
so they can be compared fairly against NBMF.

Adaptation note:
  The repo ALS solves per-user and per-item regularized least-squares systems over
  *observed* interactions.  Here we treat every entry of the dense matrix V as an
  observed interaction, making the objective equivalent to dense matrix factorization
  with L2-regularized alternating least squares.

  The repo BiasedSGD includes user/item biases and a global mean on top of latent
  factors.  Here we iterate SGD over every (i, j, V[i,j]) triple per epoch.  The
  biases give SGD extra model capacity that NBMF does not have; this is documented
  as a structural difference, not hidden.

Both wrappers record the same per-iteration metrics as NBMF (Frobenius error,
relative error, wall-clock time) for fair comparison.
"""
import time
import numpy as np
from typing import List, Dict, Any, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.als import ALS
from models.sgd import BiasedSGD


def _build_als_dicts(V: np.ndarray):
    """Build user_items_dict and item_users_dict from ALL entries of V."""
    n, m = V.shape
    user_items = {}
    item_users = {}
    for i in range(n):
        user_items[i] = [(j, float(V[i, j])) for j in range(m)]
    for j in range(m):
        item_users[j] = [(i, float(V[i, j])) for i in range(n)]
    return user_items, item_users


def run_repo_als(
    V: np.ndarray,
    k: int,
    num_iterations: int,
    seed: int,
    reg: float = 0.0,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Run the repository ALS model on dense V.

    Parameters
    ----------
    V : (n, m) dense nonneg matrix
    k : rank / number of latent factors
    num_iterations : number of ALS passes
    seed : random seed
    reg : L2 regularization (0.0 for fairest comparison with NBMF)

    Returns
    -------
    history : list of per-iteration dicts
    total_runtime : wall-clock seconds
    """
    n, m = V.shape
    model = ALS(n_users=n, n_items=m, factors=k, reg=reg, seed=seed)
    user_items, item_users = _build_als_dicts(V)

    V_norm = np.linalg.norm(V, 'fro') + 1e-12
    history = []
    t_start = time.perf_counter()

    for it in range(num_iterations):
        t_it = time.perf_counter()
        model.fit_step(user_items, item_users)
        recon = model.P @ model.Q.T
        error = float(np.linalg.norm(V - recon, 'fro'))
        rel_error = error / V_norm
        it_time = time.perf_counter() - t_it
        history.append({
            'iteration': it + 1,
            'error': error,
            'relative_error': rel_error,
            'iter_time': it_time,
        })

    total_time = time.perf_counter() - t_start
    return history, total_time


def run_repo_sgd(
    V: np.ndarray,
    k: int,
    num_iterations: int,
    seed: int,
    lr: float = 0.01,
    reg: float = 0.0,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Run the repository BiasedSGD model on dense V.

    Parameters
    ----------
    V : (n, m) dense nonneg matrix
    k : rank / number of latent factors
    num_iterations : number of SGD epochs
    seed : random seed
    lr : learning rate
    reg : L2 regularization (0.0 for fairest comparison with NBMF)

    Returns
    -------
    history : list of per-iteration dicts
    total_runtime : wall-clock seconds
    """
    n, m = V.shape
    global_mean = float(V.mean())
    model = BiasedSGD(
        n_users=n, n_items=m, global_mean=global_mean,
        factors=k, lr=lr, reg=reg, seed=seed,
    )

    users_arr = np.repeat(np.arange(n), m)
    items_arr = np.tile(np.arange(m), n)
    ratings_arr = V.ravel().astype(np.float64)

    V_norm = np.linalg.norm(V, 'fro') + 1e-12
    history = []
    rng = np.random.RandomState(seed)
    t_start = time.perf_counter()

    for it in range(num_iterations):
        t_it = time.perf_counter()
        perm = rng.permutation(len(ratings_arr))
        model.fit_epoch(users_arr[perm], items_arr[perm], ratings_arr[perm])

        recon = np.zeros((n, m))
        for i in range(n):
            recon[i, :] = model.predict_for_user(i, np.arange(m))
        error = float(np.linalg.norm(V - recon, 'fro'))
        rel_error = error / V_norm
        it_time = time.perf_counter() - t_it
        history.append({
            'iteration': it + 1,
            'error': error,
            'relative_error': rel_error,
            'iter_time': it_time,
        })

    total_time = time.perf_counter() - t_start
    return history, total_time
