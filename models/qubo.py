import numpy as np
import dimod
from typing import Dict, Tuple


def build_qubo_dict(W: np.ndarray, v_j: np.ndarray) -> Dict[Tuple[int, int], float]:
    """
    Build a QUBO dictionary for one column of H.

    Minimizes ||v_j - W q||^2 over binary q ∈ {0,1}^k.

    Expansion:
        ||v_j - Wq||^2 = v_j^T v_j  -  2 v_j^T W q  +  q^T (W^T W) q

    The constant v_j^T v_j does not affect the argmin.

    Diagonal (linear) terms:
        Q(i,i) = (W^T W)_{ii} - 2 (W^T v_j)_i

    Off-diagonal (quadratic) terms:
        Q(i,j) = 2 (W^T W)_{ij}   for i < j
    """
    k = W.shape[1]
    WtW = W.T @ W
    Wtv = W.T @ v_j

    Q = {}
    for i in range(k):
        Q[(i, i)] = float(WtW[i, i] - 2.0 * Wtv[i])
        for j in range(i + 1, k):
            Q[(i, j)] = float(2.0 * WtW[i, j])

    return Q


def build_bqm(W: np.ndarray, v_j: np.ndarray) -> dimod.BinaryQuadraticModel:
    """Build a BQM from the QUBO dictionary (some samplers prefer this)."""
    Q = build_qubo_dict(W, v_j)
    return dimod.BinaryQuadraticModel.from_qubo(Q)


def qubo_energy(Q: Dict[Tuple[int, int], float], q: np.ndarray) -> float:
    """Evaluate QUBO energy for a given binary vector."""
    energy = 0.0
    for (i, j), val in Q.items():
        energy += val * q[i] * q[j]
    return energy
