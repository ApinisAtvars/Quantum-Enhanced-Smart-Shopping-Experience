import numpy as np
import dimod
from dwave.samplers import (
    SimulatedAnnealingSampler,
    TabuSampler,
    SteepestDescentSampler,
    PathIntegralAnnealingSampler,
)
from typing import Dict, Tuple, Any, Optional


SAMPLER_REGISTRY = {
    'simulated_annealing': 'SimulatedAnnealingSampler',
    'path_integral': 'PathIntegralAnnealingSampler',
    'tabu': 'TabuSampler',
    'steepest_descent': 'SteepestDescentSampler',
    'exact': 'ExactSolver',
}


def get_sampler(name: str) -> dimod.Sampler:
    """Instantiate a sampler by name."""
    if name == 'simulated_annealing':
        return SimulatedAnnealingSampler()
    elif name == 'path_integral':
        return PathIntegralAnnealingSampler()
    elif name == 'tabu':
        return TabuSampler()
    elif name == 'steepest_descent':
        return SteepestDescentSampler()
    elif name == 'exact':
        return dimod.ExactSolver()
    else:
        raise ValueError(f"Unknown sampler: {name}. Choose from {list(SAMPLER_REGISTRY.keys())}")


def sample_qubo(
    sampler: dimod.Sampler,
    Q: Dict[Tuple[int, int], float],
    k: int,
    sampler_name: str,
    num_reads: int = 100,
    num_sweeps: int = 1000,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Solve a QUBO problem and return the best binary vector.

    Parameters
    ----------
    sampler : dimod.Sampler
    Q : QUBO dictionary
    k : number of binary variables
    sampler_name : name key for parameter dispatch
    num_reads : number of reads/samples
    num_sweeps : sweeps per read (SA only)
    seed : random seed for reproducibility

    Returns
    -------
    q : np.ndarray of shape (k,) with binary values
    """
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

    if sampler_name == 'exact':
        response = sampler.sample(bqm)
    elif sampler_name == 'simulated_annealing':
        kwargs = dict(num_reads=num_reads, num_sweeps=num_sweeps)
        if seed is not None:
            kwargs['seed'] = seed
        response = sampler.sample(bqm, **kwargs)
    elif sampler_name == 'path_integral':
        kwargs = dict(num_reads=num_reads)
        if seed is not None:
            kwargs['seed'] = seed
        response = sampler.sample(bqm, **kwargs)
    elif sampler_name == 'tabu':
        kwargs = dict(num_reads=num_reads)
        if seed is not None:
            kwargs['seed'] = seed
        response = sampler.sample(bqm, **kwargs)
    elif sampler_name == 'steepest_descent':
        kwargs = dict(num_reads=num_reads)
        if seed is not None:
            kwargs['seed'] = seed
        response = sampler.sample(bqm, **kwargs)
    else:
        raise ValueError(f"Unknown sampler_name: {sampler_name}")

    best = response.first.sample
    q = np.array([int(best.get(i, 0)) for i in range(k)], dtype=np.float64)
    return q
