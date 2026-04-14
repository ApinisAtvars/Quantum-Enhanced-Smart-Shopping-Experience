import time
import numpy as np
from scipy.optimize import nnls
from typing import Dict, List, Any, Optional

from .qubo import build_qubo_dict
from .samplers import get_sampler, sample_qubo


class NBMF:
    """
    Nonnegative/Binary Matrix Factorization via QUBO.

    V ≈ W H  where  W ∈ R+^{n×k}, H ∈ {0,1}^{k×m}

    Alternating optimization:
        1. Fix H, update W row-wise via NNLS
        2. Fix W, update H column-wise via QUBO
    """

    def __init__(
        self,
        n: int,
        m: int,
        k: int,
        sampler_name: str = 'simulated_annealing',
        num_reads: int = 100,
        num_sweeps: int = 1000,
        seed: int = 42,
    ):
        self.n = n
        self.m = m
        self.k = k
        self.sampler_name = sampler_name
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.seed = seed

        rng = np.random.RandomState(seed)
        self.W = rng.rand(n, k).astype(np.float64)
        self.H = rng.randint(0, 2, size=(k, m)).astype(np.float64)

        self.sampler = get_sampler(sampler_name)

    def fit(
        self,
        V: np.ndarray,
        num_iterations: int = 20,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the alternating NNLS + QUBO optimization.

        Returns a dict with:
            - history: list of per-iteration records
            - final_error: last reconstruction error
            - timing: aggregated timing breakdown
        """
        n, m = V.shape
        assert n == self.n and m == self.m

        history = []
        total_nnls_time = 0.0
        total_qubo_build_time = 0.0
        total_annealing_time = 0.0
        total_postprocess_time = 0.0

        t_total_start = time.perf_counter()

        for it in range(num_iterations):
            t_iter_start = time.perf_counter()

            # --- Step 1: Update W via NNLS (fix H) ---
            t_nnls_start = time.perf_counter()
            for i in range(n):
                result, _ = nnls(self.H.T, V[i, :])
                self.W[i, :] = result
            t_nnls_end = time.perf_counter()
            iter_nnls_time = t_nnls_end - t_nnls_start
            total_nnls_time += iter_nnls_time

            # --- Step 2: Update H via QUBO (fix W) ---
            iter_qubo_build = 0.0
            iter_anneal = 0.0
            iter_postproc = 0.0

            for j in range(m):
                t_qb = time.perf_counter()
                Q = build_qubo_dict(self.W, V[:, j])
                t_qb_end = time.perf_counter()
                iter_qubo_build += t_qb_end - t_qb

                t_ann = time.perf_counter()
                q = sample_qubo(
                    self.sampler, Q, self.k,
                    sampler_name=self.sampler_name,
                    num_reads=self.num_reads,
                    num_sweeps=self.num_sweeps,
                    seed=self.seed + it * m + j if self.seed is not None else None,
                )
                t_ann_end = time.perf_counter()
                iter_anneal += t_ann_end - t_ann

                t_pp = time.perf_counter()
                self.H[:, j] = q
                t_pp_end = time.perf_counter()
                iter_postproc += t_pp_end - t_pp

            total_qubo_build_time += iter_qubo_build
            total_annealing_time += iter_anneal
            total_postprocess_time += iter_postproc

            # --- Compute reconstruction error ---
            recon = self.W @ self.H
            error = float(np.linalg.norm(V - recon, 'fro'))
            rel_error = float(error / (np.linalg.norm(V, 'fro') + 1e-12))

            t_iter_end = time.perf_counter()
            iter_time = t_iter_end - t_iter_start

            record = {
                'iteration': it + 1,
                'error': error,
                'relative_error': rel_error,
                'iter_time': iter_time,
                'nnls_time': iter_nnls_time,
                'qubo_build_time': iter_qubo_build,
                'annealing_time': iter_anneal,
                'postprocess_time': iter_postproc,
            }
            history.append(record)

            if verbose:
                print(f"  iter {it+1:3d}/{num_iterations} | "
                      f"error={error:.6f} | rel={rel_error:.6f} | "
                      f"time={iter_time:.3f}s")

        t_total_end = time.perf_counter()
        total_runtime = t_total_end - t_total_start

        timing = {
            'total_runtime': total_runtime,
            'total_nnls_time': total_nnls_time,
            'total_qubo_build_time': total_qubo_build_time,
            'total_annealing_time': total_annealing_time,
            'total_postprocess_time': total_postprocess_time,
            'avg_iter_time': total_runtime / num_iterations,
        }

        return {
            'history': history,
            'final_error': history[-1]['error'],
            'final_relative_error': history[-1]['relative_error'],
            'timing': timing,
            'W': self.W.copy(),
            'H': self.H.copy(),
        }
