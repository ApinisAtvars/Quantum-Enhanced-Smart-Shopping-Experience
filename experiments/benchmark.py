"""
Core benchmark: compare primary methods on a MovieLens dense subblock.

Methods: Repo ALS, Repo SGD, NBMF+SA, NBMF+PathIntegral, NMF-MU (supplemental).

Produces:
  results/benchmark_primary_results.csv
  results/benchmark_supplemental_results.csv
  results/convergence_results.csv
  results/runtime_breakdown_results.csv
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.io import extract_dense_subblock
from utils.logging import get_logger
from models.nbmf import NBMF
from experiments.baseline_wrappers import run_repo_als, run_repo_sgd

logger = get_logger("Benchmark")


def run_classical_nmf_mu(V, k, num_iterations, seed):
    """Lee-Seung multiplicative updates NMF baseline."""
    n, m = V.shape
    rng = np.random.RandomState(seed)
    W = rng.rand(n, k) + 1e-6
    H = rng.rand(k, m) + 1e-6
    V_norm = np.linalg.norm(V, 'fro') + 1e-12

    history = []
    t_start = time.perf_counter()
    for it in range(num_iterations):
        t_it = time.perf_counter()
        H = H * (W.T @ V) / (W.T @ W @ H + 1e-12)
        W = W * (V @ H.T) / (W @ H @ H.T + 1e-12)
        error = float(np.linalg.norm(V - W @ H, 'fro'))
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


def run_nbmf(V, k, sampler_name, num_iterations, num_reads, num_sweeps, seed):
    n, m = V.shape
    model = NBMF(
        n=n, m=m, k=k,
        sampler_name=sampler_name,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        seed=seed,
    )
    result = model.fit(V, num_iterations=num_iterations, verbose=False)
    return result['history'], result['timing']['total_runtime'], result['timing']


def main():
    with open('config/experiment.yaml') as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg['general']['output_dir'], exist_ok=True)
    seeds = cfg['general']['seeds']
    num_iterations = cfg['general']['num_iterations']
    rank = cfg['benchmark']['rank']

    ref_size = cfg['benchmark']['reference_subblock']
    logger.info(f"Extracting MovieLens {ref_size[0]}x{ref_size[1]} subblock...")
    V, mask, ml_meta = extract_dense_subblock(
        cfg['movielens']['data_path'], ref_size[0], ref_size[1]
    )
    logger.info(f"Subblock shape: {V.shape}, density: {ml_meta['density']:.4f}")

    sa_cfg = cfg['samplers']['simulated_annealing']
    pi_cfg = cfg['samplers']['path_integral']

    primary_rows = []
    supplemental_rows = []
    convergence_rows = []
    breakdown_rows = []

    for seed in seeds:
        logger.info(f"=== Seed {seed} ===")

        # ---- Repo ALS ----
        logger.info("Running Repo_ALS...")
        hist, rt = run_repo_als(
            V, rank, num_iterations, seed,
            reg=cfg['benchmark']['repo_als_reg'],
        )
        primary_rows.append({
            'method': 'Repo_ALS', 'rank': rank, 'seed': seed,
            'final_error': hist[-1]['error'],
            'final_relative_error': hist[-1]['relative_error'],
            'total_runtime': rt,
            'avg_iter_time': rt / num_iterations,
            'matrix_size': f"{V.shape[0]}x{V.shape[1]}",
        })
        cum_time = 0.0
        for h in hist:
            cum_time += h['iter_time']
            convergence_rows.append({**h, 'method': 'Repo_ALS', 'seed': seed,
                                     'rank': rank, 'tier': 'primary',
                                     'cumulative_time': cum_time})

        # ---- Repo SGD ----
        logger.info("Running Repo_SGD...")
        hist, rt = run_repo_sgd(
            V, rank, num_iterations, seed,
            lr=cfg['benchmark']['repo_sgd_lr'],
            reg=cfg['benchmark']['repo_sgd_reg'],
        )
        primary_rows.append({
            'method': 'Repo_SGD', 'rank': rank, 'seed': seed,
            'final_error': hist[-1]['error'],
            'final_relative_error': hist[-1]['relative_error'],
            'total_runtime': rt,
            'avg_iter_time': rt / num_iterations,
            'matrix_size': f"{V.shape[0]}x{V.shape[1]}",
        })
        cum_time = 0.0
        for h in hist:
            cum_time += h['iter_time']
            convergence_rows.append({**h, 'method': 'Repo_SGD', 'seed': seed,
                                     'rank': rank, 'tier': 'primary',
                                     'cumulative_time': cum_time})

        # ---- NBMF: SA and PathIntegral ----
        primary_nbmf = [
            ('NBMF_SA', 'simulated_annealing', sa_cfg['num_reads'], sa_cfg['num_sweeps']),
            ('NBMF_PathIntegral', 'path_integral', pi_cfg['num_reads'], 0),
        ]
        for method_name, sname, nreads, nsweeps in primary_nbmf:
            logger.info(f"Running {method_name}...")
            hist, rt, timing = run_nbmf(V, rank, sname, num_iterations, nreads, nsweeps, seed)
            num_qubo_solves = num_iterations * V.shape[1]
            primary_rows.append({
                'method': method_name, 'rank': rank, 'seed': seed,
                'final_error': hist[-1]['error'],
                'final_relative_error': hist[-1]['relative_error'],
                'total_runtime': rt,
                'avg_iter_time': timing['avg_iter_time'],
                'nnls_time': timing['total_nnls_time'],
                'qubo_build_time': timing['total_qubo_build_time'],
                'annealing_time': timing['total_annealing_time'],
                'postprocess_time': timing['total_postprocess_time'],
                'num_qubo_solves': num_qubo_solves,
                'matrix_size': f"{V.shape[0]}x{V.shape[1]}",
            })
            breakdown_rows.append({
                'method': method_name, 'rank': rank, 'seed': seed,
                'total_runtime': rt,
                'nnls_time': timing['total_nnls_time'],
                'qubo_build_time': timing['total_qubo_build_time'],
                'annealing_time': timing['total_annealing_time'],
                'postprocess_time': timing['total_postprocess_time'],
                'num_qubo_solves': num_qubo_solves,
                'avg_qubo_solve_time': timing['total_annealing_time'] / max(num_qubo_solves, 1),
            })
            cum_time = 0.0
            for h in hist:
                cum_time += h['iter_time']
                convergence_rows.append({**h, 'method': method_name, 'seed': seed,
                                         'rank': rank, 'tier': 'primary',
                                         'cumulative_time': cum_time})

        # ---- Supplemental: NMF-MU ----
        logger.info("Running NMF_MU (supplemental)...")
        hist, rt = run_classical_nmf_mu(V, rank, num_iterations, seed)
        supplemental_rows.append({
            'method': 'NMF_MU', 'rank': rank, 'seed': seed,
            'final_error': hist[-1]['error'],
            'final_relative_error': hist[-1]['relative_error'],
            'total_runtime': rt,
            'avg_iter_time': rt / num_iterations,
            'matrix_size': f"{V.shape[0]}x{V.shape[1]}",
        })
        cum_time = 0.0
        for h in hist:
            cum_time += h['iter_time']
            convergence_rows.append({**h, 'method': 'NMF_MU', 'seed': seed,
                                     'rank': rank, 'tier': 'supplemental',
                                     'cumulative_time': cum_time})

    # ---- Save ----
    out_dir = cfg['general']['output_dir']

    pd.DataFrame(primary_rows).to_csv(
        os.path.join(out_dir, 'benchmark_primary_results.csv'), index=False)
    logger.info(f"Saved benchmark_primary_results.csv ({len(primary_rows)} rows)")

    pd.DataFrame(supplemental_rows).to_csv(
        os.path.join(out_dir, 'benchmark_supplemental_results.csv'), index=False)
    logger.info(f"Saved benchmark_supplemental_results.csv ({len(supplemental_rows)} rows)")

    pd.DataFrame(convergence_rows).to_csv(
        os.path.join(out_dir, 'convergence_results.csv'), index=False)
    logger.info(f"Saved convergence_results.csv ({len(convergence_rows)} rows)")

    pd.DataFrame(breakdown_rows).to_csv(
        os.path.join(out_dir, 'runtime_breakdown_results.csv'), index=False)
    logger.info(f"Saved runtime_breakdown_results.csv ({len(breakdown_rows)} rows)")

    print("\n=== PRIMARY BENCHMARK SUMMARY ===")
    pdf = pd.DataFrame(primary_rows)
    for method in sorted(pdf['method'].unique()):
        sub = pdf[pdf['method'] == method]
        print(f"  {method:25s}  error={sub['final_relative_error'].mean():.6f} +/- {sub['final_relative_error'].std():.6f}  "
              f"runtime={sub['total_runtime'].mean():.3f}s")


if __name__ == "__main__":
    main()
