"""
Scalability study: runtime and error across varying matrix sizes and ranks.

Also tests ExactSolver feasibility boundary on small problems.

Produces results/scalability_results.csv.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.synthetic_data import generate_synthetic
from common.io_utils import extract_dense_subblock
from common.logging_utils import get_logger
from NBMF.model import NBMF

logger = get_logger("Scalability")

TIMEOUT = 600


def run_single(V, k, sampler_name, num_iterations, num_reads, num_sweeps, seed):
    n, m = V.shape
    model = NBMF(
        n=n, m=m, k=k,
        sampler_name=sampler_name,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        seed=seed,
    )
    result = model.fit(V, num_iterations=num_iterations, verbose=False)
    num_qubo_solves = num_iterations * m
    timing = result['timing']
    return {
        'final_error': result['final_error'],
        'final_relative_error': result['final_relative_error'],
        'total_runtime': timing['total_runtime'],
        'avg_iter_time': timing['avg_iter_time'],
        'total_nnls_time': timing['total_nnls_time'],
        'total_qubo_build_time': timing['total_qubo_build_time'],
        'total_annealing_time': timing['total_annealing_time'],
        'total_postprocess_time': timing['total_postprocess_time'],
        'num_qubo_solves': num_qubo_solves,
        'avg_qubo_solve_time': timing['total_annealing_time'] / max(num_qubo_solves, 1),
    }


def get_sampler_params(cfg, sampler_name):
    if sampler_name == 'simulated_annealing':
        sa = cfg['samplers']['simulated_annealing']
        return sa['num_reads'], sa['num_sweeps']
    elif sampler_name == 'path_integral':
        pi = cfg['samplers']['path_integral']
        return pi['num_reads'], 0
    elif sampler_name == 'exact':
        return 0, 0
    else:
        return cfg['samplers'].get(sampler_name, {}).get('num_reads', 1), 0


def main():
    with open('config/experiment_config.yaml') as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg['general']['output_dir'], exist_ok=True)
    seeds = cfg['scalability']['scalability_seeds']
    num_iterations = cfg['general']['num_iterations']

    rows = []

    # --- Synthetic scalability grid ---
    syn_cfg = cfg['scalability']['synthetic_grid']
    for sampler_name in syn_cfg['samplers']:
        nreads, nsweeps = get_sampler_params(cfg, sampler_name)
        for size in syn_cfg['sizes']:
            n, m = size
            for k in syn_cfg['ranks']:
                if k >= n:
                    continue
                for seed in seeds:
                    logger.info(f"Synthetic {n}x{m} | k={k} | {sampler_name} | seed={seed}")
                    V, _, _, _ = generate_synthetic(n, m, k, seed=seed)
                    res = run_single(V, k, sampler_name, num_iterations, nreads, nsweeps, seed)
                    rows.append({
                        'data_source': 'synthetic',
                        'matrix_size': f"{n}x{m}",
                        'n': n, 'm': m, 'rank': k,
                        'sampler': sampler_name,
                        'seed': seed,
                        **res,
                    })

    # --- MovieLens scalability grid ---
    ml_cfg = cfg['scalability']['movielens_grid']
    for sampler_name in ml_cfg['samplers']:
        nreads, nsweeps = get_sampler_params(cfg, sampler_name)
        for size in ml_cfg['sizes']:
            n_u, n_i = size
            logger.info(f"Extracting MovieLens {n_u}x{n_i} subblock...")
            V_ml, mask, meta = extract_dense_subblock(
                cfg['movielens']['data_path'], n_u, n_i
            )
            logger.info(f"  actual shape: {V_ml.shape}, density: {meta['density']:.4f}")
            for k in ml_cfg['ranks']:
                for seed in seeds:
                    logger.info(f"MovieLens {V_ml.shape[0]}x{V_ml.shape[1]} | k={k} | {sampler_name} | seed={seed}")
                    res = run_single(V_ml, k, sampler_name, num_iterations, nreads, nsweeps, seed)
                    rows.append({
                        'data_source': 'movielens',
                        'matrix_size': f"{V_ml.shape[0]}x{V_ml.shape[1]}",
                        'n': V_ml.shape[0], 'm': V_ml.shape[1], 'rank': k,
                        'sampler': sampler_name,
                        'seed': seed,
                        **res,
                    })

    # --- ExactSolver feasibility boundary ---
    feas_cfg = cfg['scalability']['feasibility']
    for size in feas_cfg['exact_sizes']:
        n, m = size
        for k in feas_cfg['exact_ranks']:
            if k >= n:
                continue
            seed = seeds[0]
            logger.info(f"Feasibility: Exact {n}x{m} | k={k}")
            V, _, _, _ = generate_synthetic(n, m, k, seed=seed)
            t0 = time.perf_counter()
            try:
                res = run_single(V, k, 'exact', num_iterations, 0, 0, seed)
                elapsed = time.perf_counter() - t0
                feasible = elapsed < TIMEOUT
            except Exception as e:
                logger.warning(f"ExactSolver failed: {e}")
                res = {col: np.nan for col in [
                    'final_error', 'final_relative_error', 'total_runtime',
                    'avg_iter_time', 'total_nnls_time', 'total_qubo_build_time',
                    'total_annealing_time', 'total_postprocess_time',
                    'num_qubo_solves', 'avg_qubo_solve_time',
                ]}
                feasible = False
            rows.append({
                'data_source': 'feasibility',
                'matrix_size': f"{n}x{m}",
                'n': n, 'm': m, 'rank': k,
                'sampler': 'exact',
                'seed': seed,
                'feasible': feasible,
                **res,
            })

    df = pd.DataFrame(rows)
    out_path = os.path.join(cfg['general']['output_dir'], 'scalability_results.csv')
    df.to_csv(out_path, index=False)
    logger.info(f"Saved scalability results to {out_path} ({len(df)} rows)")

    print("\n=== SCALABILITY SUMMARY (means) ===")
    non_feas = df[df['data_source'] != 'feasibility']
    if not non_feas.empty:
        summary = non_feas.groupby(['data_source', 'sampler', 'matrix_size', 'rank']).agg({
            'final_relative_error': 'mean',
            'total_runtime': 'mean',
        }).round(4)
        print(summary.to_string())


if __name__ == "__main__":
    main()
