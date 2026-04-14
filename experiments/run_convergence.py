"""
Convergence analysis on synthetic problems with known ground truth.

Tracks error vs iteration AND error vs cumulative wall-clock time for
NBMF samplers and the adapted repo ALS/SGD baselines.

Produces results/convergence_detailed.csv.
"""
import os
import sys
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.synthetic_data import generate_synthetic
from common.logging_utils import get_logger
from NBMF.model import NBMF
from experiments.baseline_wrappers import run_repo_als, run_repo_sgd

logger = get_logger("Convergence")


def run_nbmf_convergence(V, k, sampler_name, num_iterations, num_reads, num_sweeps, seed):
    n, m = V.shape
    model = NBMF(
        n=n, m=m, k=k,
        sampler_name=sampler_name,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        seed=seed,
    )
    result = model.fit(V, num_iterations=num_iterations, verbose=False)
    cum_time = 0.0
    for h in result['history']:
        cum_time += h['iter_time']
        h['cumulative_time'] = cum_time
    return result['history']


def main():
    with open('config/experiment_config.yaml') as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg['general']['output_dir'], exist_ok=True)
    seeds = cfg['general']['seeds']
    num_iterations = cfg['general']['num_iterations']
    sa_cfg = cfg['samplers']['simulated_annealing']
    pi_cfg = cfg['samplers']['path_integral']

    conv_cfg = cfg['convergence']
    n, m = conv_cfg['synthetic_size']
    ranks = conv_cfg['ranks']

    rows = []

    for k in ranks:
        for seed in seeds:
            V, W_true, H_true, _ = generate_synthetic(n, m, k, seed=seed)

            samplers = [
                ('NBMF_SA', 'simulated_annealing', sa_cfg['num_reads'], sa_cfg['num_sweeps']),
                ('NBMF_PathIntegral', 'path_integral', pi_cfg['num_reads'], 0),
                ('NBMF_Exact', 'exact', 0, 0),
            ]

            for method_name, sname, nreads, nsweeps in samplers:
                logger.info(f"{method_name} | {n}x{m} | k={k} | seed={seed}")
                hist = run_nbmf_convergence(V, k, sname, num_iterations, nreads, nsweeps, seed)
                for h in hist:
                    rows.append({
                        **h,
                        'method': method_name,
                        'data_source': 'synthetic',
                        'matrix_size': f'{n}x{m}',
                        'rank': k,
                        'seed': seed,
                    })

            # Repo ALS
            logger.info(f"Repo_ALS | {n}x{m} | k={k} | seed={seed}")
            hist_als, _ = run_repo_als(V, k, num_iterations, seed, reg=0.0)
            cum_time = 0.0
            for h in hist_als:
                cum_time += h['iter_time']
                h['cumulative_time'] = cum_time
                rows.append({
                    **h,
                    'method': 'Repo_ALS',
                    'data_source': 'synthetic',
                    'matrix_size': f'{n}x{m}',
                    'rank': k,
                    'seed': seed,
                })

            # Repo SGD
            logger.info(f"Repo_SGD | {n}x{m} | k={k} | seed={seed}")
            hist_sgd, _ = run_repo_sgd(V, k, num_iterations, seed, lr=0.01, reg=0.0)
            cum_time = 0.0
            for h in hist_sgd:
                cum_time += h['iter_time']
                h['cumulative_time'] = cum_time
                rows.append({
                    **h,
                    'method': 'Repo_SGD',
                    'data_source': 'synthetic',
                    'matrix_size': f'{n}x{m}',
                    'rank': k,
                    'seed': seed,
                })

    df = pd.DataFrame(rows)
    out_path = os.path.join(cfg['general']['output_dir'], 'convergence_detailed.csv')
    df.to_csv(out_path, index=False)
    logger.info(f"Saved convergence data to {out_path} ({len(df)} rows)")

    print("\n=== CONVERGENCE SUMMARY (final errors, mean over seeds) ===")
    final = df[df['iteration'] == num_iterations]
    summary = final.groupby(['method', 'rank']).agg(
        error_mean=('error', 'mean'),
        error_std=('error', 'std'),
        count=('error', 'count'),
    ).round(6)
    print(summary.to_string())


if __name__ == "__main__":
    main()
