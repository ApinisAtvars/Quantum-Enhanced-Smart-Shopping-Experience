"""
Tier 1 — Correctness / sanity checks.

Validates QUBO construction, ExactSolver ground-truth recovery, and sampler
approximation quality on small synthetic problems with known W_true, H_true.

Produces results/correctness_results.csv.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.synthetic import generate_synthetic
from utils.logging import get_logger
from models.nbmf import NBMF
from models.qubo import build_qubo_dict, qubo_energy

logger = get_logger("Correctness")


def run_correctness_single(V, W_true, H_true, k, sampler_name, num_iterations,
                           num_reads, num_sweeps, seed):
    n, m = V.shape
    model = NBMF(
        n=n, m=m, k=k,
        sampler_name=sampler_name,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        seed=seed,
    )
    result = model.fit(V, num_iterations=num_iterations, verbose=False)

    H_pred = result['H']
    h_match = float(np.mean(np.abs(H_pred - H_true) < 0.5))

    recon_error = result['final_error']
    rel_error = result['final_relative_error']
    total_runtime = result['timing']['total_runtime']

    return {
        'h_recovery_rate': h_match,
        'final_error': recon_error,
        'final_relative_error': rel_error,
        'total_runtime': total_runtime,
    }


def main():
    with open('config/experiment.yaml') as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg['general']['output_dir'], exist_ok=True)
    seeds = cfg['general']['seeds']
    num_iterations = cfg['general']['num_iterations']
    sa_cfg = cfg['samplers']['simulated_annealing']
    pi_cfg = cfg['samplers']['path_integral']

    rows = []

    for size in cfg['correctness']['sizes']:
        n, m = size
        for k in cfg['correctness']['ranks']:
            for seed in seeds:
                V, W_true, H_true, _ = generate_synthetic(n, m, k, noise_std=0.0, seed=seed)

                samplers_to_run = [
                    ('NBMF_Exact', 'exact', 0, 0),
                    ('NBMF_SA', 'simulated_annealing', sa_cfg['num_reads'], sa_cfg['num_sweeps']),
                    ('NBMF_PathIntegral', 'path_integral', pi_cfg['num_reads'], 0),
                ]

                for method_name, sname, nreads, nsweeps in samplers_to_run:
                    logger.info(f"{method_name} | {n}x{m} | k={k} | seed={seed}")
                    res = run_correctness_single(
                        V, W_true, H_true, k, sname,
                        num_iterations, nreads, nsweeps, seed,
                    )
                    rows.append({
                        'method': method_name,
                        'matrix_size': f'{n}x{m}',
                        'n': n, 'm': m,
                        'rank': k,
                        'seed': seed,
                        'h_recovery_rate': res['h_recovery_rate'],
                        'final_error': res['final_error'],
                        'final_relative_error': res['final_relative_error'],
                        'total_runtime': res['total_runtime'],
                    })

    df = pd.DataFrame(rows)
    out_path = os.path.join(cfg['general']['output_dir'], 'correctness_results.csv')
    df.to_csv(out_path, index=False)
    logger.info(f"Saved correctness results to {out_path} ({len(df)} rows)")

    print("\n=== CORRECTNESS SUMMARY ===")
    summary = df.groupby(['method', 'rank']).agg({
        'h_recovery_rate': ['mean', 'std'],
        'final_error': ['mean', 'std'],
        'total_runtime': ['mean', 'std'],
    }).round(6)
    print(summary.to_string())


if __name__ == "__main__":
    main()
