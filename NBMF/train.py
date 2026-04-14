import argparse
import os
import json
import time
import numpy as np
import pandas as pd
import yaml

from common.logging_utils import get_logger
from common.io_utils import save_json
from common.synthetic_data import generate_synthetic, synthetic_meta
from .model import NBMF

logger = get_logger("NBMF")


def main():
    parser = argparse.ArgumentParser(description="Train NBMF via QUBO")
    parser.add_argument('--config', type=str, default='config/experiment_config.yaml')
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--m', type=int, default=50)
    parser.add_argument('--rank', type=int, default=5)
    parser.add_argument('--sampler', type=str, default='simulated_annealing',
                        choices=['simulated_annealing', 'path_integral', 'tabu', 'steepest_descent', 'exact'])
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--num_reads', type=int, default=100)
    parser.add_argument('--num_sweeps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default='outputs/nbmf')
    parser.add_argument('--data_source', type=str, default='synthetic',
                        choices=['synthetic', 'movielens'])
    parser.add_argument('--data_path', type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed)

    if args.data_source == 'synthetic':
        logger.info(f"Generating synthetic data: {args.n}x{args.m}, rank={args.rank}")
        V, W_true, H_true, V_noisy = generate_synthetic(
            args.n, args.m, args.rank, seed=args.seed
        )
        meta = synthetic_meta(args.n, args.m, args.rank, 0.0, args.seed)
    else:
        if args.data_path is None:
            raise ValueError("--data_path required for movielens source")
        logger.info(f"Loading MovieLens subblock from {args.data_path}")
        V = np.load(args.data_path)
        meta = {'data_type': 'movielens', 'shape': list(V.shape)}
        args.n, args.m = V.shape

    logger.info(f"Training NBMF: sampler={args.sampler}, rank={args.rank}, "
                f"iters={args.iterations}, reads={args.num_reads}")

    model = NBMF(
        n=args.n, m=args.m, k=args.rank,
        sampler_name=args.sampler,
        num_reads=args.num_reads,
        num_sweeps=args.num_sweeps,
        seed=args.seed,
    )

    result = model.fit(V, num_iterations=args.iterations, verbose=True)

    history_df = pd.DataFrame(result['history'])
    history_df.to_csv(os.path.join(args.out, 'history.csv'), index=False)

    np.save(os.path.join(args.out, 'W.npy'), result['W'])
    np.save(os.path.join(args.out, 'H.npy'), result['H'])

    run_meta = {
        **meta,
        'sampler': args.sampler,
        'rank': args.rank,
        'num_iterations': args.iterations,
        'num_reads': args.num_reads,
        'num_sweeps': args.num_sweeps,
        'seed': args.seed,
        'final_error': result['final_error'],
        'final_relative_error': result['final_relative_error'],
        'timing': result['timing'],
    }
    save_json(run_meta, os.path.join(args.out, 'run_meta.json'))
    logger.info(f"Done. Final error={result['final_error']:.6f}")


if __name__ == "__main__":
    main()
