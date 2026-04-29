"""
Aggregate the primary benchmark CSV into a per-method summary with
mean, std, count, and 95% t-based CI for final_relative_error and total_runtime.

Writes: results_final/benchmark_primary_summary.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import yaml
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _ci95(series):
    x = series.dropna().values
    n = len(x)
    if n < 2:
        return (float('nan'), float('nan'))
    mean = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, df=n - 1)
    return (mean - tcrit * se, mean + tcrit * se)


def main():
    with open('config/experiment.yaml') as f:
        cfg = yaml.safe_load(f)
    out_dir = cfg['general']['output_dir']
    path = os.path.join(out_dir, 'benchmark_primary_results.csv')
    df = pd.read_csv(path)

    rows = []
    for method, sub in df.groupby('method'):
        for col in ['final_relative_error', 'total_runtime']:
            x = sub[col].dropna()
            lo, hi = _ci95(x)
            rows.append({
                'method': method,
                'metric': col,
                'n': int(len(x)),
                'mean': float(x.mean()),
                'std': float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                'ci_lower': lo,
                'ci_upper': hi,
            })

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, 'benchmark_primary_summary.csv')
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(out_df)} rows)")

    print("\n=== PRIMARY BENCHMARK SUMMARY ===")
    pivot = out_df.pivot_table(
        index='method', columns='metric',
        values=['mean', 'std', 'ci_lower', 'ci_upper', 'n'],
    )
    print(pivot.round(6).to_string())


if __name__ == "__main__":
    main()
