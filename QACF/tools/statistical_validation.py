from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class TestResult:
    metric: str
    n_pairs: int
    mean_hybrid: float
    mean_classical: float
    mean_delta: float
    ci_low: float
    ci_high: float
    p_value_two_sided: float
    effect_size_dz: float


def paired_bootstrap_ci(deltas: np.ndarray, iters: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = []
    n = len(deltas)
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        means.append(float(np.mean(deltas[idx])))
    arr = np.array(means, dtype=np.float64)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def paired_permutation_pvalue(deltas: np.ndarray, iters: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(float(np.mean(deltas)))
    n = len(deltas)
    count = 0
    for _ in range(iters):
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=n, replace=True)
        perm_mean = abs(float(np.mean(deltas * signs)))
        if perm_mean >= observed:
            count += 1
    return float((count + 1) / (iters + 1))


def effect_size_dz(deltas: np.ndarray) -> float:
    std = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
    if std < 1e-12:
        return 0.0
    return float(np.mean(deltas) / std)


def run_tests(stress_hybrid: pd.DataFrame, stress_classical: pd.DataFrame, metrics: list[str], iters: int, seed: int) -> list[TestResult]:
    merged = stress_hybrid[["sparsity_drop", *metrics]].merge(
        stress_classical[["sparsity_drop", *metrics]],
        on="sparsity_drop",
        suffixes=("_hybrid", "_classical"),
    ).sort_values("sparsity_drop")

    results: list[TestResult] = []
    for i, metric in enumerate(metrics):
        h = merged[f"{metric}_hybrid"].to_numpy(dtype=np.float64)
        c = merged[f"{metric}_classical"].to_numpy(dtype=np.float64)
        valid = np.isfinite(h) & np.isfinite(c)
        h = h[valid]
        c = c[valid]
        if len(h) < 3:
            continue

        d = h - c
        ci_low, ci_high = paired_bootstrap_ci(d, iters=iters, seed=seed + i)
        pval = paired_permutation_pvalue(d, iters=iters, seed=seed + 10_000 + i)
        dz = effect_size_dz(d)

        results.append(
            TestResult(
                metric=metric,
                n_pairs=int(len(d)),
                mean_hybrid=float(np.mean(h)),
                mean_classical=float(np.mean(c)),
                mean_delta=float(np.mean(d)),
                ci_low=ci_low,
                ci_high=ci_high,
                p_value_two_sided=pval,
                effect_size_dz=dz,
            )
        )

    return results


def build_markdown(results: list[TestResult], iters: int) -> str:
    lines = [
        "# UBCF Statistical Validation Summary",
        "",
        f"Bootstrap/permutation iterations: **{iters}**",
        "",
        "Method:",
        "- Paired comparison across matched sparsity levels",
        "- 95% bootstrap CI for mean delta (hybrid - classical)",
        "- Two-sided paired permutation p-value",
        "- Effect size: Cohen's dz",
        "",
        "| Metric | N | Hybrid Mean | Classical Mean | Mean Delta | 95% CI (Delta) | p-value | dz |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]

    for r in results:
        lines.append(
            f"| {r.metric} | {r.n_pairs} | {r.mean_hybrid:.6f} | {r.mean_classical:.6f} | {r.mean_delta:.6f} | [{r.ci_low:.6f}, {r.ci_high:.6f}] | {r.p_value_two_sided:.6f} | {r.effect_size_dz:.4f} |"
        )

    lines.extend(
        [
            "",
            "Interpretation guide:",
            "- CI excluding 0 suggests a consistent directional difference.",
            "- Lower p-values indicate stronger evidence against no difference.",
            "- |dz| around 0.2/0.5/0.8 is often interpreted as small/medium/large effect.",
            "",
            "Caveat:",
            "- These tests operate on sparsity-level aggregates, not per-user raw predictions.",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run statistical validation for hybrid vs classical UBCF outputs.")
    parser.add_argument("--input-dir", type=Path, default=Path("QACF/data/processed_32m_ubcf"))
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    in_dir = args.input_dir
    h_path = in_dir / "hybrid_ubcf_stress.csv"
    c_path = in_dir / "classical_only_stress.csv"

    if not h_path.exists() or not c_path.exists():
        raise FileNotFoundError(f"Missing required files: {h_path} and/or {c_path}")

    stress_hybrid = pd.read_csv(h_path)
    stress_classical = pd.read_csv(c_path)

    metrics = ["hr@10", "ndcg@10", "precision@10", "mrr@10", "map@10", "novelty"]
    results = run_tests(stress_hybrid, stress_classical, metrics=metrics, iters=args.iters, seed=args.seed)

    if not results:
        raise RuntimeError("No valid paired results could be computed. Check your stress CSV files.")

    out_csv = in_dir / "statistical_validation_summary.csv"
    out_md = in_dir / "statistical_validation_summary.md"
    out_json = in_dir / "statistical_validation_summary.json"

    pd.DataFrame([r.__dict__ for r in results]).to_csv(out_csv, index=False)
    out_md.write_text(build_markdown(results, iters=args.iters), encoding="utf-8")
    out_json.write_text(json.dumps([r.__dict__ for r in results], indent=2), encoding="utf-8")

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_md}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
