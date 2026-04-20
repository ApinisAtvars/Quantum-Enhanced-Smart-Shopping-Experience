from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "QACF" / "analysis_outputs" / "experimental_improvements"


SOURCES = {
    "MovieLens-UBCF": {
        "summary": ROOT / "QACF" / "data" / "processed_32m_ubcf" / "hybrid_results_summary.json",
        "sparsity_candidates": [
            ROOT / "QACF" / "data" / "processed_32m_ubcf" / "sparsity_stress_test_results.csv",
            ROOT / "QACF" / "data" / "processed_32m_ubcf" / "hybrid_vs_classical_uplift.csv",
        ],
    },
    "Amazon-Books": {
        "summary": ROOT / "QACF" / "AmazonReviews" / "data" / "processed_amazon" / "hybrid_results_summary.json",
        "sparsity_candidates": [
            ROOT / "QACF" / "AmazonReviews" / "data" / "processed_amazon" / "sparsity_stress_test_results.csv",
            ROOT / "QACF" / "AmazonReviews" / "data" / "processed_amazon" / "hybrid_vs_classical_uplift.csv",
        ],
    },
    "Amazon-Appliances": {
        "summary": ROOT
        / "QACF"
        / "AmazonReviews"
        / "AppliencesReviews"
        / "notebooks"
        / "data"
        / "processed_appliences"
        / "hybrid_results_summary.json",
        "sparsity_candidates": [
            ROOT
            / "QACF"
            / "AmazonReviews"
            / "AppliencesReviews"
            / "notebooks"
            / "data"
            / "processed_appliences"
            / "sparsity_stress_test_results.csv",
            ROOT
            / "QACF"
            / "AmazonReviews"
            / "AppliencesReviews"
            / "notebooks"
            / "data"
            / "processed_appliences"
            / "hybrid_vs_classical_uplift.csv",
        ],
    },
    "Amazon-Fashion (BeautyReviews alias)": {
        "summary": ROOT
        / "QACF"
        / "AmazonReviews"
        / "BeautyReviews"
        / "notebooks"
        / "data"
        / "processed_beauty"
        / "hybrid_results_summary.json",
        "sparsity_candidates": [
            ROOT
            / "QACF"
            / "AmazonReviews"
            / "BeautyReviews"
            / "notebooks"
            / "data"
            / "processed_beauty"
            / "sparsity_stress_test_results.csv",
            ROOT
            / "QACF"
            / "AmazonReviews"
            / "BeautyReviews"
            / "notebooks"
            / "data"
            / "processed_beauty"
            / "hybrid_vs_classical_uplift.csv",
        ],
    },
}

RUNTIME_PATH = ROOT / "QACF" / "data" / "processed_32m_ubcf" / "runtime_complexity_ubcf.csv"
ABLATION_SUMMARY_PATH = (
    ROOT
    / "QACF"
    / "AmazonReviews"
    / "data"
    / "processed_amazon"
    / "ablation_summary_table_amazon_vs_ubcf.csv"
)
ABLATION_QUANTUM_PATH = (
    ROOT
    / "QACF"
    / "AmazonReviews"
    / "data"
    / "processed_amazon"
    / "ablation_quantum_amazon.csv"
)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _resolve_existing_path(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def build_baseline_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, paths in SOURCES.items():
        summary = _load_json(paths["summary"])
        if not summary:
            continue

        at0 = summary.get("results_at_0_drop", {})
        rows.append(
            {
                "dataset": dataset,
                "hr10_classical": at0.get("classical_hr10"),
                "hr10_hybrid": at0.get("hybrid_hr10"),
                "ndcg10_classical": at0.get("classical_ndcg10"),
                "ndcg10_hybrid": at0.get("hybrid_ndcg10"),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "baseline_comparison_table.csv", index=False)
    return out


def plot_baseline_figure(df: pd.DataFrame) -> None:
    if df.empty:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("talk", font_scale=0.85)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    hr = (
        df[["dataset", "hr10_classical", "hr10_hybrid"]]
        .melt(id_vars="dataset", var_name="model", value_name="value")
        .dropna()
    )
    hr["model"] = hr["model"].str.replace("hr10_", "", regex=False).str.title()
    sns.barplot(data=hr, x="dataset", y="value", hue="model", ax=axes[0])
    axes[0].set_title("Baseline HR@10 (Drop=0)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("HR@10")
    axes[0].tick_params(axis="x", rotation=15)

    ndcg = (
        df[["dataset", "ndcg10_classical", "ndcg10_hybrid"]]
        .melt(id_vars="dataset", var_name="model", value_name="value")
        .dropna()
    )
    if not ndcg.empty:
        ndcg["model"] = ndcg["model"].str.replace("ndcg10_", "", regex=False).str.title()
        sns.barplot(data=ndcg, x="dataset", y="value", hue="model", ax=axes[1])
        axes[1].set_title("Baseline NDCG@10 (Drop=0)")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("NDCG@10")
        axes[1].tick_params(axis="x", rotation=15)
    else:
        axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "baseline_comparison_figure.png", dpi=300)
    plt.close(fig)


def build_and_plot_sparsity() -> pd.DataFrame:
    records: list[pd.DataFrame] = []

    for dataset, paths in SOURCES.items():
        path = _resolve_existing_path(paths.get("sparsity_candidates", []))
        if path is None:
            continue

        df = pd.read_csv(path)
        sparsity_col = _pick_col(df, ["sparsity_level", "sparsity_drop"])
        classical_col = _pick_col(df, ["hr10_classical", "hr@10_classical"])
        hybrid_col = _pick_col(df, ["hr10_hybrid", "hr@10_hybrid"])

        if not sparsity_col or not classical_col or not hybrid_col:
            continue

        view = df[[sparsity_col, classical_col, hybrid_col]].copy()
        view.columns = ["sparsity", "classical", "hybrid"]
        view["dataset"] = dataset
        records.append(view)

    if not records:
        return pd.DataFrame()

    all_df = pd.concat(records, ignore_index=True)
    all_df["dropout_pct"] = all_df["sparsity"] * 100.0
    all_df.to_csv(OUT_DIR / "sparsity_hr10_curve_table.csv", index=False)

    sns.set_context("talk", font_scale=0.8)
    datasets = sorted(all_df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.8), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        sub = all_df[all_df["dataset"] == dataset].sort_values("dropout_pct")
        ax.plot(sub["dropout_pct"], sub["classical"], marker="o", label="Classical")
        ax.plot(sub["dropout_pct"], sub["hybrid"], marker="o", label="Hybrid")
        ax.set_title(dataset)
        ax.set_xlabel("Sparsity / Dropout (%)")
        ax.set_ylabel("HR@10")
        ax.legend()

    fig.suptitle("HR@10 vs Sparsity: Classical vs Hybrid", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sparsity_hr10_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return all_df


def build_and_plot_ablation() -> pd.DataFrame:
    if not ABLATION_SUMMARY_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(ABLATION_SUMMARY_PATH)
    amazon = df[df["dataset"].str.lower() == "amazon"].copy()
    amazon = amazon[
        amazon["variant"].isin(["full", "without_dwave", "without_mmr", "without_novelty", "without_shrinkage"])
    ]
    if amazon.empty:
        return pd.DataFrame()

    amazon = amazon[["variant", "hr10_hybrid_mean", "hr10_delta_vs_classical"]].copy()
    amazon = amazon.rename(
        columns={
            "variant": "component_variant",
            "hr10_hybrid_mean": "hr10_hybrid",
            "hr10_delta_vs_classical": "hr10_delta_vs_classical",
        }
    )

    label_map = {
        "full": "Full",
        "without_dwave": "No QUBO (D-Wave)",
        "without_mmr": "No MMR",
        "without_novelty": "No Novelty",
        "without_shrinkage": "No Shrinkage",
    }
    amazon["component_variant"] = amazon["component_variant"].map(label_map).fillna(amazon["component_variant"])

    q_methods = None
    if ABLATION_QUANTUM_PATH.exists():
        q_methods = pd.read_csv(ABLATION_QUANTUM_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.barplot(data=amazon, x="component_variant", y="hr10_hybrid", ax=axes[0], color="#4c72b0")
    axes[0].set_title("Ablation: Hybrid HR@10 by Component")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Mean HR@10")
    axes[0].tick_params(axis="x", rotation=25)

    if q_methods is not None and {"method", "silhouette"}.issubset(set(q_methods.columns)):
        q_methods = q_methods.copy()
        q_methods["method"] = q_methods["method"].replace(
            {
                "angle+IQP": "Angle + IQP",
                "amplitude-only": "No IQP (Amplitude)",
            }
        )
        sns.barplot(data=q_methods, x="method", y="silhouette", ax=axes[1], color="#55a868")
        axes[1].set_title("Quantum Component Sensitivity (Silhouette)")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Silhouette")
        axes[1].tick_params(axis="x", rotation=20)
    else:
        axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "ablation_summary_figure.png", dpi=300)
    plt.close(fig)

    amazon.to_csv(OUT_DIR / "ablation_summary_table.csv", index=False)
    return amazon


def build_config_and_runtime_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    config_rows: list[dict[str, Any]] = []
    for dataset, paths in SOURCES.items():
        summary = _load_json(paths["summary"])
        if not summary:
            continue

        conf = summary.get("configuration", {})
        q = summary.get("quantum_routing", {})

        config_rows.append(
            {
                "dataset": dataset,
                "eval_users": conf.get("eval_users"),
                "candidate_pool": conf.get("candidate_pool"),
                "topn": conf.get("topn"),
                "routing_groups": q.get("refined_routing_groups"),
                "d_wave_qubo_clusters": q.get("d_wave_qubo_clusters"),
                "qubits": "6 (UBCF), 5-6 (Amazon)",
                "encoding_type": "Angle + IQP (primary), amplitude/angle fusion (Amazon)",
            }
        )

    config_df = pd.DataFrame(config_rows)
    config_df.to_csv(OUT_DIR / "experimental_configuration_table.csv", index=False)

    runtime_rows: list[dict[str, Any]] = []
    if RUNTIME_PATH.exists():
        rt = pd.read_csv(RUNTIME_PATH)
        runtime_rows.append(
            {
                "component": "Quantum kernel micro-benchmark (UBCF)",
                "n_range": f"{int(rt['n'].min())}-{int(rt['n'].max())}",
                "runtime_sec_mean": rt["runtime_sec"].mean(),
                "runtime_sec_min": rt["runtime_sec"].min(),
                "runtime_sec_max": rt["runtime_sec"].max(),
                "complexity": "O(N^2 q^2)",
                "note": "Measured from runtime_complexity_ubcf.csv",
            }
        )

    runtime_rows.extend(
        [
            {
                "component": "Classical routing / neighborhood scoring",
                "n_range": "dataset dependent",
                "runtime_sec_mean": None,
                "runtime_sec_min": None,
                "runtime_sec_max": None,
                "complexity": "Approx. O(Nk) per query",
                "note": "Fast at inference; lower overhead than pairwise quantum kernels",
            },
            {
                "component": "Hybrid orchestration",
                "n_range": "dataset dependent",
                "runtime_sec_mean": None,
                "runtime_sec_min": None,
                "runtime_sec_max": None,
                "complexity": "Classical + selective quantum path",
                "note": "Trades compute cost for robustness under high sparsity",
            },
        ]
    )

    runtime_df = pd.DataFrame(runtime_rows)
    runtime_df.to_csv(OUT_DIR / "runtime_comparison_summary.csv", index=False)

    return config_df, runtime_df


def write_markdown_summary(
    baseline_df: pd.DataFrame,
    sparsity_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    config_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Experimental Improvements Addendum\n")
    lines.append("This file is auto-generated by `QACF/tools/generate_experimental_improvements.py`.\n")

    lines.append("## Added Figures\n")
    lines.append("- `baseline_comparison_figure.png`: baseline HR@10/NDCG@10 visual comparison of classical vs hybrid.")
    lines.append("- `sparsity_hr10_comparison.png`: HR@10 vs sparsity (classical vs hybrid) across available datasets.")
    lines.append("- `ablation_summary_figure.png`: component-removal ablation summary (includes No QUBO and No IQP proxies where available).\n")

    lines.append("## Baseline Summary (Drop=0)\n")
    if not baseline_df.empty:
        lines.append(baseline_df.to_markdown(index=False))
    else:
        lines.append("No baseline summary data found.")

    lines.append("\n## Experimental Configuration\n")
    if not config_df.empty:
        lines.append(config_df.to_markdown(index=False))
    else:
        lines.append("No configuration summaries found.")

    lines.append("\n## Runtime Comparison Summary\n")
    lines.append(runtime_df.to_markdown(index=False))
    lines.append(
        "\nPractical trade-off: classical routing has lower compute overhead, while quantum/hybrid paths increase cost but can improve robustness under severe sparsity in selected regimes."
    )

    lines.append("\n## Data Coverage\n")
    lines.append(f"- Baseline rows: {len(baseline_df)}")
    lines.append(f"- Sparsity rows: {len(sparsity_df)}")
    lines.append(f"- Ablation rows: {len(ablation_df)}")

    (OUT_DIR / "experimental_improvements_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_df = build_baseline_table()
    plot_baseline_figure(baseline_df)

    sparsity_df = build_and_plot_sparsity()
    ablation_df = build_and_plot_ablation()
    config_df, runtime_df = build_config_and_runtime_tables()

    write_markdown_summary(baseline_df, sparsity_df, ablation_df, config_df, runtime_df)

    print(f"Saved experimental-improvement outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
