"""
Generate all plots from results/*.csv into plots/*.png.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = 'results'
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

METHOD_COLORS = {
    'Repo_ALS': '#1f77b4',
    'Repo_SGD': '#ff7f0e',
    'NBMF_SA': '#2ca02c',
    'NBMF_PathIntegral': '#e377c2',
    'NBMF_Tabu': '#d62728',
    'NBMF_SteepestDescent': '#9467bd',
    'NBMF_Exact': '#8c564b',
    'NMF_MU': '#7f7f7f',
    'NMF_SGD': '#bcbd22',
}

METHOD_LABELS = {
    'Repo_ALS': 'ALS (dense)',
    'Repo_SGD': 'SGD (dense)',
    'NBMF_SA': 'NBMF + Sim. Annealing',
    'NBMF_PathIntegral': 'NBMF + Path Integral',
    'NBMF_Tabu': 'NBMF + Tabu',
    'NBMF_SteepestDescent': 'NBMF + Steepest Desc.',
    'NBMF_Exact': 'NBMF + Exact',
    'NMF_MU': 'NMF (Mult. Updates)',
    'NMF_SGD': 'NMF (Proj. SGD)',
}

SAMPLER_LABELS = {
    'simulated_annealing': 'Sim. Annealing',
    'path_integral': 'Path Integral',
    'exact': 'Exact',
}


def _label(m):
    return METHOD_LABELS.get(m, m)

def _color(m):
    return METHOD_COLORS.get(m, '#333333')


def plot_convergence_primary(conv_df):
    """Error vs iteration for primary methods on MovieLens benchmark."""
    primary = conv_df[conv_df['tier'] == 'primary'] if 'tier' in conv_df.columns else conv_df
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in sorted(primary['method'].unique()):
        sub = primary[primary['method'] == method]
        grouped = sub.groupby('iteration')['error']
        mean_err = grouped.mean()
        std_err = grouped.std().fillna(0)
        ax.plot(mean_err.index, mean_err.values, color=_color(method),
                label=_label(method), linewidth=2)
        ax.fill_between(mean_err.index,
                        (mean_err - std_err).values,
                        (mean_err + std_err).values,
                        alpha=0.15, color=_color(method))
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Reconstruction Error (Frobenius)', fontsize=12)
    ax.set_title('Convergence: Primary Methods on MovieLens Subblock', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'convergence_primary.png'), dpi=150)
    plt.close(fig)
    print("  -> convergence_primary.png")


def plot_convergence_small(corr_df):
    """Final error bar chart for correctness experiment (small synthetic)."""
    ranks = sorted(corr_df['rank'].unique())
    fig, axes = plt.subplots(1, len(ranks), figsize=(6 * len(ranks), 5))
    if len(ranks) == 1:
        axes = [axes]

    for ax, rank in zip(axes, ranks):
        sub = corr_df[corr_df['rank'] == rank]
        methods = sorted(sub['method'].unique())
        for i, method in enumerate(methods):
            msub = sub[sub['method'] == method]
            ax.bar(i, msub['final_error'].mean(),
                   yerr=msub['final_error'].std(),
                   color=_color(method), capsize=5, label=_label(method))
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([_label(m) for m in methods], rotation=15, ha='right', fontsize=9)
        ax.set_title(f'Rank = {rank}', fontsize=12)
        ax.set_ylabel('Final Reconstruction Error', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Correctness: Small Synthetic (Exact vs SA)', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'correctness_small.png'), dpi=150)
    plt.close(fig)
    print("  -> correctness_small.png")


def plot_benchmark_bars(primary_df):
    """Side-by-side bars for final error and runtime across primary methods."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    grouped = primary_df.groupby('method')
    methods = sorted(primary_df['method'].unique())
    x = np.arange(len(methods))
    colors = [_color(m) for m in methods]

    err_mean = [grouped.get_group(m)['final_relative_error'].mean() for m in methods]
    err_std = [grouped.get_group(m)['final_relative_error'].std() for m in methods]
    axes[0].bar(x, err_mean, yerr=err_std, color=colors, capsize=5,
                edgecolor='black', linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([_label(m) for m in methods], rotation=20, ha='right', fontsize=10)
    axes[0].set_ylabel('Relative Error (Frobenius)', fontsize=12)
    axes[0].set_title('Final Relative Error', fontsize=13)
    axes[0].grid(True, alpha=0.3, axis='y')

    rt_mean = [grouped.get_group(m)['total_runtime'].mean() for m in methods]
    rt_std = [grouped.get_group(m)['total_runtime'].std() for m in methods]
    axes[1].bar(x, rt_mean, yerr=rt_std, color=colors, capsize=5,
                edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([_label(m) for m in methods], rotation=20, ha='right', fontsize=10)
    axes[1].set_ylabel('Total Runtime (s)', fontsize=12)
    axes[1].set_title('Total Runtime', fontsize=13)
    axes[1].grid(True, alpha=0.3, axis='y')

    fig.suptitle('Primary Benchmark on MovieLens Subblock', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'benchmark_bars.png'), dpi=150)
    plt.close(fig)
    print("  -> benchmark_bars.png")


def plot_runtime_breakdown(breakdown_df):
    """Stacked bar chart of runtime components for NBMF variants."""
    methods = sorted(breakdown_df['method'].unique())
    means = breakdown_df.groupby('method').agg({
        'nnls_time': 'mean',
        'qubo_build_time': 'mean',
        'annealing_time': 'mean',
        'postprocess_time': 'mean',
    }).reindex(methods)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    w = 0.5
    bottom = np.zeros(len(methods))
    components = [
        ('nnls_time', 'NNLS', '#1f77b4'),
        ('qubo_build_time', 'QUBO Build', '#ff7f0e'),
        ('annealing_time', 'Sampler', '#2ca02c'),
        ('postprocess_time', 'Postprocess', '#d62728'),
    ]
    for col, label, color in components:
        vals = means[col].fillna(0).values
        ax.bar(x, vals, w, bottom=bottom, label=label, color=color)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([_label(m) for m in methods], rotation=15, ha='right')
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_title('Runtime Breakdown by Component (NBMF variants)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'runtime_breakdown_stacked.png'), dpi=150)
    plt.close(fig)
    print("  -> runtime_breakdown_stacked.png")


def plot_scalability(scale_df):
    """Runtime and error vs matrix size and rank."""
    non_feas = scale_df[scale_df['data_source'] != 'feasibility']
    if non_feas.empty:
        print("  -> scalability plots SKIPPED (no data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for source_label, sub_df in non_feas.groupby('data_source'):
        for sampler in sub_df['sampler'].unique():
            ssub = sub_df[sub_df['sampler'] == sampler]
            for k in sorted(ssub['rank'].unique()):
                ksub = ssub[ssub['rank'] == k]
                grouped = ksub.groupby('n')['total_runtime'].agg(['mean', 'std']).reset_index()
                label = f'{source_label} {SAMPLER_LABELS.get(sampler, sampler)} k={k}'
                axes[0].errorbar(grouped['n'], grouped['mean'], yerr=grouped['std'],
                                 marker='o', linewidth=1.5, capsize=3, label=label)

    axes[0].set_xlabel('Matrix Size (n)', fontsize=12)
    axes[0].set_ylabel('Total Runtime (s)', fontsize=12)
    axes[0].set_title('Runtime vs Matrix Size', fontsize=13)
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)

    for source_label, sub_df in non_feas.groupby('data_source'):
        for sampler in sub_df['sampler'].unique():
            ssub = sub_df[sub_df['sampler'] == sampler]
            for sz in sorted(ssub['matrix_size'].unique(), key=lambda s: int(s.split('x')[0])):
                szsub = ssub[ssub['matrix_size'] == sz]
                grouped = szsub.groupby('rank')['final_relative_error'].agg(['mean', 'std']).reset_index()
                label = f'{source_label} {sz}'
                axes[1].errorbar(grouped['rank'], grouped['mean'], yerr=grouped['std'],
                                 marker='s', linewidth=1.5, capsize=3, label=label)

    axes[1].set_xlabel('Rank (k)', fontsize=12)
    axes[1].set_ylabel('Final Relative Error', fontsize=12)
    axes[1].set_title('Error vs Rank', fontsize=13)
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'scalability.png'), dpi=150)
    plt.close(fig)
    print("  -> scalability.png")

    # Feasibility
    feas = scale_df[scale_df['data_source'] == 'feasibility']
    if not feas.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for sz in sorted(feas['matrix_size'].unique(), key=lambda s: int(s.split('x')[0])):
            sub = feas[feas['matrix_size'] == sz].sort_values('rank')
            ax.plot(sub['rank'], sub['total_runtime'], marker='D', linewidth=2, label=sz)
        ax.set_xlabel('Rank (k)', fontsize=12)
        ax.set_ylabel('Total Runtime (s)', fontsize=12)
        ax.set_title('ExactSolver Feasibility: Runtime vs Rank', fontsize=13)
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, 'feasibility_exact.png'), dpi=150)
        plt.close(fig)
        print("  -> feasibility_exact.png")


def plot_convergence_by_rank(conv_detail_df):
    """Convergence curves per rank on synthetic data."""
    ranks = sorted(conv_detail_df['rank'].unique())
    n_ranks = len(ranks)
    fig, axes = plt.subplots(1, n_ranks, figsize=(6 * n_ranks, 5), sharey=False)
    if n_ranks == 1:
        axes = [axes]

    for ax, rank in zip(axes, ranks):
        sub = conv_detail_df[conv_detail_df['rank'] == rank]
        for method in sorted(sub['method'].unique()):
            msub = sub[sub['method'] == method]
            grouped = msub.groupby('iteration')['error']
            mean_err = grouped.mean()
            std_err = grouped.std().fillna(0)
            ax.plot(mean_err.index, mean_err.values, color=_color(method),
                    label=_label(method), linewidth=2)
            ax.fill_between(mean_err.index,
                            (mean_err - std_err).values,
                            (mean_err + std_err).values,
                            alpha=0.12, color=_color(method))
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_title(f'Rank = {rank}', fontsize=12)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel('Reconstruction Error', fontsize=11)
        ax.legend(fontsize=7)

    fig.suptitle('Convergence by Rank (Synthetic)', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'convergence_by_rank.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  -> convergence_by_rank.png")


def plot_convergence_wallclock(conv_detail_df):
    """Error vs cumulative wall-clock time."""
    if 'cumulative_time' not in conv_detail_df.columns:
        print("  -> convergence_wallclock.png SKIPPED (no cumulative_time)")
        return

    ranks = sorted(conv_detail_df['rank'].unique())
    n_ranks = len(ranks)
    fig, axes = plt.subplots(1, n_ranks, figsize=(6 * n_ranks, 5), sharey=False)
    if n_ranks == 1:
        axes = [axes]

    for ax, rank in zip(axes, ranks):
        sub = conv_detail_df[conv_detail_df['rank'] == rank]
        for method in sorted(sub['method'].unique()):
            msub = sub[sub['method'] == method]
            grouped = msub.groupby('iteration').agg(
                mean_err=('error', 'mean'),
                mean_time=('cumulative_time', 'mean'),
            ).reset_index()
            ax.plot(grouped['mean_time'], grouped['mean_err'],
                    color=_color(method), label=_label(method), linewidth=2, marker='.')
        ax.set_xlabel('Cumulative Wall-Clock Time (s)', fontsize=11)
        ax.set_title(f'Rank = {rank}', fontsize=12)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel('Reconstruction Error', fontsize=11)
        ax.legend(fontsize=7)

    fig.suptitle('Error vs Wall-Clock Time (Synthetic)', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'convergence_wallclock.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  -> convergence_wallclock.png")


def main():
    print("Generating plots from results/*.csv ...")

    conv_path = os.path.join(RESULTS_DIR, 'convergence_results.csv')
    conv_detail_path = os.path.join(RESULTS_DIR, 'convergence_detailed.csv')
    primary_path = os.path.join(RESULTS_DIR, 'benchmark_primary_results.csv')
    corr_path = os.path.join(RESULTS_DIR, 'correctness_results.csv')
    breakdown_path = os.path.join(RESULTS_DIR, 'runtime_breakdown_results.csv')
    scale_path = os.path.join(RESULTS_DIR, 'scalability_results.csv')

    if os.path.exists(conv_path):
        plot_convergence_primary(pd.read_csv(conv_path))

    if os.path.exists(corr_path):
        plot_convergence_small(pd.read_csv(corr_path))

    if os.path.exists(primary_path):
        plot_benchmark_bars(pd.read_csv(primary_path))

    if os.path.exists(breakdown_path):
        plot_runtime_breakdown(pd.read_csv(breakdown_path))

    if os.path.exists(scale_path):
        plot_scalability(pd.read_csv(scale_path))

    if os.path.exists(conv_detail_path):
        conv_detail_df = pd.read_csv(conv_detail_path)
        plot_convergence_by_rank(conv_detail_df)
        plot_convergence_wallclock(conv_detail_df)

    print("\nAll plots saved to plots/")


if __name__ == "__main__":
    main()
