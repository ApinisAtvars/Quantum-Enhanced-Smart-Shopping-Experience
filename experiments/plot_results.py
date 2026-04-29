"""
Generate all plots from the final CSVs.

Reads:    results_final/*.csv
Writes:   plots_final/*.png

Required figures (see protocol_final.md Section 15):
  - convergence_primary.png
  - convergence_wallclock.png
  - convergence_by_rank.png
  - correctness_small.png
  - benchmark_error_bar.png
  - benchmark_runtime_bar.png
  - runtime_breakdown_stacked.png
  - scalability_runtime_vs_size.png
  - scalability_error_vs_rank.png
  - feasibility_exactsolver.png
"""
import os
import sys
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_cfg():
    with open('config/experiment.yaml') as f:
        return yaml.safe_load(f)


CFG = _load_cfg()
RESULTS_DIR = CFG['general']['output_dir']
PLOTS_DIR = CFG['general']['plots_dir']
os.makedirs(PLOTS_DIR, exist_ok=True)

METHOD_COLORS = {
    'Repo_ALS': '#1f77b4',
    'Repo_SGD': '#ff7f0e',
    'NBMF_SA': '#2ca02c',
    'NBMF_PathIntegral': '#e377c2',
    'NBMF_Exact': '#8c564b',
    'NMF_MU': '#7f7f7f',
}

METHOD_LABELS = {
    'Repo_ALS': 'ALS (dense)',
    'Repo_SGD': 'SGD (dense)',
    'NBMF_SA': 'NBMF + Sim. Annealing',
    'NBMF_PathIntegral': 'NBMF + Path Integral',
    'NBMF_Exact': 'NBMF + Exact',
    'NMF_MU': 'NMF (Mult. Updates)',
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


def plot_correctness_small(corr_df):
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
    fig.suptitle('Correctness: Small Synthetic (Exact vs SA vs PathIntegral)', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'correctness_small.png'), dpi=150)
    plt.close(fig)
    print("  -> correctness_small.png")


def plot_benchmark_bars(primary_df):
    """Two separate figures: error bar and runtime bar."""
    methods = sorted(primary_df['method'].unique())
    x = np.arange(len(methods))
    colors = [_color(m) for m in methods]
    grouped = primary_df.groupby('method')

    # Error
    err_mean = [grouped.get_group(m)['final_relative_error'].mean() for m in methods]
    err_std = [grouped.get_group(m)['final_relative_error'].std() for m in methods]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, err_mean, yerr=err_std, color=colors, capsize=5,
           edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(m) for m in methods], rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Final Relative Error (Frobenius)', fontsize=12)
    ax.set_title('Primary Benchmark — Final Relative Error', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'benchmark_error_bar.png'), dpi=150)
    plt.close(fig)
    print("  -> benchmark_error_bar.png")

    # Runtime
    rt_mean = [grouped.get_group(m)['total_runtime'].mean() for m in methods]
    rt_std = [grouped.get_group(m)['total_runtime'].std() for m in methods]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, rt_mean, yerr=rt_std, color=colors, capsize=5,
           edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(m) for m in methods], rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Total Runtime (s)', fontsize=12)
    ax.set_title('Primary Benchmark — Total Runtime', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'benchmark_runtime_bar.png'), dpi=150)
    plt.close(fig)
    print("  -> benchmark_runtime_bar.png")


def plot_runtime_breakdown(breakdown_df):
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
    non_feas = scale_df[scale_df['data_source'] != 'feasibility']
    if non_feas.empty:
        print("  -> scalability plots SKIPPED (no data)")
        return

    # Runtime vs size
    fig, ax = plt.subplots(figsize=(9, 6))
    for source_label, sub_df in non_feas.groupby('data_source'):
        for sampler in sorted(sub_df['sampler'].unique()):
            ssub = sub_df[sub_df['sampler'] == sampler]
            for k in sorted(ssub['rank'].unique()):
                ksub = ssub[ssub['rank'] == k]
                grouped = ksub.groupby('n')['total_runtime'].agg(['mean', 'std']).reset_index()
                label = f'{source_label} {SAMPLER_LABELS.get(sampler, sampler)} k={k}'
                ax.errorbar(grouped['n'], grouped['mean'], yerr=grouped['std'],
                            marker='o', linewidth=1.5, capsize=3, label=label)
    ax.set_xlabel('Matrix Size (n = m)', fontsize=12)
    ax.set_ylabel('Total Runtime (s)', fontsize=12)
    ax.set_title('Scalability — Runtime vs Matrix Size', fontsize=13)
    ax.set_yscale('log')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'scalability_runtime_vs_size.png'), dpi=150)
    plt.close(fig)
    print("  -> scalability_runtime_vs_size.png")

    # Error vs rank
    fig, ax = plt.subplots(figsize=(9, 6))
    for source_label, sub_df in non_feas.groupby('data_source'):
        for sampler in sorted(sub_df['sampler'].unique()):
            ssub = sub_df[sub_df['sampler'] == sampler]
            for sz in sorted(ssub['matrix_size'].unique(), key=lambda s: int(s.split('x')[0])):
                szsub = ssub[ssub['matrix_size'] == sz]
                grouped = szsub.groupby('rank')['final_relative_error'].agg(['mean', 'std']).reset_index()
                label = f'{source_label} {SAMPLER_LABELS.get(sampler, sampler)} {sz}'
                ax.errorbar(grouped['rank'], grouped['mean'], yerr=grouped['std'],
                            marker='s', linewidth=1.5, capsize=3, label=label)
    ax.set_xlabel('Rank (k)', fontsize=12)
    ax.set_ylabel('Final Relative Error', fontsize=12)
    ax.set_title('Scalability — Error vs Rank', fontsize=13)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'scalability_error_vs_rank.png'), dpi=150)
    plt.close(fig)
    print("  -> scalability_error_vs_rank.png")


def plot_feasibility(scale_df):
    feas = scale_df[scale_df['data_source'] == 'feasibility']
    if feas.empty:
        print("  -> feasibility_exactsolver.png SKIPPED (no data)")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for sz in sorted(feas['matrix_size'].unique(), key=lambda s: int(s.split('x')[0])):
        sub = feas[feas['matrix_size'] == sz].sort_values('rank')
        ax.plot(sub['rank'], sub['total_runtime'], marker='D', linewidth=2, label=sz)
    ax.set_xlabel('Rank (k)', fontsize=12)
    ax.set_ylabel('Total Runtime (s)', fontsize=12)
    ax.set_title('ExactSolver Feasibility — Runtime vs Rank', fontsize=13)
    ax.set_yscale('log')
    ax.legend(fontsize=10, title='matrix size')
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'feasibility_exactsolver.png'), dpi=150)
    plt.close(fig)
    print("  -> feasibility_exactsolver.png")


def plot_convergence_by_rank(conv_detail_df):
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
    fig.suptitle('Convergence by Rank (Synthetic 30x30)', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'convergence_by_rank.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  -> convergence_by_rank.png")


def plot_convergence_wallclock(conv_detail_df):
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
    fig.suptitle('Error vs Wall-Clock Time (Synthetic 30x30)', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'convergence_wallclock.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  -> convergence_wallclock.png")


def main():
    print(f"Generating plots from {RESULTS_DIR}/*.csv into {PLOTS_DIR}/ ...")

    paths = {
        'conv': os.path.join(RESULTS_DIR, 'convergence_results.csv'),
        'conv_detail': os.path.join(RESULTS_DIR, 'convergence_detailed.csv'),
        'primary': os.path.join(RESULTS_DIR, 'benchmark_primary_results.csv'),
        'correctness': os.path.join(RESULTS_DIR, 'correctness_results.csv'),
        'breakdown': os.path.join(RESULTS_DIR, 'runtime_breakdown_results.csv'),
        'scale': os.path.join(RESULTS_DIR, 'scalability_results.csv'),
    }

    if os.path.exists(paths['conv']):
        plot_convergence_primary(pd.read_csv(paths['conv']))
    if os.path.exists(paths['correctness']):
        plot_correctness_small(pd.read_csv(paths['correctness']))
    if os.path.exists(paths['primary']):
        plot_benchmark_bars(pd.read_csv(paths['primary']))
    if os.path.exists(paths['breakdown']):
        plot_runtime_breakdown(pd.read_csv(paths['breakdown']))
    if os.path.exists(paths['scale']):
        sdf = pd.read_csv(paths['scale'])
        plot_scalability(sdf)
        plot_feasibility(sdf)
    if os.path.exists(paths['conv_detail']):
        cdf = pd.read_csv(paths['conv_detail'])
        plot_convergence_by_rank(cdf)
        plot_convergence_wallclock(cdf)

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
