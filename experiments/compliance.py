"""
Machine-checkable compliance matrix for the final run.

Checks every numbered requirement in protocol_final.md Section 17 against
the regenerated CSVs in results_final/ and writes:

    results_final/compliance_matrix.csv

Also prints a human-readable summary to stdout with PASS/FAIL counts.
"""
import os
import sys
import re
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _expect_set(actual, expected, label):
    actual = set(actual)
    expected = set(expected)
    missing = expected - actual
    extra = actual - expected
    if not missing and not extra:
        return 'PASS', f'{label} exactly matches {sorted(expected)}'
    parts = []
    if missing:
        parts.append(f'missing={sorted(missing)}')
    if extra:
        parts.append(f'extra={sorted(extra)}')
    return 'FAIL', f'{label}: ' + '; '.join(parts)


def _timeout_from_code():
    path = 'experiments/scalability.py'
    with open(path) as f:
        txt = f.read()
    m = re.search(r'^TIMEOUT\s*=\s*(\d+)', txt, re.MULTILINE)
    return int(m.group(1)) if m else None


def main():
    with open('config/experiment.yaml') as f:
        cfg = yaml.safe_load(f)
    out_dir = cfg['general']['output_dir']
    rows = []

    def add(req_id, requirement, status, tier, evidence_file, evidence_detail):
        rows.append({
            'requirement_id': req_id,
            'requirement': requirement,
            'status': status,
            'tier': tier,
            'evidence_file': evidence_file,
            'evidence_detail': evidence_detail,
        })

    # --- Primary benchmark ---
    p_path = os.path.join(out_dir, 'benchmark_primary_results.csv')
    if os.path.exists(p_path):
        pdf = pd.read_csv(p_path)
        status, detail = _expect_set(
            pdf['method'].unique(),
            {'Repo_ALS', 'Repo_SGD', 'NBMF_SA', 'NBMF_PathIntegral'},
            'primary method set',
        )
        add('17.1', 'Primary benchmark has exactly {Repo_ALS, Repo_SGD, NBMF_SA, NBMF_PathIntegral}',
            status, 'primary', p_path, detail)

        expected_seeds = set(cfg['general']['seeds'])
        status, detail = _expect_set(pdf['seed'].unique(), expected_seeds, 'primary seed set')
        add('17.2', f'Primary benchmark uses exactly 5 seeds {sorted(expected_seeds)}',
            status, 'primary', p_path, detail)

        sizes = pdf['matrix_size'].unique()
        if len(sizes) == 1:
            add('17.3', 'Primary benchmark uses a single matrix size (actual extracted subblock)',
                'PASS', 'primary', p_path, f'matrix_size={sizes[0]}')
        else:
            add('17.3', 'Primary benchmark uses a single matrix size',
                'FAIL', 'primary', p_path, f'found {sorted(sizes)}')

        ranks = pdf['rank'].unique()
        if set(ranks) == {cfg['benchmark']['rank']}:
            add('17.4', f"Primary benchmark rank is {cfg['benchmark']['rank']} for every row",
                'PASS', 'primary', p_path, f'ranks={sorted(set(ranks))}')
        else:
            add('17.4', f"Primary benchmark rank is {cfg['benchmark']['rank']} for every row",
                'FAIL', 'primary', p_path, f'ranks={sorted(set(ranks))}')

        forbidden = {'NBMF_Exact', 'NBMF_Tabu', 'NBMF_SteepestDescent',
                     'NMF_MU', 'NMF_SGD'}
        present_forbidden = forbidden & set(pdf['method'].unique())
        if not present_forbidden:
            add('17.5', 'Primary benchmark has no forbidden methods',
                'PASS', 'primary', p_path, f'forbidden set absent: {sorted(forbidden)}')
        else:
            add('17.5', 'Primary benchmark has no forbidden methods',
                'FAIL', 'primary', p_path, f'found forbidden: {sorted(present_forbidden)}')
    else:
        add('17.1-17.5', 'Primary benchmark CSV present', 'FAIL', 'primary', p_path, 'missing')

    # --- Correctness ---
    c_path = os.path.join(out_dir, 'correctness_results.csv')
    if os.path.exists(c_path):
        cdf = pd.read_csv(c_path)
        ok_m, d_m = _expect_set(cdf['method'].unique(),
                                {'NBMF_Exact', 'NBMF_SA', 'NBMF_PathIntegral'},
                                'correctness method set')
        ok_s, d_s = _expect_set(cdf['seed'].unique(), set(cfg['general']['seeds']),
                                'correctness seed set')
        ok_r, d_r = _expect_set(cdf['rank'].unique(), set(cfg['correctness']['ranks']),
                                'correctness rank set')
        ok_sz, d_sz = _expect_set(cdf['matrix_size'].unique(),
                                  {f"{s[0]}x{s[1]}" for s in cfg['correctness']['sizes']},
                                  'correctness size set')
        overall = 'PASS' if all(x == 'PASS' for x in [ok_m, ok_s, ok_r, ok_sz]) else 'FAIL'
        add('17.6', 'Correctness has required methods, 5 seeds, sizes {10x10}, ranks {2,5}',
            overall, 'correctness', c_path,
            '; '.join([d_m, d_s, d_r, d_sz]))
    else:
        add('17.6', 'Correctness CSV present', 'FAIL', 'correctness', c_path, 'missing')

    # --- Convergence detailed ---
    cd_path = os.path.join(out_dir, 'convergence_detailed.csv')
    if os.path.exists(cd_path):
        ddf = pd.read_csv(cd_path)
        ok_m, d_m = _expect_set(
            ddf['method'].unique(),
            {'Repo_ALS', 'Repo_SGD', 'NBMF_SA', 'NBMF_PathIntegral', 'NBMF_Exact'},
            'convergence method set',
        )
        ok_r, d_r = _expect_set(ddf['rank'].unique(), set(cfg['convergence']['ranks']),
                                'convergence rank set')
        ok_s, d_s = _expect_set(ddf['seed'].unique(), set(cfg['general']['seeds']),
                                'convergence seed set')
        expected_size = f"{cfg['convergence']['synthetic_size'][0]}x{cfg['convergence']['synthetic_size'][1]}"
        ok_sz, d_sz = _expect_set(ddf['matrix_size'].unique(), {expected_size},
                                  'convergence size set')
        overall = 'PASS' if all(x == 'PASS' for x in [ok_m, ok_r, ok_s, ok_sz]) else 'FAIL'
        add('17.7',
            'Convergence detailed has required methods, 5 seeds, size 30x30, ranks {2,5,10}',
            overall, 'convergence', cd_path,
            '; '.join([d_m, d_r, d_s, d_sz]))
    else:
        add('17.7', 'Convergence detailed CSV present', 'FAIL', 'convergence', cd_path, 'missing')

    # --- Runtime breakdown ---
    b_path = os.path.join(out_dir, 'runtime_breakdown_results.csv')
    if os.path.exists(b_path):
        bdf = pd.read_csv(b_path)
        req = {'nnls_time', 'qubo_build_time', 'annealing_time',
               'postprocess_time', 'num_qubo_solves', 'avg_qubo_solve_time'}
        missing = req - set(bdf.columns)
        if missing:
            add('17.8', 'Runtime breakdown has all required columns',
                'FAIL', 'runtime', b_path, f'missing cols={sorted(missing)}')
        else:
            nb_methods = set(bdf['method'].unique())
            expected_nb = {'NBMF_SA', 'NBMF_PathIntegral'}
            if expected_nb.issubset(nb_methods):
                add('17.8', 'Runtime breakdown covers both NBMF primary methods with all columns',
                    'PASS', 'runtime', b_path,
                    f'methods={sorted(nb_methods)}, rows={len(bdf)}')
            else:
                add('17.8', 'Runtime breakdown covers both NBMF primary methods',
                    'FAIL', 'runtime', b_path,
                    f'methods={sorted(nb_methods)}, missing {sorted(expected_nb - nb_methods)}')
    else:
        add('17.8', 'Runtime breakdown CSV present', 'FAIL', 'runtime', b_path, 'missing')

    # --- Scalability ---
    s_path = os.path.join(out_dir, 'scalability_results.csv')
    if os.path.exists(s_path):
        sdf = pd.read_csv(s_path)
        syn = sdf[sdf['data_source'] == 'synthetic']
        ml = sdf[sdf['data_source'] == 'movielens']
        feas = sdf[sdf['data_source'] == 'feasibility']

        # Synthetic expected combinations
        syn_cfg = cfg['scalability']['synthetic_grid']
        exp_syn_sizes = {f"{s[0]}x{s[1]}" for s in syn_cfg['sizes']}
        ok_syn_sz, d_syn_sz = _expect_set(syn['matrix_size'].unique(), exp_syn_sizes,
                                          'synthetic sizes')
        ok_syn_smp, d_syn_smp = _expect_set(syn['sampler'].unique(), set(syn_cfg['samplers']),
                                            'synthetic samplers')
        # ranks: actual present should equal expected after filtering k<n
        actual_syn_ranks = set(syn['rank'].unique())
        expected_syn_ranks = set(syn_cfg['ranks'])
        if actual_syn_ranks.issubset(expected_syn_ranks):
            d_syn_r = f'synthetic ranks present={sorted(actual_syn_ranks)}'
            ok_syn_r = 'PASS'
        else:
            d_syn_r = f'unexpected extra ranks={sorted(actual_syn_ranks - expected_syn_ranks)}'
            ok_syn_r = 'FAIL'

        ok_syn_seed, d_syn_seed = _expect_set(
            syn['seed'].unique(), set(cfg['scalability']['scalability_seeds']),
            'synthetic seeds')

        overall_syn = 'PASS' if all(x == 'PASS' for x in [ok_syn_sz, ok_syn_smp, ok_syn_r, ok_syn_seed]) else 'FAIL'
        add('17.9a', 'Scalability synthetic grid matches protocol',
            overall_syn, 'scalability', s_path,
            '; '.join([d_syn_sz, d_syn_smp, d_syn_r, d_syn_seed]))

        ml_cfg = cfg['scalability']['movielens_grid']
        ok_ml_smp, d_ml_smp = _expect_set(ml['sampler'].unique(), set(ml_cfg['samplers']),
                                          'movielens samplers')
        ok_ml_r, d_ml_r = _expect_set(ml['rank'].unique(), set(ml_cfg['ranks']),
                                      'movielens ranks')
        ok_ml_seed, d_ml_seed = _expect_set(ml['seed'].unique(),
                                            set(cfg['scalability']['scalability_seeds']),
                                            'movielens seeds')
        overall_ml = 'PASS' if all(x == 'PASS' for x in [ok_ml_smp, ok_ml_r, ok_ml_seed]) else 'FAIL'
        add('17.9b', 'Scalability MovieLens grid matches protocol',
            overall_ml, 'scalability', s_path,
            '; '.join([d_ml_smp, d_ml_r, d_ml_seed]))

        # Feasibility
        feas_cfg = cfg['scalability']['feasibility']
        exp_feas_sizes = {f"{s[0]}x{s[1]}" for s in feas_cfg['exact_sizes']}
        ok_feas_sz, d_feas_sz = _expect_set(feas['matrix_size'].unique(), exp_feas_sizes,
                                            'feasibility sizes')
        actual_feas_ranks = set(feas['rank'].unique())
        expected_feas_ranks = set(feas_cfg['exact_ranks'])
        if actual_feas_ranks.issubset(expected_feas_ranks):
            ok_feas_r, d_feas_r = 'PASS', f'feasibility ranks present={sorted(actual_feas_ranks)}'
        else:
            ok_feas_r, d_feas_r = 'FAIL', (
                f'unexpected ranks={sorted(actual_feas_ranks - expected_feas_ranks)}'
            )
        ok_feas_seed, d_feas_seed = _expect_set(
            feas['seed'].unique(), {feas_cfg['exact_seed']}, 'feasibility seed')
        overall_feas = 'PASS' if all(x == 'PASS' for x in
                                     [ok_feas_sz, ok_feas_r, ok_feas_seed]) else 'FAIL'
        add('17.10', 'Feasibility subset matches protocol',
            overall_feas, 'feasibility', s_path,
            '; '.join([d_feas_sz, d_feas_r, d_feas_seed]))
    else:
        add('17.9-17.10', 'Scalability CSV present', 'FAIL', 'scalability', s_path, 'missing')

    # --- Summary aggregation ---
    sm_path = os.path.join(out_dir, 'benchmark_primary_summary.csv')
    if os.path.exists(sm_path):
        sdf = pd.read_csv(sm_path)
        req = {'method', 'metric', 'n', 'mean', 'std', 'ci_lower', 'ci_upper'}
        missing = req - set(sdf.columns)
        if missing:
            add('17.11', 'Primary summary has mean/std/count/CI', 'FAIL',
                'summary', sm_path, f'missing={sorted(missing)}')
        else:
            add('17.11', 'Primary summary has mean/std/count/CI', 'PASS',
                'summary', sm_path, f'rows={len(sdf)}')
    else:
        add('17.11', 'Primary summary CSV present', 'FAIL', 'summary', sm_path, 'missing')

    # --- Figures ---
    plots_dir = cfg['general']['plots_dir']
    required_pngs = [
        'convergence_primary.png', 'convergence_wallclock.png',
        'convergence_by_rank.png', 'correctness_small.png',
        'benchmark_error_bar.png', 'benchmark_runtime_bar.png',
        'runtime_breakdown_stacked.png', 'scalability_runtime_vs_size.png',
        'scalability_error_vs_rank.png', 'feasibility_exactsolver.png',
    ]
    missing = [p for p in required_pngs if not os.path.exists(os.path.join(plots_dir, p))]
    if not missing:
        add('17.12', 'All required PNGs exist', 'PASS', 'plots', plots_dir,
            f'all {len(required_pngs)} PNGs present')
    else:
        add('17.12', 'All required PNGs exist', 'FAIL', 'plots', plots_dir,
            f'missing={missing}')

    # --- Timeout consistency ---
    code_timeout = _timeout_from_code()
    cfg_timeout = cfg['scalability'].get('timeout_seconds')
    if code_timeout == cfg_timeout == 600:
        add('17.13', 'TIMEOUT constant matches config and protocol (600s)', 'PASS',
            'code', 'experiments/scalability.py', f'code={code_timeout}, cfg={cfg_timeout}')
    else:
        add('17.13', 'TIMEOUT constant matches config and protocol (600s)', 'FAIL',
            'code', 'experiments/scalability.py', f'code={code_timeout}, cfg={cfg_timeout}')

    # --- Write ---
    df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, 'compliance_matrix.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df)} rows)")

    print("\n=== COMPLIANCE MATRIX ===")
    for _, r in df.iterrows():
        print(f"  [{r['status']}] {r['requirement_id']:7s}  {r['requirement']}")
        print(f"           -> {r['evidence_detail']}")

    pass_count = int((df['status'] == 'PASS').sum())
    fail_count = int((df['status'] == 'FAIL').sum())
    print(f"\n  PASS: {pass_count}  FAIL: {fail_count}  TOTAL: {len(df)}")
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
