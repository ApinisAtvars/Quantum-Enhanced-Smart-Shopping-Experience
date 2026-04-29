#!/usr/bin/env python
"""
Final publication run orchestrator.

Runs the full pipeline declared in protocol_final.md:

  1. Correctness validation (10x10 synthetic)
  2. Primary + supplemental benchmark + per-iteration traces (MovieLens)
  3. Convergence analysis (30x30 synthetic)
  4. Scalability sweep + ExactSolver feasibility
  5. Summarize primary benchmark -> benchmark_primary_summary.csv
  6. Plot generation -> plots_final/*.png
  7. Compliance matrix -> compliance_matrix.csv

All CSVs land in results_final/, figures in plots_final/, and a human-readable
run log is appended to final_logs/01_final_run_log.md.
"""
import os
import sys
import time
import datetime as dt
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

PYTHON_BIN = sys.executable or 'python3'

LOGS_DIR = 'final_logs'
RESULTS_DIR = 'results_final'
PLOTS_DIR = 'plots_final'

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

LOG_PATH = os.path.join(LOGS_DIR, '01_final_run_log.md')


def _log(msg):
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')


def _snapshot_outputs(before):
    def listing(d):
        if not os.path.isdir(d):
            return set()
        return {os.path.join(d, f) for f in os.listdir(d) if not f.startswith('.')}
    after_r = listing(RESULTS_DIR)
    after_p = listing(PLOTS_DIR)
    new_r = sorted(after_r - before['results'])
    new_p = sorted(after_p - before['plots'])
    before['results'] = after_r
    before['plots'] = after_p
    return new_r, new_p


def run_step(label, module, before):
    banner = '=' * 70
    print(f"\n{banner}\n  STEP: {label}\n{banner}")
    start = dt.datetime.utcnow().isoformat(timespec='seconds') + 'Z'
    t0 = time.time()
    result = subprocess.run(
        [PYTHON_BIN, '-m', module],
        cwd=ROOT,
        env={**os.environ, 'PYTHONPATH': ROOT},
    )
    elapsed = time.time() - t0
    end = dt.datetime.utcnow().isoformat(timespec='seconds') + 'Z'
    ok = result.returncode == 0
    status = 'OK' if ok else f'FAILED (exit {result.returncode})'
    print(f"\n  [{status}] {label} completed in {elapsed:.1f}s\n")
    new_r, new_p = _snapshot_outputs(before)
    _log(f"### {label}")
    _log(f"- command: `python -m {module}`")
    _log(f"- start: `{start}`")
    _log(f"- end: `{end}`")
    _log(f"- elapsed: `{elapsed:.2f}s`")
    _log(f"- status: `{status}`")
    _log(f"- new CSVs: {new_r or 'none'}")
    _log(f"- new PNGs: {new_p or 'none'}")
    _log('')
    return ok


def main():
    run_start = dt.datetime.utcnow().isoformat(timespec='seconds') + 'Z'
    with open(LOG_PATH, 'a') as f:
        f.write(f"\n# Final Run Log — started {run_start}\n\n")
        f.write(f"- protocol: `protocol_final.md`\n")
        f.write(f"- config: `config/experiment.yaml`\n")
        f.write(f"- results dir: `{RESULTS_DIR}/`\n")
        f.write(f"- plots dir: `{PLOTS_DIR}/`\n\n")

    before = {
        'results': {os.path.join(RESULTS_DIR, f) for f in os.listdir(RESULTS_DIR)}
                   if os.path.isdir(RESULTS_DIR) else set(),
        'plots':   {os.path.join(PLOTS_DIR, f) for f in os.listdir(PLOTS_DIR)}
                   if os.path.isdir(PLOTS_DIR) else set(),
    }

    t_total = time.time()
    steps = [
        ("Correctness validation", "experiments.correctness"),
        ("Main benchmark (MovieLens)", "experiments.benchmark"),
        ("Convergence analysis (synthetic)", "experiments.convergence"),
        ("Scalability + feasibility sweep", "experiments.scalability"),
        ("Summarize primary benchmark", "experiments.summarize"),
        ("Plot generation", "experiments.plot_results"),
        ("Compliance audit", "experiments.compliance"),
    ]

    results = {}
    for label, module in steps:
        ok = run_step(label, module, before)
        results[label] = ok

    total = time.time() - t_total
    run_end = dt.datetime.utcnow().isoformat(timespec='seconds') + 'Z'

    banner = '=' * 70
    print(f"\n{banner}\n  PIPELINE COMPLETE  ({total:.1f}s total)\n{banner}")
    for label, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    _log(f"## Summary")
    _log(f"- run end: `{run_end}`")
    _log(f"- total elapsed: `{total:.2f}s`")
    for label, ok in results.items():
        _log(f"- {label}: **{'PASS' if ok else 'FAIL'}**")
    _log('')

    print(f"\nOutputs:")
    for d in [RESULTS_DIR, PLOTS_DIR]:
        files = sorted(os.listdir(d))
        if files:
            print(f"  {d}/")
            for f in files:
                print(f"    {f}")
    print()


if __name__ == "__main__":
    main()
