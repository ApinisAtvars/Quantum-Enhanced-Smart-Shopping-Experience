#!/usr/bin/env python
"""
Master script: runs the full experiment pipeline end-to-end.

1. Correctness validation (small synthetic)
2. Main benchmark (MovieLens subblock)
3. Convergence analysis (synthetic)
4. Scalability sweep
5. Plot generation

All results go to results/*.csv, all plots to plots/*.png.
"""
import os
import sys
import time
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

VENV_PYTHON = os.path.join(ROOT, '.venv', 'bin', 'python')
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = sys.executable


def run_step(label, module):
    print(f"\n{'='*60}")
    print(f"  STEP: {label}")
    print(f"{'='*60}\n")
    t0 = time.time()
    result = subprocess.run(
        [VENV_PYTHON, '-m', module],
        cwd=ROOT,
        env={**os.environ, 'PYTHONPATH': ROOT},
    )
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n  [{status}] {label} completed in {elapsed:.1f}s\n")
    return result.returncode == 0


def main():
    t_total = time.time()

    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('research_logs', exist_ok=True)

    steps = [
        ("Correctness validation", "experiments.run_correctness"),
        ("Main benchmark (MovieLens)", "experiments.run_benchmarks"),
        ("Convergence analysis (synthetic)", "experiments.run_convergence"),
        ("Scalability sweep", "experiments.run_scalability"),
        ("Plot generation", "experiments.generate_plots"),
    ]

    results = {}
    for label, module in steps:
        ok = run_step(label, module)
        results[label] = ok

    total_time = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE  ({total_time:.1f}s total)")
    print(f"{'='*60}")
    for label, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}")

    print(f"\nOutputs:")
    for d in ['results', 'plots']:
        files = sorted(os.listdir(d))
        if files:
            print(f"  {d}/")
            for f in files:
                print(f"    {f}")

    print()


if __name__ == "__main__":
    main()
