# Quantum-Assisted Collaborative Filtering (QACF)

This repository implements a MovieLens 32M recommendation research workflow with classical, quantum, and hybrid collaborative filtering pipelines.

## Notebook Flow (UBCF track)

- `QACF/01_data_prep_ubcf.ipynb` — data engineering and feature construction
- `QACF/02_classical_baselines_ubcf.ipynb` — classical clustering + UBCF baseline
- `QACF/03_quantum_prototype_ubcf.ipynb` — quantum kernel/clustering prototype
- `QACF/04_hybrid_integration_ubcf_improved.ipynb` — hybrid routing + sparsity stress
- `QACF/05_validation_ubcf.ipynb` — validation, confidence intervals, complexity analysis

## Environment Setup

```powershell
python -m pip install -r requirements.txt
```

## Reproducibility & Statistical Validation

Generate a reproducibility snapshot (system, package versions, artifact hashes):

```powershell
python QACF/tools/reproducibility_snapshot.py
```

Run statistical validation for hybrid vs classical stress outputs:

```powershell
python QACF/tools/statistical_validation.py --input-dir QACF/data/processed_32m_ubcf --iters 1000 --seed 42
```

Generated outputs are saved in `QACF/data/processed_32m_ubcf`.

## Mitacs Plan Compliance

See `QACF/MITACS_PLAN_GAP_CLOSURE.md` for a requirement-by-requirement status and gap-closure evidence.
