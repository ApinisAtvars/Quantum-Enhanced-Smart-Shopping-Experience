# Mitacs GRA 2026 — Gap Closure Checklist (Alessia)

Date: 2026-03-10  
Scope: `QACF` notebooks 01–05 and generated artifacts

## Compliance Matrix

| Plan Requirement | Status | Evidence | Gap Closed By |
|---|---|---|---|
| Shared MovieLens pipeline | Pass | notebooks 01–05 + `data/processed_32m_ubcf` outputs | Existing implementation |
| Core metrics (RMSE, MAE, HR@10, NDCG@10, Precision@K) | Pass | `ubcf_metrics.csv`, stress outputs | Existing implementation |
| Classical baseline + classical CF | Pass | notebook 02 (`MiniBatchKMeans`, UBCF adjusted cosine) | Existing implementation |
| Quantum clustering prototype (Qiskit/PennyLane) | Pass | notebook 03 (kernel + clustering variants) | Existing implementation |
| Hybrid integration with sparsity routing | Pass | notebook 04 (`classical_active` / `quantum_sparse` / `content_cold`) | Existing implementation |
| Statistical validation (5–10 trial rigor expectation) | Partial | bootstrap CI in notebook 05 | `QACF/tools/statistical_validation.py` |
| Reproducibility protocol (seeds + hardware/env logs) | Partial | fixed seeds in notebooks | `QACF/tools/reproducibility_snapshot.py` |
| Runtime/complexity comparison | Pass | `runtime_complexity_ubcf.csv`, notebook 05 | Existing implementation |
| NISQ limitations disclosure | Pass | notebook 03/05 and reports | Existing implementation |
| PyTorch in core stack | Partial | not central in current pipeline | Add only if supervisor requires direct PyTorch experiments |

## What Was Added to Close Gaps

1. **Formal reproducibility snapshot utility**
   - File: `QACF/tools/reproducibility_snapshot.py`
   - Produces: `data/processed_32m_ubcf/reproducibility_snapshot.json`
   - Captures: Python/system metadata, package versions, artifact hashes

2. **Formal statistical validation utility**
   - File: `QACF/tools/statistical_validation.py`
   - Produces:
     - `data/processed_32m_ubcf/statistical_validation_summary.csv`
     - `data/processed_32m_ubcf/statistical_validation_summary.md`
     - `data/processed_32m_ubcf/statistical_validation_summary.json`
   - Methods: paired bootstrap CI + paired permutation tests + effect size

## Run Commands

From repository root:

```powershell
python QACF/tools/reproducibility_snapshot.py
python QACF/tools/statistical_validation.py --input-dir QACF/data/processed_32m_ubcf --iters 1000 --seed 42
```

## Final Assessment

Your implementation is **correct for the defined QACF experiment design** and now has explicit tooling to support reproducibility/statistical validation claims. 

To claim full “all available dataset information” usage, you would still need to integrate additional side-info channels (e.g., deeper use of `links.csv` metadata enrichment and richer temporal modeling), which is outside the current notebook scope.
