# Classical Matrix Factorization Baselines

ALS and biased SGD implementations for explicit-feedback matrix factorization on MovieLens 100K. These are the classical baselines for a research project on quantum-enhanced recommendation — results and exported subproblems here feed directly into the QUBO stage.

---

## Repository Structure

```
ALS/            Alternating Least Squares model and training pipeline
SGD/            Biased SGD model and training pipeline
common/         Shared data loading, metrics, I/O, logging, and benchmark utilities
configs/        Default hyperparameters (JSON) for both models
data/           MovieLens 100K ratings (u.data, tab-separated)
outputs/        Generated artifacts — metrics, factors, history, QUBO exports
scripts/        Shell scripts to run each model end-to-end
```

---

## Models

**ALS** solves for user and item latent factors by alternating closed-form least-squares updates. Each pass over users and items is vectorized with NumPy, so it's fast but has no bias terms — it tends to memorize training data quickly at higher factor counts.

**Biased SGD** adds a global mean, per-user bias, and per-item bias on top of the latent dot product. Updates are applied per interaction (shuffled each epoch), which is slower but generalizes significantly better, especially on sparse datasets like ML-100K.

Both models use K=50 latent factors and λ=0.1 regularization by default.

---

## Evaluation and Outputs

**Metrics computed at each epoch:** training and validation RMSE and MAE.

**Final test-set ranking metrics:** HR@10, Precision@10, Recall@10, NDCG@10 — evaluated over unseen items only (training items excluded from the candidate pool). An item is considered relevant if its rating ≥ 4.0.

Each run writes to `outputs/{model}/`:

| File | Contents |
|---|---|
| `P.npy`, `Q.npy` | Learned latent factor matrices |
| `b_u.npy`, `b_i.npy` | User/item biases (SGD only) |
| `history.csv` | Per-epoch RMSE, MAE, objective, runtime |
| `metrics.json` | Final validation + ranking metrics |
| `{model}_benchmark_summary.json` | Combined runtime and metric summary |
| `{model}_qubo_ratings.npy` | Dense 50×50 predicted rating block (QUBO export) |
| `{model}_qubo_mask.npy` | Observation mask for that block |
| `{model}_qubo_meta.json` | Metadata mapping sub-indices to original IDs |

---

## How to Run

```bash
pip install -r requirements.txt

# From project root
python -m ALS.train --data data/u.data --out outputs/als --config configs/als_default.json
python -m SGD.train --data data/u.data --out outputs/sgd --config configs/sgd_default.json
```

Or use the shell scripts: `./scripts/run_als.sh` and `./scripts/run_sgd.sh`.

Edit `configs/als_default.json` or `configs/sgd_default.json` to change hyperparameters.

---

## Results

From actual runs on MovieLens 100K (943 users, 1682 items, 100K interactions, 93.7% sparse).

| Metric | ALS | SGD |
|---|---|---|
| Val RMSE | 1.6157 | **0.9831** |
| Val MAE | 1.2816 | **0.7818** |
| HR@10 | 0.0041 | **0.0351** |
| NDCG@10 | 0.0014 | **0.0141** |
| Total runtime | **2.2 s** (15 epochs) | 14.2 s (20 epochs) |
| Avg epoch time | **~0.14 s** | ~0.71 s |

**Key observations:**
- SGD is the better generalizer — 39% lower RMSE and ~8.5× better HR@10
- ALS overfits: train RMSE drops to 0.25 while val RMSE climbs to 1.62 over 15 epochs
- ALS is ~5× faster per epoch due to vectorized closed-form solves
- SGD converges smoothly and monotonically; ALS peaks early (epoch 2) then diverges on val
- The biases in SGD are doing real work on this dataset — pure dot-product ALS struggles on ML-100K's rating scale

---

## Notes

All runs are fully reproducible with `seed=42`. Configs are copied to `outputs/{model}/config_used.json` on each run. The `outputs/` directory is gitignored.

The QUBO bridge exports (`*_qubo_ratings.npy`, `*_qubo_mask.npy`, `*_qubo_meta.json`) extract the densest 50×50 user-item subblock — the most active 50 users and top 50 items — as a starting point for quantum annealing experiments in the next stage.
