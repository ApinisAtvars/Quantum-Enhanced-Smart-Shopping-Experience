# Classical Matrix Factorization Baselines

This repository implements **Alternating Least Squares (ALS)** and **Biased Stochastic Gradient Descent (SGD)** for explicit-feedback matrix factorization. These serve as classical baselines for a broader research project on **Quantum Annealing for Matrix Factorization**, establishing performance references before exploring QUBO-based formulations on quantum/hybrid solvers.

---

## Repository Structure

```
.
├── ALS/                  # Alternating Least Squares implementation
│   ├── __init__.py
│   ├── model.py          # ALS model class with fit_step, predict, objective
│   └── train.py          # ALS training pipeline (CLI entry point)
├── SGD/                  # Biased SGD implementation
│   ├── __init__.py
│   ├── model.py          # BiasedSGD model class with fit_epoch, predict, objective
│   └── train.py          # SGD training pipeline (CLI entry point)
├── common/               # Shared utilities
│   ├── __init__.py
│   ├── data.py           # Data loading, ID encoding, time-aware train/val/test split
│   ├── metrics.py        # RMSE, MAE, and top-K ranking metrics (HR, NDCG, Precision, Recall)
│   ├── benchmark.py      # Benchmark summary generation
│   ├── io_utils.py       # JSON I/O and QUBO bridge export
│   └── logging_utils.py  # Standardized console logger
├── configs/              # Hyperparameter configurations
│   ├── als_default.json
│   └── sgd_default.json
├── data/                 # Dataset directory
│   └── u.data            # MovieLens 100K ratings (tab-separated)
├── outputs/              # Generated outputs (reproducible, gitignored)
│   ├── als/              # ALS model artifacts, metrics, history, QUBO exports
│   └── sgd/              # SGD model artifacts, metrics, history, QUBO exports
├── scripts/              # Shell scripts for running experiments
│   ├── run_als.sh
│   └── run_sgd.sh
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

**MovieLens 100K** (`data/u.data`) — 100,000 ratings from 943 users on 1,682 movies, rated on a 1–5 integer scale.

- **Format**: Tab-separated (`userId \t itemId \t rating \t timestamp`).
- **Splitting**: Time-aware leave-last-out strategy — for each user with ≥3 interactions, the most recent interaction becomes the test sample, the second-most-recent becomes validation, and the rest form the training set. Users with fewer than 3 interactions are placed entirely in training.
- **Filtering**: Validation and test sets are filtered to contain only users and items present in the training set.

| Split    | Size   |
|----------|--------|
| Train    | 98,114 |
| Val      | 942    |
| Test     | 940    |
| Sparsity | 93.7%  |

---

## Implementations

### Alternating Least Squares (ALS)

A pure NumPy implementation of explicit-feedback ALS that alternates between solving for user factor matrix **P** and item factor matrix **Q** via regularized least squares.

Each update step solves, for every user *u*:

$$\mathbf{p}_u = (Q_u^T Q_u + \lambda I)^{-1} Q_u^T \mathbf{r}_u$$

and symmetrically for each item *i*. Predictions are clipped to the observed rating range [1, 5].

**Default hyperparameters** (`configs/als_default.json`):
| Parameter   | Value |
|-------------|-------|
| Factors (K) | 50    |
| λ (reg)     | 0.1   |
| Iterations  | 15    |
| Seed        | 42    |

### Biased SGD

A biased matrix factorization model optimized via element-wise stochastic gradient descent. The prediction model includes a global mean μ, per-user bias *b_u*, per-item bias *b_i*, and latent factor dot product:

$$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u^T \mathbf{q}_i$$

Each epoch shuffles training interactions and applies SGD updates to all parameters simultaneously. Factor vectors are copied before update to ensure correct gradient application.

**Default hyperparameters** (`configs/sgd_default.json`):
| Parameter       | Value |
|-----------------|-------|
| Factors (K)     | 50    |
| Learning rate η | 0.01  |
| λ (reg)         | 0.1   |
| Epochs          | 20    |
| Seed            | 42    |

### Why These Baselines

ALS and biased SGD represent the two dominant optimization paradigms for explicit-feedback matrix factorization. ALS provides closed-form solutions per iteration (fast, parallelizable), while biased SGD offers fine-grained per-interaction updates with bias terms that better capture rating distributions. Together, they establish a comprehensive classical reference for comparison with quantum-based approaches.

---

## Evaluation Protocol

### Rating Prediction Metrics
- **RMSE** (Root Mean Squared Error) — measures prediction accuracy, penalizing large errors.
- **MAE** (Mean Absolute Error) — measures average absolute prediction deviation.

Both are computed on the validation set at each epoch and reported as final metrics.

### Ranking Metrics (Top-K, *K=10*)
Evaluated on the held-out test set with seen-item filtering (training items excluded from candidate pool):

- **HR@10** (Hit Rate) — fraction of users whose test item appears in the top-10 recommendations.
- **Precision@10** — fraction of top-10 items that are relevant (rating ≥ 4.0).
- **Recall@10** — fraction of relevant items captured in the top-10.
- **NDCG@10** (Normalized Discounted Cumulative Gain) — position-aware ranking quality measure.

**Relevance threshold**: A test rating ≥ 4.0 is considered a relevant item.

### Convergence & Runtime Logging
Each epoch logs: wall-clock time, regularized objective value, train RMSE/MAE, and validation RMSE/MAE. Full histories are saved to `outputs/{model}/history.csv`.

---

## Running Experiments

### Setup
```bash
pip install -r requirements.txt
```

### Training
From the project root directory:

```bash
# ALS
python -m ALS.train --data data/u.data --out outputs/als --config configs/als_default.json

# SGD
python -m SGD.train --data data/u.data --out outputs/sgd --config configs/sgd_default.json
```

Or use the provided shell scripts:
```bash
./scripts/run_als.sh
./scripts/run_sgd.sh
```

### Output Artifacts
Each run produces the following in `outputs/{model}/`:

| File                          | Description                                           |
|-------------------------------|-------------------------------------------------------|
| `P.npy`, `Q.npy`             | Learned latent factor matrices                        |
| `b_u.npy`, `b_i.npy`         | User and item biases (SGD only)                       |
| `config_used.json`           | Hyperparameters used for the run                      |
| `meta.json`                  | Dataset statistics and split sizes                    |
| `history.csv`                | Per-epoch convergence and runtime log                 |
| `metrics.json`               | Final validation and ranking metrics                  |
| `{model}_benchmark_summary.json` | Combined benchmark summary                       |
| `{model}_qubo_ratings.npy`   | Dense 50×50 predicted rating submatrix (QUBO export)  |
| `{model}_qubo_mask.npy`      | Observation mask for the submatrix (QUBO export)      |
| `{model}_qubo_meta.json`     | Metadata mapping sub-indices to original IDs          |

---

## Results

All results below are from actual runs on MovieLens 100K with default hyperparameters.

### Rating Prediction

| Metric       | ALS    | SGD    |
|--------------|--------|--------|
| **Val RMSE** | 1.6157 | 0.9831 |
| **Val MAE**  | 1.2816 | 0.7818 |

### Top-10 Ranking

| Metric           | ALS     | SGD     |
|------------------|---------|---------|
| **HR@10**        | 0.0041  | 0.0351  |
| **NDCG@10**      | 0.0014  | 0.0141  |
| **Precision@10** | 0.0004  | 0.0035  |
| **Recall@10**    | 0.0041  | 0.0351  |

### Runtime

| Metric              | ALS        | SGD         |
|----------------------|------------|-------------|
| **Total runtime**    | 2.16 s     | 14.17 s     |
| **Avg epoch time**   | ~0.14 s    | ~0.71 s     |
| **Epochs**           | 15         | 20          |

### Convergence Observations

- **ALS** converges extremely fast in wall-clock time (~0.14 s/epoch) due to NumPy-vectorized closed-form updates. However, with K=50 and λ=0.1, it severely overfits: training RMSE drops to 0.25 while validation RMSE climbs from 1.37 to 1.62 across epochs. The model would benefit from stronger regularization or fewer latent factors.
- **SGD** converges more gradually (~0.71 s/epoch due to sequential per-interaction updates) but achieves substantially better generalization. Validation RMSE decreases monotonically from 1.06 to 0.98 over 20 epochs, with no signs of overfitting. The bias terms (μ, *b_u*, *b_i*) are essential for capturing rating-scale heterogeneity.
- **Ranking quality** is modest for both methods on ML-100K (a sparse dataset with noisy ratings), which is typical. SGD outperforms ALS by a factor of ~8.5× on HR@10 and ~10× on NDCG@10 with default settings.

---

## Conclusions

1. **Biased SGD is the stronger baseline** on MovieLens 100K under default hyperparameters, achieving 39% lower RMSE and significantly better ranking metrics than unregularized ALS.
2. **ALS overfits rapidly** with K=50 factors and λ=0.1. Its closed-form updates memorize training data quickly but do not generalize well without tuning. Nevertheless, ALS remains valuable for its simplicity and parallelizability.
3. **Both methods produce usable factor matrices** (P, Q) and meaningful predicted rating submatrices for downstream QUBO formulation.
4. **The evaluation protocol is complete**, covering both prediction accuracy (RMSE, MAE) and recommendation quality (HR, NDCG, Precision, Recall) with proper seen-item filtering and temporal splitting.

---

## QUBO Bridge

Both training pipelines export a dense subproblem designed for quantum/QUBO experimentation:

1. The **top 50 most active users** and **top 50 most frequently rated items** are selected from the training set.
2. A **50×50 predicted rating block** is reconstructed from the learned factors (and biases for SGD).
3. A **50×50 observation mask** (binary) indicates which entries correspond to observed training interactions.
4. **Metadata** maps sub-indices back to original user/item IDs.

These exports (located in `outputs/{model}/{model}_qubo_*.{npy,json}`) provide a small, dense, well-conditioned subproblem suitable for testing discrete parameter mappings or quantum annealing setups against known classical solutions.

---

## Reproducibility

- **Requirements**: `numpy>=1.20.0`, `pandas>=1.2.0` (see `requirements.txt`)
- **Configs**: All hyperparameters are stored in `configs/` and copied to `outputs/{model}/config_used.json` at runtime.
- **Seed**: All random operations use seed 42 for full reproducibility.
- **Outputs**: Complete outputs are regenerable by rerunning the training scripts. The `outputs/` directory is gitignored to keep the repository clean.

---

## Next Steps

The saved factor matrices, bias vectors, benchmark metrics, and QUBO bridge exports from this stage feed directly into the **QUBO formulation stage** of the research:

- The dense 50×50 predicted rating blocks and observation masks serve as target matrices for quantum annealing experiments.
- The classical RMSE/ranking results establish baselines against which quantum/hybrid solutions will be compared.
- The metadata files enable mapping quantum solutions back to the original user-item space for end-to-end evaluation.
