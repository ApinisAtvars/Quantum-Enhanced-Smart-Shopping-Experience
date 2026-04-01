Source: https://github.com/yihong-chen/neural-collaborative-filtering/tree/master

# Results
---
- All metrics are for ml32m subset of 1000 users.
- All metrics are from final epoch

| Metric | 28_03_b256_classical_neumf_baseline |  26_03_subset_no_qvc  | 
|---|---| --- |
| HR@10 | 0.791 | 0.774 |
| NDCG@10 | 0.525 | 0.5149 |
| BCE Loss | 559 | 10 608 |
| Train Time (minutes) | 2 | 25 |
| #Epochs | 10 | 10 |
| Quantum Circuit |  | |
| Batch Size | 256 | 16 |
| Note | No extra dense layers, just the model as it was described exactly in the "Neural Collaborative Filtering" paper. | This is a baseline to compare against the 23_03_gaussian_dqn_v2 model. Includes the exact pre and post net classical layers that Gaussian dressed circuit has. |
|  |  |  |
|  |  | |
|  |  | |
|  |  | |
|  |  | |
|  |  | |
|  |  | |


