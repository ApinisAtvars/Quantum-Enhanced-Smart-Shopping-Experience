
# Final Run Log — started 2026-04-22T23:32:04Z

- protocol: `protocol_final.md`
- config: `config/experiment.yaml`
- results dir: `results_final/`
- plots dir: `plots_final/`

### Correctness validation
- command: `python -m experiments.correctness`
- start: `2026-04-22T23:32:04Z`
- end: `2026-04-22T23:32:09Z`
- elapsed: `4.48s`
- status: `OK`
- new CSVs: ['results_final/correctness_results.csv']
- new PNGs: none

### Main benchmark (MovieLens)
- command: `python -m experiments.benchmark`
- start: `2026-04-22T23:32:09Z`
- end: `2026-04-22T23:32:38Z`
- elapsed: `29.02s`
- status: `OK`
- new CSVs: ['results_final/benchmark_primary_results.csv', 'results_final/benchmark_supplemental_results.csv', 'results_final/convergence_results.csv', 'results_final/runtime_breakdown_results.csv']
- new PNGs: none

### Convergence analysis (synthetic)
- command: `python -m experiments.convergence`
- start: `2026-04-22T23:32:38Z`
- end: `2026-04-22T23:33:12Z`
- elapsed: `33.73s`
- status: `OK`
- new CSVs: ['results_final/convergence_detailed.csv']
- new PNGs: none

### Scalability + feasibility sweep
- command: `python -m experiments.scalability`
- start: `2026-04-22T23:33:12Z`
- end: `2026-04-22T23:35:31Z`
- elapsed: `139.69s`
- status: `OK`
- new CSVs: ['results_final/scalability_results.csv']
- new PNGs: none

### Summarize primary benchmark
- command: `python -m experiments.summarize`
- start: `2026-04-22T23:35:31Z`
- end: `2026-04-22T23:35:33Z`
- elapsed: `1.60s`
- status: `OK`
- new CSVs: ['results_final/benchmark_primary_summary.csv']
- new PNGs: none

### Plot generation
- command: `python -m experiments.plot_results`
- start: `2026-04-22T23:35:33Z`
- end: `2026-04-22T23:35:34Z`
- elapsed: `1.64s`
- status: `OK`
- new CSVs: none
- new PNGs: ['plots_final/benchmark_error_bar.png', 'plots_final/benchmark_runtime_bar.png', 'plots_final/convergence_by_rank.png', 'plots_final/convergence_primary.png', 'plots_final/convergence_wallclock.png', 'plots_final/correctness_small.png', 'plots_final/feasibility_exactsolver.png', 'plots_final/runtime_breakdown_stacked.png', 'plots_final/scalability_error_vs_rank.png', 'plots_final/scalability_runtime_vs_size.png']

### Compliance audit
- command: `python -m experiments.compliance`
- start: `2026-04-22T23:35:34Z`
- end: `2026-04-22T23:35:35Z`
- elapsed: `0.28s`
- status: `OK`
- new CSVs: ['results_final/compliance_matrix.csv']
- new PNGs: none

## Summary
- run end: `2026-04-22T23:35:35Z`
- total elapsed: `210.46s`
- Correctness validation: **PASS**
- Main benchmark (MovieLens): **PASS**
- Convergence analysis (synthetic): **PASS**
- Scalability + feasibility sweep: **PASS**
- Summarize primary benchmark: **PASS**
- Plot generation: **PASS**
- Compliance audit: **PASS**

