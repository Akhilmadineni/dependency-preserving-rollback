# A5P2 code: Tabu search for minimal dependency-preserving rollback sets

## How to run
```
pip install -r requirements.txt
python3 run_all.py                      # about 5-10 min on 2 CPU cores; reads ../data, writes ../results and ../figures
dot -Tpng -Gdpi=300 fig1_flowchart.dot -o ../figures/fig1_flowchart.png   # Fig. 1 (Graphviz)
python3 -c "from PIL import Image; Image.open('../figures/fig1_flowchart.png').convert('RGB').save('../results/fig1_flowchart.jpg')"
```
All seeds are fixed (LightGBM seeds 0-4, scalability instances seeds 7 and 11). LightGBM runs single-threaded.

## What run_all.py does
1. Loads `A5P2_rollback_components.csv` and `A5P2_rollback_edges.csv`. It builds graph symptom features from the edge table.
   `root_cause`, `must_rollback` and `coupled_rollback_required` are never used as features.
2. Pre-processing is fitted on training folds only: median imputation of `test_coverage`, `p99_latency_delta_ms` and `telemetry_gap_pct`, then 1/99 % winsorising of heavy-tailed metrics.
3. Fault model: LightGBM, out-of-fold predictions with GroupKFold(5) over releases, 5 seeds. It runs in two evidence regimes:
   G (graph-aware: local + edge-symptom features) and L (component-local features only).
4. Dependency model: an implication graph from contract-change and schema-coupling edges, plus noisy-OR must-rollback probabilities.
   The adoption rate rho is estimated on training folds only.
5. Solvers per release:
   - full rollback, naive component rollback and greedy closure (baselines);
   - the proposed Tabu search, with flip and closure moves, tabu tenure, aspiration and an adaptive penalty (strategic oscillation);
   - an exact enumeration reference, plus an exact MILP (scipy/HiGHS) for large synthetic graphs.
6. Evaluation, ablation, miss-cost sensitivity, scalability (n = 16..256), the Tabu tenure x penalty grid, Wilcoxon tests and figures.

## Outputs (../results)
| file | content |
|---|---|
| main_comparison.csv / _per_seed.csv | precision, recall, F1, exact set match, FP/FN/violations per release, health restored, operational cost, runtime (mean, std over 5 seeds) for both regimes |
| ablation.csv | Tabu variants (no propagation, single flips, soft constraints, F = 0), both regimes |
| sensitivity_wM.csv | trade-off as the miss cost w_M varies |
| scalability.csv / scalability_summary.csv | objective gap to the MILP optimum and runtime vs graph size |
| tabu_param_sensitivity.csv | gap for tenure x initial penalty (single-flip stress test) |
| plane_breakdown.csv | control- vs data-plane F1 |
| fault_model_auc.csv, feature_importance.csv | fault-model quality and LightGBM gain importance |
| metrics.json | all headline numbers, the exact-optimality check and the paired statistics |
| fig*.jpg | JPG copies of the figures (the PNGs are in ../figures): fig1 flowchart, fig2 importance, fig3 comparison, fig4 trade-off, fig5 scalability; fig6 (tenure x penalty heatmap) is summarised in the paper text only |
| run_log.txt | console log of the final run |
