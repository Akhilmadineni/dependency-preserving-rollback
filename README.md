# Dependency-Preserving Partial Rollback for Coupled Control-Plane and Data-Plane Deployments

This repository contains the experiment code, synthetic data, result tables, and figures for selecting compatible partial rollback sets in a distributed energy resource management system. A LightGBM fault model supplies component probabilities, noisy-OR propagation adds dependency risk, and Tabu search minimizes expected rollback cost subject to contract and schema implications.

## Repository contents

- `code/`: experiment scripts, code notes, and figure sources.
- `data/`: synthetic input CSVs and their data dictionaries.
- `results/`: supplied CSV results, JPG figures, and `metrics.json`.
- `requirements.txt`: Python dependencies for installation from the repository root.

## Data

The repository includes `data/A5P2_rollback_components.csv` (15,858 rows and 24 columns) and `data/A5P2_rollback_edges.csv` (17,443 rows and 11 columns), describing 1,500 synthetic releases on a 16-component system. The data dictionary is [`data/A5P2_README.md`](data/A5P2_README.md). These are simulated records, not utility customer logs.

## Environment and installation

The reported experiments used Python 3.13. `requirements.txt` preserves the supplied pinned dependency versions. Install Graphviz separately if you want to regenerate the Figure 1 flowchart, and ensure its `dot` executable is on `PATH`. The numeric pipeline does not render Figure 1 automatically.

Run these commands from the repository root, after selecting the appropriate Python interpreter:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

## Reproduce the experiments

From the repository root:

```sh
OMP_NUM_THREADS=1 python code/run_all.py
```

The script resolves input and output paths relative to its own `__file__`, so `python code/run_all.py` works from the repository root. The source's alternative `cd code` followed by `python run_all.py` also works. The experiments use fixed seeds and CPU execution. Rerunning the pipeline rewrites the result files in `results/` and creates PNGs in `figures/`.

To regenerate Figure 1 from the supplied Graphviz source:

```sh
mkdir -p figures
dot -Tpng -Gdpi=300 code/fig1_flowchart.dot -o figures/fig1_flowchart.png
```

The existing JPG is included as `results/fig1_flowchart.jpg`. The full pipeline generates the remaining result figures.

## Result files

- `results/main_comparison.csv` and `main_comparison_per_seed.csv`: policy comparison for graph-aware and local evidence.
- `results/ablation.csv` and `sensitivity_wM.csv`: ablation and miss-cost trade-offs.
- `results/scalability.csv`, `scalability_summary.csv`, and `tabu_param_sensitivity.csv`: solver scaling and search sensitivity.
- `results/fault_model_auc.csv`, `plane_breakdown.csv`, and `feature_importance.csv`: model and plane-specific evidence.
- `results/metrics.json`: structured summary.
- `results/fig*.jpg` and generated `figures/*.png`: figures.

See [`code/README.md`](code/README.md) for method and output details. Supplied runtime estimates depend on hardware and installed dependency versions.

## Scope and limitations

All data and scale-study graphs are synthetic. Compatibility rules match the optimization constraints by construction, and the cost weights are assumptions. Greedy closure performs comparably to Tabu on the small modeled system. Exact enumeration and MILP solve the tested linear formulation faster than the Python Tabu implementation. The dataset documentation names an upstream `gen_author5.py` generator, but it was not supplied in this archive; the included CSVs are sufficient to run the experiments.

This release package contains code, data, and experiment outputs. Manuscripts, presentation decks, operating-system metadata, virtual environments, and transient run logs are excluded.

## Author and license

Author: Akhil Venkata Sai Madineni. This package is distributed under the [MIT license](LICENSE).

## Citation and archive

Source repository: [https://github.com/Akhilmadineni/dependency-preserving-rollback](https://github.com/Akhilmadineni/dependency-preserving-rollback).

Use [`CITATION.cff`](CITATION.cff) for the software citation. The matching [`.zenodo.json`](.zenodo.json) supplies the archive metadata for release `v1.0.0`. Version `v1.0.0` is archived at [10.5281/zenodo.24411016](https://doi.org/10.5281/zenodo.24411016). Use this version DOI to cite the fixed experiment archive.

## Data and code availability

The code, synthetic data, and results are available at https://github.com/Akhilmadineni/dependency-preserving-rollback, with version v1.0.0 archived at https://doi.org/10.5281/zenodo.24411016.
