# RepMOBO

**Representation-Guided Multi-Objective Bayesian Optimization for Pareto-Optimal MOF Discovery**

Research code and reference data for three-objective MOF discovery over **7,532 materials**,
with CEES, DMMP and EtCl adsorption targets. The optimization study covers **105 tasks**:
five representations, two model-guided strategies, ARD on/off and five seeds, plus five Random runs.

This repository is being prepared for author and peer review. It does not assert publication,
an archival DOI or a software reuse license. Existing reference results and new runs are
different artifacts; see [reproducibility notes](docs/REPRODUCIBILITY.md).

## Start here

The core optimization inputs are included in Git: no separate data download is needed
for the input-validation demo. Use **Python 3.10** in a dedicated environment.

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install numpy==1.26.4 pandas==2.3.3
python -B 4-repmobo_optimization/scripts/04_validate_review_package.py
```

Expected output: `INPUT VALIDATION PASSED`. This checks data and configurations; it does
not train a model or reproduce the optimization campaign. See [actual local checks](docs/VALIDATION.md). For GP execution, install the
full module environment and run a single task:

```sh
python -m pip install -r 4-repmobo_optimization/requirements.txt
cd 4-repmobo_optimization
python -B scripts/03_run_al_task.py --config configs/ReMOBO_Hybrid_PCA85_58D_ARDoff_batch10_seed42.json --check-only
python -B scripts/03_run_al_task.py --config configs/ReMOBO_Hybrid_PCA85_58D_ARDoff_batch10_seed42.json
```

`--device cpu` forces CPU execution; the default prefers an available compatible accelerator.
Cluster DCU execution requires the facility's compatible PyTorch/runtime build. Full GP runs
can be expensive and are not part of the demo.

## Repository layout

| Module | Purpose | Entry point |
| --- | --- | --- |
| [1-data_preprocessing](1-data_preprocessing/README.md) | CIF checks and manual descriptors | Four preprocessing scripts; upstream prerequisites documented |
| [2-representation_benchmark](2-representation_benchmark/README.md) | LightGBM and Ridge across five representations | Saved metrics, predictions, split file and training scripts |
| [3-oracle_pareto_analysis](3-oracle_pareto_analysis/README.md) | Full-library Pareto analysis | Oracle membership/statistics and analysis script |
| [4-repmobo_optimization](4-repmobo_optimization/README.md) | GP, qEHVI, Greedy and Random | Three scripts, 105 configurations and aligned inputs |
| [docs](docs/DATA.md) | Data access, provenance and validation scope | Release asset instructions and SHA256 manifest |

## Running all optimization tasks

From `4-repmobo_optimization/`:

```sh
python -B scripts/06_run_all.py             # list only
python -B scripts/06_run_all.py --execute   # runs all 105 tasks
```

Successful tasks write `selection_events.csv`, `round_metrics.csv` and `final_summary.json`
under `results/tasks/<task_id>/`. Existing task outputs are not overwritten.
The supplied `expected_outputs/` and `source_data/` contain previously generated reference
tables and Fig. 4/Fig. 5/SI numerical source data; these are not overwritten by the commands above.

## Data downloads

The repository contains core optimization data, small benchmark inputs, reference summaries
and figure source data. The full CIF collection, large representation matrices and detailed
query history are packaged separately for a repository Release. See [data access](docs/DATA.md)
for filenames, extraction paths and checksum verification. Their remote availability is not
assumed until the Release has been uploaded.

## Citation and reuse

Refer to the accompanying manuscript, *RepMOBO: Representation-Guided Multi-Objective Bayesian
Optimization for Pareto-Optimal MOF Discovery*. Author metadata, publication DOI and a public
reuse license will be added only after author confirmation. Third-party software and source
structures/representations retain their respective terms; repository inclusion does not grant
new rights over third-party materials. No license has been selected on the authors' behalf.
