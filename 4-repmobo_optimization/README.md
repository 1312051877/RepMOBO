# RepMOBO optimization

## Installation

Use Python 3.10 and install the dependencies from the module root:

```sh
python -m pip install -r requirements.txt
```

GP tasks require PyTorch, GPyTorch and BoTorch. A compatible PyTorch build is
needed for GPU/DCU execution; otherwise execution uses CPU.

## Inputs

`configs/` contains 105 JSON task configurations. The eight required data files are:

- `data/full_library_master.csv`: 7532 MOFs, labels and saved evaluation flags.
- `data/initial_designs_checked.csv`: 100 initial MOFs for each of five seeds.
- `data/oracle_reference.json`: reference identifiers, reference point and oracle HV.
- Five CSVs in `data/latent_coordinates/`: Manual and PCA80/85/90/95 feature
  matrices with 14/41/58/86/145 features and aligned MOF IDs.

The three targets are `CEES_true`, `DMMP_true` and `EtCl_true`.

## Scripts and commands

Run commands from this module's root directory.

`04_validate_review_package.py` checks configurations, parameters, input files,
MOF alignment, feature dimensions, labels and initial designs. It does not fit
models or write files:

```sh
python -B scripts/04_validate_review_package.py
```

Successful validation prints `INPUT VALIDATION PASSED`.

`03_run_al_task.py` runs one task:

```sh
python -B scripts/03_run_al_task.py --config configs/ReMOBO_Hybrid_PCA85_58D_ARDoff_batch10_seed42.json
```

Add `--check-only` to validate the single-task input interface without running.
Add `--device cpu` to explicitly use CPU. An optional `--max-budget-override 110`
limits a test to 110 cumulative MOFs, including the initial 100, without editing
its configuration.

`06_run_all.py` reads the 105 JSON files directly. The first command only lists
them; only the second starts optimization:

```sh
python -B scripts/06_run_all.py
python -B scripts/06_run_all.py --execute
```

Default device selection prefers an available GPU/DCU. Accelerator out-of-memory
errors trigger a fresh CPU process with the same seed and initial design. Failed
attempt records are discarded; attempts are never appended together. CPU thread
limits are detected automatically from process affinity, scheduler allocations
and available CPU quotas. Full optimization is computationally expensive.

## Outputs

Each successful task writes only these files to `results/tasks/<task_id>/`:

- `selection_events.csv`: initial MOFs and subsequent selections in query order.
- `round_metrics.csv`: recorded-budget HV, Pareto capture and target recovery metrics.
- `final_summary.json`: final metrics, actual device, parameters and fallback status.

Existing task directories are not overwritten. No snapshots or checkpoint files
are produced. GP execution requires the complete dependencies; validation alone
is not evidence of a successful optimization run.
