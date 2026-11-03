# Reproducibility and provenance

## Reference results versus rerun configurations

The supplied results are historical data. They are not outputs from executing the repository
in this packaging session. The 105 current JSON configurations define the requested rerun:

| Parameter | Original 105 task configurations | Current 105 rerun configurations |
| --- | ---: | ---: |
| candidate_screen_size | 512 | 2048 |
| qehvi_mc_samples | 16 | 64 |
| acq_chunk_size | 8 | 256 |
| max_fit_iter | 80 | 75 |

Both columns were checked against the respective 105 JSON files. Original task configurations
are in the author's AL project, under `AL_results/tasks/<task_id>/config.json`; they are not
the current configurations distributed in this repository. Do not interpret a new run with
current settings as an exact replay of the saved query sequence. Reconciliation of the
manuscript/SI descriptions with the historical settings remains an author responsibility.

The 105 configurations consist of 50 qEHVI tasks (Manual_AL/ReMOBO), 50 Greedy tasks and five
Random tasks. The current loader and data validator provide the executable consistency checks.

## Packaging changes

The original submission folder was left unchanged. Input data, result tables, source-data
workbooks, optimization scripts and all 105 current JSON configurations were copied byte-for-byte.
The GitHub working copy changes filesystem locations in the preprocessing and benchmark
scripts, and adds a read-time `EtCL` to `EtCl` header/split-sheet alias to the benchmark and oracle readers.
No model algorithm, threshold, random seed, acquisition setting or numerical dataset was changed.
The original combined README was replaced by a repository landing page and module instructions.

`source_manifest.csv` lists every original file, its original SHA256 and its distribution
location. Modified Python scripts have a separate packaged SHA256. Release archives preserve
the original repository-relative paths so extraction does not require moving individual files.

## Limits

- The preprocessing descriptor workbook used before edge-case filtering is not supplied.
- Zeo++ requires a separate compatible installation; its driver retains Windows/Cygwin assumptions.
- Benchmark training, GP optimization and the full 105-task campaign were not rerun for packaging.
- GitHub Actions is configured for syntax/input checks only; a local pass does not imply a cloud pass.
- A public license, third-party redistribution clearance and publication citation metadata require author confirmation.
