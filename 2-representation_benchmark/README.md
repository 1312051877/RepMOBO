# Representation benchmark

Five representations are benchmarked using LightGBM and Ridge for CEES, DMMP and EtCl.
`data/Manual.csv` and `data/fixed_splits.xlsx` are tracked here. Extract
`benchmark_features.zip` into the repository root to restore the other four CSVs.

Use a separate Python 3.10 environment for this module; its recorded dependencies differ
from the optimization environment.

```sh
python -m pip install -r requirements.txt
python -B scripts/LGBM.py
python -B scripts/Ridge.py
```

Run these commands from this module directory. They perform new hyperparameter searches
and model training; they are not fast validation commands. Outputs go to `results_lgbm/`
and `results_ridge/`. The existing `expected_outputs/` files remain reference artifacts.
`configs/ALL_BEST_PARAMS_*.xlsx` document saved parameters; the scripts search anew and do
not automatically load those workbooks as fixed-fit parameters.

`expected_outputs/benchmark_metrics.xlsx` contains reported benchmark metrics and
`test_predictions.csv` contains merged test predictions. `EtCL` in the original Manual
CSV and split-workbook sheet is recognized as `EtCl` in memory. Source CSV contents and values remain unchanged.
