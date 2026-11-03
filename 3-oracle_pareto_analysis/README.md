# Oracle Pareto analysis

The input is the Manual table already included in the benchmark module. Use an environment
with NumPy, pandas and SciPy (the benchmark requirements include these dependencies).
From this module directory:

```sh
python -B scripts/oracle_pareto_analysis.py --input ../2-representation_benchmark/data/Manual.csv --output-dir results
```

The command computes full-library Pareto layers and writes `oracle_membership.csv` and
`oracle_statistics.csv` to `results/`. Existing `expected_outputs/` remain unchanged.
It is a full analysis, not the input-only demo. The legacy `EtCL` input header is normalized
to `EtCl` in memory; historical reference files keep their original headings.
