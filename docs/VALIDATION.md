# Validation scope

Local checks were executed with Python 3.10.9, NumPy 1.26.4 and pandas 2.2.3 on Windows.
The recorded full module requirements are retained; they are not a claim that every listed
package was exercised in this local check.

| Check | Result |
| --- | --- |
| Syntax of all 10 Python files | Passed |
| Unchanged algorithm function bodies, except documented IO adapters | Passed |
| Optimization input validation: 105 configs, 7,532 MOFs, five feature matrices | Passed |
| Single-task `--check-only` | Passed |
| Batch dry list: 105 tasks, no execution | Passed |
| Benchmark readers: five tables x three targets for both scripts | Passed; 6,025 train and 1,507 test rows per target |
| Oracle input reader, including legacy EtCL header | Passed |
| Three Release archives: CRC and every member SHA256 | Passed |
| Original submission files unchanged | Passed; 7,697 files |
| GitHub Actions, benchmark model fitting, GP/device fallback and full campaign | Not executed |

The input-only demo took 1.412 seconds in this local execution. Timing depends on hardware
and does not estimate a full optimization run. No installation timing is asserted.
