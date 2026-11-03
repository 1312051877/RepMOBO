# Structure preprocessing

Download and extract `cif_structures.zip` into the repository root (see [data access](../docs/DATA.md)).
This restores 7,532 CIFs and `MOF_ID_to_CSD_mapping.xlsx` under `cifs/`.
The source-structure redistribution terms must be checked before a public release.

Use a separate Python environment with pandas, openpyxl and pymatgen. This package does not
establish a tested pymatgen version. The Zeo++ driver additionally requires a working Windows
Cygwin/Zeo++ installation; its external tools are not bundled.

Run from this module directory:

```sh
python -B script/normalized_tdu_check_cif.py
python -B script/pymatgen_features_clean.py
```

These read `cifs/` and write under `outputs/`. They process structures and may take substantial time.

`zeopp_features_core.py` retains the original Windows/Cygwin command construction. Set
`CYGWIN_BASH`, `ZEO_DIR_WIN` and `ZEO_DIR_CYG` environment variables to your own installation
paths before running it. It has not been tested as a native Linux driver.

`edge_case_filter.py` requires an upstream descriptor workbook that is **not included**.
Set `REPMOBO_EDGE_INPUT` to that workbook and optionally `REPMOBO_EDGE_OUTPUT` to the desired
output workbook. The default input is `data/Data7.xlsx`. Required columns and threshold rules
are specified in the script. The supplied final CIF library does not replace this missing
pre-filter workbook, so end-to-end regeneration of the initial filtering is not claimed.
