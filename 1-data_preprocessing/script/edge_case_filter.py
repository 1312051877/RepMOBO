# -*- coding: utf-8 -*-
import os
from pathlib import Path
import math
import pandas as pd


# =========================================================
# 1. Path configuration
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_XLSX = str(Path(os.environ.get("REPMOBO_EDGE_INPUT", str(BASE_DIR / "data" / "Data7.xlsx"))))
OUTPUT_XLSX = str(Path(os.environ.get("REPMOBO_EDGE_OUTPUT", str(BASE_DIR / "Data7_edge_case_filtered.xlsx"))))

# Change this setting if the data are not in the first sheet
INPUT_SHEET_NAME = 0  # 0 selects the first sheet; a specific name such as "Sheet1" can also be used


# =========================================================
# 2. Column configuration
# =========================================================
# Assuming the column positions are:
# Column F -> LCD
# Column G -> PLD
# Column H -> Density
# Column M -> Mean_Metal_Coord
#
# Using column names directly below is more robust.
# If the headers differ, replace these names with the actual column names in the Excel file.
COL_LCD = "LCD"
COL_PLD = "PLD"
COL_DENSITY = "Density"
COL_MEAN_METAL_COORD = "Mean_Metal_Coord"

# To prioritize material names in the output, specify the corresponding column name here;
# otherwise, leave it as None
COL_MATERIAL = "Material"   # Set to None if this column is absent


# =========================================================
# 3. Threshold configuration
# =========================================================
DENSITY_STRICT_MIN = 100.0       # kg/m^3
LARGE_PORE_THRESHOLD = 50.0      # Å
DENSITY_LARGE_PORE_MAX = 300.0   # kg/m^3
METAL_COORD_MAX = 12.0
METAL_COORD_MIN = 2.0


# =========================================================
# 4. Utility functions
# =========================================================
def safe_float(x):
    """
    Attempt to convert any value to float.
    Return NaN if conversion fails.
    This handles missing values, strings, and unexpected characters in the Excel data.
    """
    try:
        return float(x)
    except Exception:
        return math.nan


def require_columns(df: pd.DataFrame, required_cols):
    """
    Check whether the input table contains the required columns.
    If any columns are missing, raise an error and list all available column names for troubleshooting.
    """
    missing = [c for c in required_cols if c is not None and c not in df.columns]
    if missing:
        raise KeyError(
            "The input table is missing the following columns: {}\nAvailable columns in the table:\n{}".format(
                missing, list(df.columns)
            )
        )


def decide_remove_reasons(row):
    """
    Determine whether a sample row should be removed and return a list of removal reasons.

    Returns:
        reasons: list[str]
            Empty list -> retain
            Non-empty list -> remove
    """
    lcd = safe_float(row[COL_LCD])
    pld = safe_float(row[COL_PLD])
    density = safe_float(row[COL_DENSITY])
    mean_coord = safe_float(row[COL_MEAN_METAL_COORD])

    reasons = []

    # -----------------------------------------------------
    # Rule 1: Extremely low density
    # Density < 100 -> remove
    # -----------------------------------------------------
    if not math.isnan(density) and density < DENSITY_STRICT_MIN:
        reasons.append("density_lt_100")

    # -----------------------------------------------------
    # Rule 2: Large pores + low density
    # (LCD > 50 or PLD > 50) and Density < 300 -> remove
    # -----------------------------------------------------
    has_large_lcd = (not math.isnan(lcd)) and (lcd > LARGE_PORE_THRESHOLD)
    has_large_pld = (not math.isnan(pld)) and (pld > LARGE_PORE_THRESHOLD)
    has_low_density_for_large_pore = (not math.isnan(density)) and (density < DENSITY_LARGE_PORE_MAX)

    if (has_large_lcd or has_large_pld) and has_low_density_for_large_pore:
        reasons.append("large_pore_and_low_density")

    # -----------------------------------------------------
    # Rule 3: Excessively high mean metal coordination number
    # Mean_Metal_Coord > 12 -> remove
    # -----------------------------------------------------
    if not math.isnan(mean_coord) and mean_coord > METAL_COORD_MAX:
        reasons.append("mean_metal_coord_gt_12")

    # -----------------------------------------------------
    # Rule 4: Excessively low mean metal coordination number
    # Mean_Metal_Coord < 2 -> remove
    # -----------------------------------------------------
    if not math.isnan(mean_coord) and mean_coord < METAL_COORD_MIN:
        reasons.append("mean_metal_coord_lt_2")

    return reasons


# =========================================================
# 5. Main program
# =========================================================
def main():
    # -----------------------------
    # 5.1 Read Excel
    # -----------------------------
    if not os.path.exists(INPUT_XLSX):
        raise FileNotFoundError(f"Input file does not exist: {INPUT_XLSX}")

    print("=" * 80)
    print("Starting MOF edge-case filtering")
    print(f"Input file: {INPUT_XLSX}")
    print(f"Input sheet: {INPUT_SHEET_NAME}")
    print(f"Output file: {OUTPUT_XLSX}")
    print("=" * 80)

    df = pd.read_excel(INPUT_XLSX, sheet_name=INPUT_SHEET_NAME)

    # -----------------------------
    # 5.2 Check that the columns exist
    # -----------------------------
    required_cols = [
        COL_LCD,
        COL_PLD,
        COL_DENSITY,
        COL_MEAN_METAL_COORD,
    ]
    if COL_MATERIAL is not None:
        required_cols.append(COL_MATERIAL)

    require_columns(df, required_cols)

    # -----------------------------
    # 5.3 Determine removal reasons for each row
    # -----------------------------
    remove_flags = []
    remove_reason_texts = []
    rule1_flags = []
    rule2_flags = []
    rule3_flags = []
    rule4_flags = []

    for _, row in df.iterrows():
        reasons = decide_remove_reasons(row)

        remove_flags.append("Y" if reasons else "N")
        remove_reason_texts.append("; ".join(reasons))

        rule1_flags.append("Y" if "density_lt_100" in reasons else "N")
        rule2_flags.append("Y" if "large_pore_and_low_density" in reasons else "N")
        rule3_flags.append("Y" if "mean_metal_coord_gt_12" in reasons else "N")
        rule4_flags.append("Y" if "mean_metal_coord_lt_2" in reasons else "N")

    df["EdgeCase_Remove"] = remove_flags
    df["EdgeCase_Remove_Reasons"] = remove_reason_texts
    df["Flag_Density_LT_100"] = rule1_flags
    df["Flag_LargePore_And_LowDensity"] = rule2_flags
    df["Flag_MeanMetalCoord_GT_12"] = rule3_flags
    df["Flag_MeanMetalCoord_LT_2"] = rule4_flags

    # -----------------------------
    # 5.4 Split into retained / removed
    # -----------------------------
    retained_df = df[df["EdgeCase_Remove"] == "N"].copy()
    removed_df = df[df["EdgeCase_Remove"] == "Y"].copy()

    # If the Material column is present, sort by Material for easier inspection
    if COL_MATERIAL is not None and COL_MATERIAL in retained_df.columns:
        retained_df.sort_values(by=[COL_MATERIAL], inplace=True)
        removed_df.sort_values(by=[COL_MATERIAL], inplace=True)

    # -----------------------------
    # 5.5 Generate summary
    # -----------------------------
    summary_rows = [
        {"Metric": "Input rows", "Value": len(df)},
        {"Metric": "Retained rows", "Value": len(retained_df)},
        {"Metric": "Removed rows", "Value": len(removed_df)},
        {"Metric": "Flag: density_lt_100", "Value": int((df["Flag_Density_LT_100"] == "Y").sum())},
        {"Metric": "Flag: large_pore_and_low_density", "Value": int((df["Flag_LargePore_And_LowDensity"] == "Y").sum())},
        {"Metric": "Flag: mean_metal_coord_gt_12", "Value": int((df["Flag_MeanMetalCoord_GT_12"] == "Y").sum())},
        {"Metric": "Flag: mean_metal_coord_lt_2", "Value": int((df["Flag_MeanMetalCoord_LT_2"] == "Y").sum())},
    ]
    summary_df = pd.DataFrame(summary_rows)

    # -----------------------------
    # 5.6 Write Excel
    # -----------------------------
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        retained_df.to_excel(writer, sheet_name="retained", index=False)
        removed_df.to_excel(writer, sheet_name="removed", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    # -----------------------------
    # 5.7 Print results
    # -----------------------------
    print("\nFiltering complete")
    print(f"Total samples: {len(df)}")
    print(f"Retained samples: {len(retained_df)}")
    print(f"Removed samples: {len(removed_df)}")
    print(f"Output file saved to: {OUTPUT_XLSX}")
    print("=" * 80)


if __name__ == "__main__":
    main()