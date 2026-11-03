from pathlib import Path
import math
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from pymatgen.core import Element, Structure
from pymatgen.analysis.local_env import CrystalNN


base_dir = Path(__file__).resolve().parents[1]
input_cif_dir = base_dir / "cifs"
output_dir = base_dir / "outputs"
output_xlsx = output_dir / "pymatgen_features_parallel.xlsx"

MAX_WORKERS = 10
CHUNKSIZE = 12
PRINT_EVERY = 50


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def weighted_en_stats(structure):
    comp = structure.composition
    en_values = []
    weights = []

    for el, _ in comp.items():
        element = Element(el.symbol)
        en = element.X
        if en is None:
            raise ValueError(f"Missing Pauling electronegativity for element {element.symbol}")
        try:
            en = float(en)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Pauling electronegativity for element {element.symbol}") from exc
        if not math.isfinite(en):
            raise ValueError(f"Non-finite Pauling electronegativity for element {element.symbol}")

        frac = float(comp.get_atomic_fraction(element))
        if not math.isfinite(frac) or frac < 0:
            raise ValueError(f"Invalid atomic fraction for element {element.symbol}")
        en_values.append(en)
        weights.append(frac)

    total_w = sum(weights)
    if not en_values or not math.isfinite(total_w) or total_w <= 0:
        raise ValueError("No valid elemental composition for electronegativity calculation")

    avg_en = sum(w * x for w, x in zip(weights, en_values)) / total_w
    var_en = sum(w * (x - avg_en) ** 2 for w, x in zip(weights, en_values)) / total_w
    std_en = math.sqrt(max(var_en, 0.0))
    return float(avg_en), float(std_en)


def element_fraction(structure, symbol):
    return safe_float(structure.composition.get_atomic_fraction(Element(symbol)))


def halogen_fraction(structure):
    comp = structure.composition
    return float(sum(
        comp.get_atomic_fraction(Element(el.symbol))
        for el, _ in comp.items()
        if Element(el.symbol).is_halogen
    ))


def metal_fraction(structure):
    comp = structure.composition
    return float(sum(
        comp.get_atomic_fraction(Element(el.symbol))
        for el, _ in comp.items()
        if Element(el.symbol).is_metal
    ))


def mean_metal_coordination(structure):
    cnn = CrystalNN(
        weighted_cn=True,
        cation_anion=False,
        distance_cutoffs=(0.5, 1),
        x_diff_weight=1.5,
        porous_adjustment=True,
        search_cutoff=4.5,
        fingerprint_length=None,
    )

    metal_indices = [
        i for i, site in enumerate(structure.sites)
        if getattr(site.specie, "is_metal", False)
    ]
    if not metal_indices:
        raise ValueError("No metal sites found for coordination calculation")

    metal_cn_list = []
    for i in metal_indices:
        try:
            cn = float(cnn.get_cn(structure, i, use_weights=True))
        except Exception as exc:
            raise ValueError(f"CrystalNN failed at metal site index {i}") from exc
        if not math.isfinite(cn):
            raise ValueError(f"Non-finite coordination number at metal site index {i}")
        metal_cn_list.append(cn)

    return float(sum(metal_cn_list) / len(metal_cn_list))


FEATURE_COLUMNS = [
    "Material_Name",
    "C_Fraction",
    "O_Fraction",
    "N_Fraction",
    "Halogen_Fraction",
    "Metal_Fraction",
    "Avg_EN",
    "EN_std",
    "Mean_Metal_Coord",
    "Status",
    "Error",
]


def extract_features_from_cif(cif_path_str):
    cif_path = Path(cif_path_str)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structure = Structure.from_file(cif_path)
            avg_en, en_std = weighted_en_stats(structure)
            mean_coord = mean_metal_coordination(structure)

        row = {
            "Material_Name": cif_path.stem,
            "C_Fraction": element_fraction(structure, "C"),
            "O_Fraction": element_fraction(structure, "O"),
            "N_Fraction": element_fraction(structure, "N"),
            "Halogen_Fraction": halogen_fraction(structure),
            "Metal_Fraction": metal_fraction(structure),
            "Avg_EN": avg_en,
            "EN_std": en_std,
            "Mean_Metal_Coord": mean_coord,
            "Status": "Success",
            "Error": "",
        }
        for column in FEATURE_COLUMNS[1:-2]:
            if not math.isfinite(row[column]):
                raise ValueError(f"Non-finite feature value: {column}")
        return row
    except Exception as exc:
        row = {col: math.nan for col in FEATURE_COLUMNS[1:-2]}
        row.update({
            "Material_Name": cif_path.stem,
            "Status": "Failed",
            "Error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        })
        return row


def format_seconds(seconds):
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def main():
    if not input_cif_dir.exists():
        raise FileNotFoundError(f"CIF directory does not exist: {input_cif_dir}")

    cif_files = sorted(input_cif_dir.glob("*.cif"))
    if not cif_files:
        raise FileNotFoundError(f"No .cif files found in: {input_cif_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(cif_files)
    start_time = time.time()
    results = []

    print(f"CIF files: {total}; parallel workers: {MAX_WORKERS}")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        mapped_results = executor.map(
            extract_features_from_cif,
            [str(path) for path in cif_files],
            chunksize=CHUNKSIZE,
        )
        for idx, row in enumerate(mapped_results, start=1):
            results.append(row)
            if idx % PRINT_EVERY == 0 or idx == total:
                elapsed = time.time() - start_time
                remaining_seconds = elapsed / idx * (total - idx)
                print(f"Processed {idx}/{total}; estimated remaining time: {format_seconds(remaining_seconds)}")

    df = pd.DataFrame(results, columns=FEATURE_COLUMNS)
    df_success = df[df["Status"] == "Success"].sort_values("Material_Name")
    df_error = df[df["Status"] == "Failed"].sort_values("Material_Name")

    elapsed = time.time() - start_time
    summary_df = pd.DataFrame([
        ["Input CIF directory", str(input_cif_dir)],
        ["Output Excel file", str(output_xlsx)],
        ["Total CIF files", total],
        ["Successful extractions", len(df_success)],
        ["Failed extractions", len(df_error)],
        ["Parallel workers", MAX_WORKERS],
        ["Chunk size", CHUNKSIZE],
        ["Elapsed time", format_seconds(elapsed)],
        ["Average seconds per CIF", round(elapsed / total, 4)],
    ], columns=["Item", "Value"])

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        df_success.to_excel(writer, sheet_name="Features", index=False)
        if not df_error.empty:
            df_error.to_excel(writer, sheet_name="Errors", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"Finished: {len(df_success)} successful; {len(df_error)} failed")
    print(f"Output file: {output_xlsx}")


if __name__ == "__main__":
    main()
