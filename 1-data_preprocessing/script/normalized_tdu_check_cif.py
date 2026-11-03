import os
import re
import math
import time
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import pandas as pd
from pymatgen.core import Structure, Element


# =========================================================
# 1. Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = str(BASE_DIR / "cifs")
OUTPUT_DIR = str(BASE_DIR / "outputs" / "structure_checks")
OUTPUT_XLSX = os.path.join(OUTPUT_DIR, "normalized_tdu_and_cif_checks.xlsx")
CHECKPOINT_CSV = os.path.join(OUTPUT_DIR, "normalized_tdu_and_cif_checks_checkpoint.csv")

# =========================================================
# 2. Parallel settings
# =========================================================
CPU_COUNT = os.cpu_count() or 1
MAX_WORKERS = 10

# =========================================================
# 3. Test mode
# =========================================================
TEST_MODE = False
TEST_N = 20

# =========================================================
# 4. Check thresholds
# =========================================================
SEVERE_CLOSE_CONTACT = 0.50   # Angstrom
WARN_CLOSE_CONTACT = 0.80     # Angstrom
NEIGHBOR_SEARCH_RADIUS = 1.20 # Angstrom

PRINT_EVERY = 50
CHECKPOINT_EVERY = 200


# =========================================================
# 5. Utilities
# =========================================================
def ensure_dir(path_str: str):
    os.makedirs(path_str, exist_ok=True)


def list_cif_files(input_dir: str):
    p = Path(input_dir)

    files = []
    seen = set()

    for x in p.iterdir():
        if x.is_file() and x.suffix.lower() == ".cif":
            key = str(x.resolve()).lower()
            if key not in seen:
                seen.add(key)
                files.append(str(x))

    files.sort()
    return files


def stringify_list(xs):
    xs = [str(x) for x in xs if str(x).strip()]
    return "; ".join(xs)


def count_duplicates(items):
    c = Counter(items)
    return sum(v - 1 for v in c.values() if v > 1)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def is_valid_element_symbol(sym: str):
    try:
        Element(sym)
        return True
    except Exception:
        return False


def normalize_element_symbol(sym: str):
    if sym is None:
        return None

    s = str(sym).strip()
    if not s:
        return None

    # Try the original symbol
    try:
        return Element(s).symbol
    except Exception:
        pass

    # Extract the first one or two letters
    m = re.match(r"([A-Za-z]{1,2})", s)
    if not m:
        return None

    cand = m.group(1)
    cand = cand[0].upper() + cand[1:].lower()

    try:
        return Element(cand).symbol
    except Exception:
        pass

    # Fall back to a single letter
    cand1 = cand[0].upper()
    try:
        return Element(cand1).symbol
    except Exception:
        return None


def parse_formula_sum_elements_simple(formula_sum: str):
    if not formula_sum or not str(formula_sum).strip():
        return set()

    tokens = re.findall(r"([A-Z][a-z]?)", str(formula_sum))
    valid = set()
    for t in tokens:
        if is_valid_element_symbol(t):
            valid.add(t)
    return valid


# =========================================================
# 6. Raw CIF parsing
# =========================================================
TOKEN_RE = re.compile(r"""'(?:[^']*)'|"(?:[^"]*)"|[^\s]+""")


def strip_quotes(s: str):
    s = s.strip()
    if len(s) >= 2 and ((s[0] == "'" and s[-1] == "'") or (s[0] == '"' and s[-1] == '"')):
        return s[1:-1]
    return s


def tokenize_cif_line(line: str):
    return [strip_quotes(m.group(0)) for m in TOKEN_RE.finditer(line)]


def read_raw_cif_tags(cif_path: str):
    raw = {
        "raw_read_ok": False,
        "raw_read_error": "",
        "raw_type_symbols": [],
        "raw_labels": [],
        "chemical_formula_sum": "",
    }

    try:
        with open(cif_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if not line or line.startswith("#"):
                i += 1
                continue

            low = line.lower()

            # Single-value tag: _chemical_formula_sum
            if line.startswith("_chemical_formula_sum"):
                toks = tokenize_cif_line(line)
                if len(toks) >= 2:
                    raw["chemical_formula_sum"] = " ".join(toks[1:]).strip()
                else:
                    j = i + 1
                    while j < len(lines):
                        nxt = lines[j].strip()
                        if nxt and not nxt.startswith("#"):
                            raw["chemical_formula_sum"] = strip_quotes(nxt)
                            break
                        j += 1
                i += 1
                continue

            # Loop block
            if low == "loop_":
                i += 1
                cols = []

                while i < len(lines):
                    s = lines[i].strip()
                    if not s or s.startswith("#"):
                        i += 1
                        continue
                    if s.startswith("_"):
                        cols.append(s.split()[0])
                        i += 1
                    else:
                        break

                need_atom_site = any(
                    c in cols for c in ["_atom_site_type_symbol", "_atom_site_label"]
                )

                buffer = []
                data_rows = []

                while i < len(lines):
                    s = lines[i].strip()
                    low2 = s.lower()

                    if not s or s.startswith("#"):
                        i += 1
                        continue

                    if low2 == "loop_" or s.startswith("_") or low2.startswith("data_"):
                        break

                    toks = tokenize_cif_line(s)
                    buffer.extend(toks)

                    while len(buffer) >= len(cols) and len(cols) > 0:
                        row = buffer[:len(cols)]
                        buffer = buffer[len(cols):]
                        data_rows.append(row)

                    i += 1

                if need_atom_site and cols:
                    idx_type = cols.index("_atom_site_type_symbol") if "_atom_site_type_symbol" in cols else None
                    idx_label = cols.index("_atom_site_label") if "_atom_site_label" in cols else None

                    for row in data_rows:
                        if idx_type is not None and idx_type < len(row):
                            raw["raw_type_symbols"].append(row[idx_type])
                        if idx_label is not None and idx_label < len(row):
                            raw["raw_labels"].append(row[idx_label])

                continue

            i += 1

        raw["raw_read_ok"] = True

    except Exception as e:
        raw["raw_read_ok"] = False
        raw["raw_read_error"] = f"{type(e).__name__}: {e}"

    return raw


# =========================================================
# 7. Structure calculations
# =========================================================
def compute_min_interatomic_distance(structure: Structure, radius=NEIGHBOR_SEARCH_RADIUS):
    """
    Find the shortest interatomic distance within the search radius.
    """
    min_dist = math.inf

    try:
        for site in structure.sites:
            neighs = structure.get_neighbors(site, radius)
            for nn in neighs:
                d = float(nn.nn_distance)
                if d > 1e-8 and d < min_dist:
                    min_dist = d
    except Exception:
        return math.nan

    return float(min_dist) if not math.isinf(min_dist) else math.nan


def compute_tdu_from_counts(elem_counts: dict):
    """
    TDU = (2C + 2 + N - (H + F + Cl + Br + I)) / 2
    Normalized_TDU = TDU / C
    """
    C = float(elem_counts.get("C", 0.0))
    H = float(elem_counts.get("H", 0.0) + elem_counts.get("D", 0.0))
    N = float(elem_counts.get("N", 0.0))
    F = float(elem_counts.get("F", 0.0))
    Cl = float(elem_counts.get("Cl", 0.0))
    Br = float(elem_counts.get("Br", 0.0))
    I = float(elem_counts.get("I", 0.0))

    tdu = (2 * C + 2 + N - (H + F + Cl + Br + I)) / 2.0
    normalized_tdu = (tdu / C) if C > 0 else math.nan

    return {
        "TDU_raw": tdu,
        "Normalized_TDU": normalized_tdu,
        "C_count": C,
        "H_count": H,
        "N_count": N,
        "F_count": F,
        "Cl_count": Cl,
        "Br_count": Br,
        "I_count": I,
        "Halogen_count": F + Cl + Br + I,
    }


# =========================================================
# 8. Analyze one CIF
# =========================================================
def analyze_one_cif(cif_path: str):
    warnings.filterwarnings(
        "ignore",
        message="dict interface .* is deprecated.*",
        category=DeprecationWarning
    )
    warnings.filterwarnings(
        "ignore",
        message="Issues encountered while parsing CIF:.*",
        category=UserWarning
    )
    file_name = os.path.basename(cif_path)
    stem = os.path.splitext(file_name)[0]

    row = {
        "Material": stem,
        "File_Name": file_name,
        "File_Path": cif_path,

        "Issue_Level": "ERROR",
        "Issue_Summary": "",
        "Parse_Error_Pymatgen": "",

        "Raw_Read_OK": False,
        "Raw_Read_Error": "",
        "Raw_Invalid_Type_Symbols": "",
        "Duplicate_Atom_Labels": 0,
        "Chemical_Formula_Sum": "",
        "Formula_Sum_ElementSet_Mismatch": "N",


        "Volume_A3": math.nan,
        "a_A": math.nan,
        "b_A": math.nan,
        "c_A": math.nan,
        "alpha_deg": math.nan,
        "beta_deg": math.nan,
        "gamma_deg": math.nan,

        "Num_Sites_Pymatgen": math.nan,

        "Partial_Occupancy_Site_Count": 0,
        "Occupancy_GT1_Site_Count": 0,
        "Occupancy_LE0_Site_Count": 0,
        "Has_Disorder_Site": "N",

        "Min_Interatomic_Dist_A": math.nan,
        "Severe_Close_Contact_Flag": "N",
        "Warn_Close_Contact_Flag": "N",

        "Has_Metal": "N",
        "Has_Carbon": "N",

        "C_count": 0.0,
        "H_count": 0.0,
        "N_count": 0.0,
        "O_count": 0.0,
        "S_count": 0.0,
        "F_count": 0.0,
        "Cl_count": 0.0,
        "Br_count": 0.0,
        "I_count": 0.0,
        "Halogen_count": 0.0,
        "Metal_count": 0.0,
        "Total_Atoms_Occ_UnitCell": 0.0,

        "TDU_raw": math.nan,
        "Normalized_TDU": math.nan,
        "TDU_Negative_Flag": "N",
        "No_Carbon_Flag": "N",
        "No_Metal_Flag": "N",
    }

    issues = []

    # -----------------------------------------------------
    # 1) Raw CIF checks
    # -----------------------------------------------------
    raw = read_raw_cif_tags(cif_path)
    row["Raw_Read_OK"] = raw["raw_read_ok"]
    row["Raw_Read_Error"] = raw["raw_read_error"]
    row["Duplicate_Atom_Labels"] = count_duplicates(raw["raw_labels"])
    row["Chemical_Formula_Sum"] = raw["chemical_formula_sum"]

    raw_invalid = []
    for s in raw["raw_type_symbols"]:
        guessed = normalize_element_symbol(s)
        if guessed is None:
            raw_invalid.append(s)

    if raw_invalid:
        row["Raw_Invalid_Type_Symbols"] = stringify_list(sorted(set(raw_invalid)))
        issues.append("invalid_raw_type_symbol")

    if row["Duplicate_Atom_Labels"] > 0:
        issues.append("duplicate_atom_labels")

    # -----------------------------------------------------
    # 2) Parse the structure with pymatgen
    # -----------------------------------------------------
    try:
        structure = Structure.from_file(cif_path)

        # Lattice properties
        lat = structure.lattice
        row["Volume_A3"] = float(lat.volume)
        row["a_A"] = float(lat.a)
        row["b_A"] = float(lat.b)
        row["c_A"] = float(lat.c)
        row["alpha_deg"] = float(lat.alpha)
        row["beta_deg"] = float(lat.beta)
        row["gamma_deg"] = float(lat.gamma)
        row["Num_Sites_Pymatgen"] = len(structure)

        # Lattice validation
        if row["Volume_A3"] <= 0:
            issues.append("nonpositive_volume")
        if min(row["a_A"], row["b_A"], row["c_A"]) <= 0:
            issues.append("nonpositive_lattice_length")
        if not (0 < row["alpha_deg"] < 180 and 0 < row["beta_deg"] < 180 and 0 < row["gamma_deg"] < 180):
            issues.append("invalid_lattice_angle")

        # Occupancy and disorder checks
        partial_occ = 0
        occ_gt1 = 0
        occ_le0 = 0
        has_disorder = False

        for site in structure.sites:
            occ = float(site.species.num_atoms)

            if occ < 0.999:
                partial_occ += 1
            if occ > 1.001:
                occ_gt1 += 1
            if occ <= 0:
                occ_le0 += 1
            if not site.is_ordered:
                has_disorder = True

        row["Partial_Occupancy_Site_Count"] = partial_occ
        row["Occupancy_GT1_Site_Count"] = occ_gt1
        row["Occupancy_LE0_Site_Count"] = occ_le0
        row["Has_Disorder_Site"] = "Y" if has_disorder else "N"

        if partial_occ > 0:
            issues.append("partial_occupancy")
        if occ_gt1 > 0:
            issues.append("occupancy_gt_1")
        if occ_le0 > 0:
            issues.append("occupancy_le_0")
        if has_disorder:
            issues.append("disorder_site_present")

        # Elemental composition
        elem_counts = structure.composition.get_el_amt_dict()
        row["Total_Atoms_Occ_UnitCell"] = float(sum(float(v) for v in elem_counts.values()))

        row["C_count"] = float(elem_counts.get("C", 0.0))
        row["H_count"] = float(elem_counts.get("H", 0.0) + elem_counts.get("D", 0.0))
        row["N_count"] = float(elem_counts.get("N", 0.0))
        row["O_count"] = float(elem_counts.get("O", 0.0))
        row["S_count"] = float(elem_counts.get("S", 0.0))
        row["F_count"] = float(elem_counts.get("F", 0.0))
        row["Cl_count"] = float(elem_counts.get("Cl", 0.0))
        row["Br_count"] = float(elem_counts.get("Br", 0.0))
        row["I_count"] = float(elem_counts.get("I", 0.0))
        row["Halogen_count"] = row["F_count"] + row["Cl_count"] + row["Br_count"] + row["I_count"]

        metal_count = 0.0
        has_metal = False
        has_carbon = False

        for sym, cnt in elem_counts.items():
            try:
                e = Element(sym)
                if e.is_metal:
                    metal_count += float(cnt)
                    has_metal = True
                if sym == "C" and cnt > 0:
                    has_carbon = True
            except Exception:
                pass

        row["Metal_count"] = metal_count
        row["Has_Metal"] = "Y" if has_metal else "N"
        row["Has_Carbon"] = "Y" if has_carbon else "N"

        if not has_carbon:
            row["No_Carbon_Flag"] = "Y"
            issues.append("no_carbon")
        if not has_metal:
            row["No_Metal_Flag"] = "Y"
            issues.append("no_metal")

        # Cross-check formula and parsed elements
        if row["Chemical_Formula_Sum"]:
            formula_set = parse_formula_sum_elements_simple(row["Chemical_Formula_Sum"])
            count_set = set(k for k, v in elem_counts.items() if v > 0)
            if formula_set and formula_set != count_set:
                row["Formula_Sum_ElementSet_Mismatch"] = "Y"
                issues.append("formula_sum_elementset_mismatch")

        # TDU / Normalized_TDU
        tdu_info = compute_tdu_from_counts(elem_counts)
        row["TDU_raw"] = tdu_info["TDU_raw"]
        row["Normalized_TDU"] = tdu_info["Normalized_TDU"]

        if not math.isnan(row["TDU_raw"]) and row["TDU_raw"] < -1e-8:
            row["TDU_Negative_Flag"] = "Y"
            issues.append("negative_tdu")

        # Minimum interatomic distance
        min_dist = compute_min_interatomic_distance(structure)
        row["Min_Interatomic_Dist_A"] = min_dist

        if not math.isnan(min_dist):
            if min_dist < SEVERE_CLOSE_CONTACT:
                row["Severe_Close_Contact_Flag"] = "Y"
                row["Warn_Close_Contact_Flag"] = "Y"
                issues.append("severe_close_contact")
            elif min_dist < WARN_CLOSE_CONTACT:
                row["Warn_Close_Contact_Flag"] = "Y"
                issues.append("close_contact_warning")

    except Exception as e:
        row["Parse_Error_Pymatgen"] = f"{type(e).__name__}: {e}"
        issues.append("pymatgen_parse_failed")

    # -----------------------------------------------------
    # 3) Issue classification
    # -----------------------------------------------------
    hard_errors = {
        "pymatgen_parse_failed",
        "invalid_raw_type_symbol",
        "nonpositive_volume",
        "nonpositive_lattice_length",
        "invalid_lattice_angle",
        "occupancy_gt_1",
        "occupancy_le_0",
        "severe_close_contact",
    }

    if any(x in hard_errors for x in issues):
        row["Issue_Level"] = "ERROR"
    elif len(issues) > 0:
        row["Issue_Level"] = "WARN"
    else:
        row["Issue_Level"] = "OK"

    row["Issue_Summary"] = stringify_list(sorted(set(issues)))
    return row


# =========================================================
# 9. Main workflow
# =========================================================
def main():
    if not os.path.isdir(INPUT_DIR):
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")

    ensure_dir(OUTPUT_DIR)

    cif_files = list_cif_files(INPUT_DIR)
    if not cif_files:
        raise RuntimeError(f"No CIF files found in directory: {INPUT_DIR}")

    if TEST_MODE:
        cif_files = cif_files[:TEST_N]

    print("=" * 88)
    print("Starting Normalized_TDU calculation and CIF quality checks")
    print(f"Input directory: {INPUT_DIR}")
    print(f"CIF count: {len(cif_files)}")
    print(f"Output file: {OUTPUT_XLSX}")
    print(f"Logical CPU count: {CPU_COUNT}")
    print(f"Parallel workers: {MAX_WORKERS}")
    print(f"TEST_MODE = {TEST_MODE}")
    if TEST_MODE:
        print(f"TEST_N = {TEST_N}")
    print("=" * 88)

    rows = []
    total = len(cif_files)
    start_time = time.time()

    ok_count = 0
    warn_count = 0
    error_count = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(analyze_one_cif, cif_path): cif_path for cif_path in cif_files}

        done_n = 0
        for future in as_completed(futures):
            done_n += 1
            row = future.result()
            rows.append(row)

            if row["Issue_Level"] == "OK":
                ok_count += 1
            elif row["Issue_Level"] == "WARN":
                warn_count += 1
            else:
                error_count += 1

            elapsed = time.time() - start_time
            speed = done_n / elapsed if elapsed > 0 else 0
            remain = (total - done_n) / speed if speed > 0 else float("inf")

            if done_n % PRINT_EVERY == 0 or done_n == total:
                remain_str = "unknown" if math.isinf(remain) else f"{int(remain // 60)}m {int(remain % 60)}s"
                print(
                    f"[{done_n}/{total}] "
                    f"Elapsed {int(elapsed)}s | "
                    f"Estimated remaining {remain_str} | "
                    f"OK={ok_count}, WARN={warn_count}, ERROR={error_count}"
                )

            if done_n % CHECKPOINT_EVERY == 0 or done_n == total:
                df_ckpt = pd.DataFrame(rows)
                if not df_ckpt.empty:
                    df_ckpt.sort_values(by=["Material"], inplace=True)
                    df_ckpt.to_csv(CHECKPOINT_CSV, index=False, encoding="utf-8-sig")

    df = pd.DataFrame(rows)
    df.sort_values(by=["Material"], inplace=True)

    flagged_df = df[df["Issue_Level"].isin(["ERROR", "WARN"])].copy()

    summary_rows = [
        {"Metric": "Total CIF files", "Value": len(df)},
        {"Metric": "Issue Level = OK", "Value": int((df["Issue_Level"] == "OK").sum())},
        {"Metric": "Issue Level = WARN", "Value": int((df["Issue_Level"] == "WARN").sum())},
        {"Metric": "Issue Level = ERROR", "Value": int((df["Issue_Level"] == "ERROR").sum())},
        {"Metric": "Has invalid raw type symbol", "Value": int(df["Raw_Invalid_Type_Symbols"].fillna("").astype(str).str.len().gt(0).sum())},
        {"Metric": "Duplicate atom labels CIFs", "Value": int(df["Duplicate_Atom_Labels"].fillna(0).gt(0).sum())},
        {"Metric": "Partial occupancy CIFs", "Value": int(df["Partial_Occupancy_Site_Count"].fillna(0).gt(0).sum())},
        {"Metric": "Occupancy > 1 CIFs", "Value": int(df["Occupancy_GT1_Site_Count"].fillna(0).gt(0).sum())},
        {"Metric": "Occupancy <= 0 CIFs", "Value": int(df["Occupancy_LE0_Site_Count"].fillna(0).gt(0).sum())},
        {"Metric": "Disorder site CIFs", "Value": int((df["Has_Disorder_Site"] == "Y").sum())},
        {"Metric": "Formula_sum mismatch CIFs", "Value": int((df["Formula_Sum_ElementSet_Mismatch"] == "Y").sum())},
        {"Metric": "Severe close contact CIFs", "Value": int((df["Severe_Close_Contact_Flag"] == "Y").sum())},
        {"Metric": "Close contact warning CIFs", "Value": int((df["Warn_Close_Contact_Flag"] == "Y").sum())},
        {"Metric": "No metal CIFs", "Value": int((df["No_Metal_Flag"] == "Y").sum())},
        {"Metric": "No carbon CIFs", "Value": int((df["No_Carbon_Flag"] == "Y").sum())},
        {"Metric": "Negative TDU CIFs", "Value": int((df["TDU_Negative_Flag"] == "Y").sum())},
        {"Metric": "Has valid Normalized_TDU", "Value": int(df["Normalized_TDU"].notna().sum())},
    ]
    summary_df = pd.DataFrame(summary_rows)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="tdu_results")
        flagged_df.to_excel(writer, index=False, sheet_name="flagged_issues")
        summary_df.to_excel(writer, index=False, sheet_name="summary")

    total_elapsed = time.time() - start_time

    print("\n" + "=" * 88)
    print("Processing completed")
    print(f"Total runtime: {int(total_elapsed)} s")
    print(f"Results saved to: {OUTPUT_XLSX}")
    print(f"Checkpoint CSV: {CHECKPOINT_CSV}")
    print("=" * 88)


if __name__ == "__main__":
    main()
