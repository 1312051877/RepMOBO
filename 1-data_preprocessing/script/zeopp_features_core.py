import os
import re
import time
import math
import traceback
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
CYGWIN_BASH = os.environ.get("CYGWIN_BASH", "")
ZEO_DIR_WIN = os.environ.get("ZEO_DIR_WIN", "")
ZEO_DIR_CYG = os.environ.get("ZEO_DIR_CYG", "")
CIF_DIR = str(BASE_DIR / "cifs")
OUTPUT_XLSX = str(BASE_DIR / "outputs" / "zeopp_features.xlsx")

RAW_ROOT = str(BASE_DIR / "outputs" / "zeopp_raw")
RES_DIR = os.path.join(RAW_ROOT, "res")
SA_DIR = os.path.join(RAW_ROOT, "sa")
VOL_DIR = os.path.join(RAW_ROOT, "vol")
VOLPO_DIR = os.path.join(RAW_ROOT, "volpo")
LOG_DIR = str(BASE_DIR / "outputs" / "zeopp_logs")

N_WORKERS = 4
PROBE_RADIUS = 1.82
CHAN_RADIUS = 1.82
SA_SAMPLES_PER_ATOM = 2000
VOL_SAMPLES_TOTAL = 50000
VOLPO_SAMPLES_TOTAL = 200000
TIMEOUT_SECONDS = 1800
SKIP_IF_DONE = False
TEST_MODE = False
TEST_N = 3


def ensure_dir(path_str):
    Path(path_str).mkdir(parents=True, exist_ok=True)


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def win_to_cyg(path_str):
    path = Path(path_str).resolve()
    value = str(path).replace("\\", "/")
    drive = value[0].lower()
    rest = value[2:]
    return f"/cygdrive/{drive}{rest}"


def read_text_if_exists(path_str):
    path = Path(path_str)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_first_float(pattern, source_text):
    match = re.search(pattern, source_text, flags=re.I | re.S)
    if not match:
        return None
    return safe_float(match.group(1))


def parse_res_file(res_file):
    result = {"LCD": None, "PLD": None}
    source_text = read_text_if_exists(res_file).strip()
    if not source_text:
        return result

    values = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?", source_text)
    if len(values) >= 3:
        result["LCD"] = safe_float(values[-3])
        result["PLD"] = safe_float(values[-2])
    return result


def parse_density_file(file_path):
    source_text = read_text_if_exists(file_path)
    if not source_text.strip():
        return None
    return extract_first_float(r"Density:\s*([0-9Ee+\-.]+)", source_text)


def build_cmd_str(cif_path, res_out, sa_out, vol_out, volpo_out):
    cif_cyg = win_to_cyg(cif_path)
    res_cyg = win_to_cyg(res_out)
    sa_cyg = win_to_cyg(sa_out)
    vol_cyg = win_to_cyg(vol_out)
    volpo_cyg = win_to_cyg(volpo_out)

    commands = [
        f"cd '{ZEO_DIR_CYG}'",
        f"./network -ha -res '{res_cyg}' '{cif_cyg}'",
        f"./network -ha -sa {CHAN_RADIUS} {PROBE_RADIUS} {SA_SAMPLES_PER_ATOM} '{sa_cyg}' '{cif_cyg}'",
        f"./network -ha -vol {CHAN_RADIUS} {PROBE_RADIUS} {VOL_SAMPLES_TOTAL} '{vol_cyg}' '{cif_cyg}'",
        f"./network -ha -volpo {CHAN_RADIUS} {PROBE_RADIUS} {VOLPO_SAMPLES_TOTAL} '{volpo_cyg}' '{cif_cyg}'",
    ]
    return " && ".join(commands)


def run_shell_command(command):
    return subprocess.run(
        [CYGWIN_BASH, "-lc", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=TIMEOUT_SECONDS,
    )


def run_one_cif(cif_path):
    cif_path = str(Path(cif_path).resolve())
    material = Path(cif_path).stem

    res_file = str(Path(RES_DIR) / f"{material}.res")
    sa_file = str(Path(SA_DIR) / f"{material}.sa")
    vol_file = str(Path(VOL_DIR) / f"{material}.vol")
    volpo_file = str(Path(VOLPO_DIR) / f"{material}.volpo")
    log_file = str(Path(LOG_DIR) / f"{material}.log")

    result_dict = {
        "Material": material,
        "CIF_File": cif_path,
        "PLD": None,
        "LCD": None,
        "Density": None,
        "status": "FAILED",
        "returncode": None,
        "runtime_s": None,
        "error": "",
        "RES_File": res_file,
        "SA_File": sa_file,
        "VOL_File": vol_file,
        "VOLPO_File": volpo_file,
        "LOG_File": log_file,
    }
    start = time.time()

    def read_results():
        pore_data = parse_res_file(res_file)
        density = parse_density_file(sa_file)
        if density is None:
            density = parse_density_file(vol_file)
        if density is None:
            density = parse_density_file(volpo_file)
        return {
            "PLD": pore_data["PLD"],
            "LCD": pore_data["LCD"],
            "Density": density,
        }

    try:
        if not os.path.exists(CYGWIN_BASH):
            raise FileNotFoundError(f"Missing Cygwin bash: {CYGWIN_BASH}")
        if not os.path.exists(cif_path):
            raise FileNotFoundError(f"Missing CIF file: {cif_path}")

        outputs = [res_file, sa_file, vol_file, volpo_file]
        if SKIP_IF_DONE and all(os.path.exists(path) for path in outputs):
            result_dict.update(read_results())
            result_dict["status"] = "SKIPPED"
            result_dict["runtime_s"] = time.time() - start
            return result_dict

        command = build_cmd_str(cif_path, res_file, sa_file, vol_file, volpo_file)
        process = run_shell_command(command)
        result_dict["returncode"] = process.returncode

        with open(log_file, "w", encoding="utf-8", errors="ignore") as stream:
            stream.write("COMMAND:\n")
            stream.write(f"{CYGWIN_BASH} -lc {command}\n\n")
            stream.write("RETURNCODE:\n")
            stream.write(f"{process.returncode}\n\n")
            stream.write("STDOUT:\n")
            stream.write(process.stdout or "")
            stream.write("\n\nSTDERR:\n")
            stream.write(process.stderr or "")

        result_dict.update(read_results())
        has_any = any(
            result_dict[column] is not None
            for column in ["PLD", "LCD", "Density"]
        )
        if process.returncode == 0 and has_any:
            result_dict["status"] = "OK"
        elif has_any:
            result_dict["status"] = "PARTIAL"
            result_dict["error"] = "Nonzero return code; partial output was parsed."
        else:
            result_dict["status"] = "FAILED"
            result_dict["error"] = "No parsable output generated."

    except subprocess.TimeoutExpired as exc:
        result_dict["status"] = "FAILED"
        result_dict["error"] = f"Timeout exceeded: {TIMEOUT_SECONDS} seconds"
        with open(log_file, "w", encoding="utf-8", errors="ignore") as stream:
            stream.write("COMMAND:\n")
            stream.write(
                f"{CYGWIN_BASH} -lc "
                f"{build_cmd_str(cif_path, res_file, sa_file, vol_file, volpo_file)}\n\n"
            )
            stream.write("EXCEPTION:\n")
            stream.write(repr(exc))
    except Exception as exc:
        result_dict["status"] = "FAILED"
        result_dict["error"] = f"{type(exc).__name__}: {exc}"
        with open(log_file, "w", encoding="utf-8", errors="ignore") as stream:
            stream.write("EXCEPTION:\n")
            stream.write(traceback.format_exc())

    result_dict["runtime_s"] = time.time() - start
    return result_dict


def main():
    if not os.path.exists(CYGWIN_BASH):
        raise FileNotFoundError(f"Missing Cygwin bash: {CYGWIN_BASH}")
    if not os.path.exists(ZEO_DIR_WIN):
        raise FileNotFoundError(f"Missing Zeo++ directory: {ZEO_DIR_WIN}")
    if not os.path.exists(CIF_DIR):
        raise FileNotFoundError(f"Missing CIF directory: {CIF_DIR}")

    for directory in [RAW_ROOT, RES_DIR, SA_DIR, VOL_DIR, VOLPO_DIR, LOG_DIR]:
        ensure_dir(directory)

    cif_files = sorted(str(path) for path in Path(CIF_DIR).glob("*.cif"))
    if TEST_MODE:
        cif_files = cif_files[:TEST_N]

    total = len(cif_files)
    if total == 0:
        raise RuntimeError(f"No CIF files found in {CIF_DIR}")
    workers = max(1, min(N_WORKERS, total))

    print(f"CIF files: {total}; workers: {workers}")
    print(f"Probe radius: {PROBE_RADIUS}; channel radius: {CHAN_RADIUS}")
    print(f"Test mode: {TEST_MODE}; output: {OUTPUT_XLSX}")

    start_time = time.time()
    results = []
    counts = {"OK": 0, "PARTIAL": 0, "FAILED": 0, "SKIPPED": 0}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one_cif, path): path for path in cif_files}
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            results.append(row)
            counts[row["status"]] += 1

            elapsed = time.time() - start_time
            rate = index / elapsed if elapsed > 0 else 0
            remaining = (total - index) / rate if rate > 0 else math.inf
            time_left = "unknown" if math.isinf(remaining) else f"{int(remaining)}s"
            print(
                f"[{index}/{total}] remaining={time_left}; "
                f"OK={counts['OK']} PARTIAL={counts['PARTIAL']} "
                f"SKIPPED={counts['SKIPPED']} FAILED={counts['FAILED']}"
            )

    df = pd.DataFrame(results)
    preferred_cols = [
        "Material",
        "CIF_File",
        "PLD",
        "LCD",
        "Density",
        "status",
        "returncode",
        "runtime_s",
        "error",
        "RES_File",
        "SA_File",
        "VOL_File",
        "VOLPO_File",
        "LOG_File",
    ]
    df = df[[col for col in preferred_cols if col in df.columns]]
    df.sort_values(by="Material", inplace=True)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="features")
        df[df["status"].isin(["FAILED", "PARTIAL"])].to_excel(
            writer, index=False, sheet_name="failures"
        )

    print(f"Completed in {int(time.time() - start_time)} seconds")
    print(f"Status counts: {counts}")
    print(f"Output: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
