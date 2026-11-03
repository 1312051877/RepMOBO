#!/usr/bin/env python3
"""List 105 configs; execute sequentially only with --execute."""
import argparse
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Actually run all 105 tasks")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:N")
    args = parser.parse_args(argv)
    if args.device not in ["auto", "cpu"] and not re.fullmatch(r"cuda:\d+", args.device):
        parser.error("--device must be auto, cpu or cuda:N")
    return args

def main():
    args = parse_args()
    spec = importlib.util.spec_from_file_location("input_validation", ROOT / "scripts/04_validate_review_package.py")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    configs = validator.load_configs()
    if args.execute:
        validator.validate()
        existing = [c["task_id"] for _, c in configs if (ROOT / c["task_output_dir"]).exists()]
        if existing:
            raise FileExistsError("Refusing to overwrite existing tasks: " + ", ".join(existing))
    for path, cfg in configs:
        relative = path.relative_to(ROOT).as_posix()
        if args.execute:
            subprocess.run([sys.executable, "-B", str(ROOT / "scripts/03_run_al_task.py"),
                            "--config", relative, "--device", args.device], cwd=ROOT, check=True)
        else:
            print(relative)
    if not args.execute:
        print("105 TASKS LISTED; NO OPTIMIZATION STARTED")

if __name__ == "__main__":
    main()
