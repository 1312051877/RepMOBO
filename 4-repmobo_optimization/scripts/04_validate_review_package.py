#!/usr/bin/env python3
"""Validate 105 configurations and the eight saved input files; write nothing."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
SEEDS = {42, 123, 456, 789, 1024}
FEATURES = {"Manual": 14, "Hybrid_PCA80_41D": 41, "Hybrid_PCA85_58D": 58,
            "Hybrid_PCA90_86D": 86, "Hybrid_PCA95_145D": 145}
PARAMETERS = {"candidate_screen_size": 2048, "qehvi_mc_samples": 64,
              "acq_chunk_size": 256, "max_fit_iter": 75}
OBJECTIVES = ["CEES_true", "DMMP_true", "EtCl_true"]

def require(condition, message):
    if not condition:
        raise ValueError(message)

def load_configs():
    paths = sorted((ROOT / "configs").glob("*.json"))
    require(len(paths) == 105, "Expected 105 JSON files directly in configs/")
    configs = [(p, json.loads(p.read_text(encoding="utf-8-sig"))) for p in paths]
    expected = set()
    for feature in FEATURES:
        for ard in [True, False]:
            for seed in SEEDS:
                expected.add(("Greedy", feature, ard, seed))
                expected.add(("Manual_AL" if feature == "Manual" else "ReMOBO", feature, ard, seed))
    expected.update(("Random", "Manual", False, seed) for seed in SEEDS)
    actual, ids = [], []
    for path, cfg in configs:
        tid = cfg["task_id"]
        require(path.stem == tid, f"Filename/task_id mismatch: {path.name}")
        ids.append(tid)
        require(type(cfg["ARD"]) is bool, f"ARD must be boolean: {tid}")
        actual.append((cfg["method"], cfg["feature_type"], cfg["ARD"], cfg["seed"]))
        require(cfg["feature_type"] in FEATURES, f"Unknown feature space: {tid}")
        require(cfg["latent_dim"] == FEATURES[cfg["feature_type"]], f"Wrong dimension: {tid}")
        require(cfg["initial_size"] == 100, f"Wrong initial size: {tid}")
        require(cfg["objectives"] == OBJECTIVES, f"Wrong objectives: {tid}")
        for key, value in PARAMETERS.items():
            require(type(cfg.get(key)) is int and cfg[key] == value, f"Wrong {key}: {tid}")
        require(cfg["task_output_dir"] == f"results/tasks/{tid}", f"Wrong output directory: {tid}")
        paths = {"full_library_master_file": "data/full_library_master.csv",
                 "initial_design_file": "data/initial_designs_checked.csv",
                 "oracle_reference_file": "data/oracle_reference.json",
                 "latent_coordinate_file": f"data/latent_coordinates/{cfg['feature_type']}_latent.csv"}
        for key, value in paths.items():
            require(cfg[key] == value and (ROOT / value).is_file(), f"Missing/mismatched {key}: {tid}")
    require(len(set(ids)) == 105, "Duplicate task_id")
    require(len(set(actual)) == 105 and set(actual) == expected, "Expected 21 configurations x 5 seeds")
    return configs

def validate():
    load_configs()
    data = ROOT / "data"
    full = pd.read_csv(data / "full_library_master.csv", dtype={"mof_id": str})
    initial = pd.read_csv(data / "initial_designs_checked.csv", dtype={"mof_id": str})
    oracle = json.loads((data / "oracle_reference.json").read_text(encoding="utf-8-sig"))
    require(len(full) == full.mof_id.nunique() == 7532, "Expected 7532 unique MOFs")
    require(full.mof_id.notna().all() and full.mof_id.str.strip().ne("").all(), "Missing MOF IDs")
    require(full.mof_index.tolist() == list(range(7532)), "mof_index must match row order")
    flags = ["is_oracle_front", "is_top3_layer", "is_top1_composite"]
    flags += [f"is_top{n}_{gas}" for gas in ["CEES", "DMMP", "EtCl"] for n in [1]]
    require(set(OBJECTIVES + flags + ["oracle_pareto_layer"]) <= set(full), "Missing labels/metric flags")
    require(np.isfinite(full[OBJECTIVES].to_numpy(float)).all(), "Invalid labels")
    require(np.isfinite(full.oracle_pareto_layer.to_numpy(float)).all(), "Invalid Pareto layers")
    for column in flags:
        require(full[column].notna().all() and full[column].isin([True, False]).all(), f"Invalid flag: {column}")
    require(set(oracle) == {"oracle_reference_id", "reference_point_id", "reference_point", "oracle_HV"}, "Unexpected oracle fields")
    require(set(oracle["reference_point"]) == set(OBJECTIVES), "Oracle objectives mismatch")
    require(np.isfinite([oracle["reference_point"][n] for n in OBJECTIVES]).all(), "Invalid reference point")
    require(np.isfinite(oracle["oracle_HV"]) and oracle["oracle_HV"] > 0, "Invalid oracle HV")
    require(len(list((data / "latent_coordinates").glob("*.csv"))) == 5, "Expected five feature CSVs")
    for feature, dim in FEATURES.items():
        latent = pd.read_csv(data / "latent_coordinates" / f"{feature}_latent.csv", dtype={"mof_id": str})
        columns = [c for c in latent if c.startswith("latent_")]
        require(columns == [f"latent_{i}" for i in range(1, dim + 1)], f"Wrong feature dimensions/order: {feature}")
        require(latent.mof_id.tolist() == full.mof_id.tolist(), f"ID/order mismatch: {feature}")
        require(latent.mof_index.tolist() == full.mof_index.tolist(), f"Index mismatch: {feature}")
        require(np.isfinite(latent[columns + ["PCA2_x", "PCA2_y"]].to_numpy(float)).all(), f"Invalid features: {feature}")
    require(list(initial) == ["seed", "init_rank", "mof_index", "mof_id"], "Unexpected initial-design fields")
    require(len(initial) == 500 and set(initial.seed) == SEEDS, "Expected five initial designs")
    require(not initial.isna().any().any(), "Missing initial values")
    for seed, group in initial.groupby("seed"):
        require(len(group) == group.mof_id.nunique() == 100, f"Initial design must have 100 unique MOFs: {seed}")
        require(set(group.init_rank) == set(range(1, 101)), f"Invalid initial ranks: {seed}")
        index = group.mof_index.to_numpy()
        require(np.equal(index, index.astype(int)).all() and ((index >= 0) & (index < 7532)).all(), f"Invalid initial indices: {seed}")
        require(full.iloc[index.astype(int)].mof_id.tolist() == group.mof_id.tolist(), f"Initial ID/index mismatch: {seed}")
    print("INPUT VALIDATION PASSED")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    validate()
