import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ("CEES", "DMMP", "EtCl")


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"mof_id": "string"})
    if "EtCl" not in data.columns and "EtCL" in data.columns:
        data = data.rename(columns={"EtCL": "EtCl"})
    required = ("mof_id", *TARGETS)
    missing = [name for name in required if name not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if data.empty:
        raise ValueError("The input dataset is empty.")
    if data["mof_id"].isna().any() or data["mof_id"].str.strip().eq("").any():
        raise ValueError("Missing mof_id values were found.")
    if data["mof_id"].duplicated().any():
        raise ValueError("Duplicate mof_id values were found.")

    data = data.loc[:, required].copy()
    for target in TARGETS:
        data[target] = pd.to_numeric(data[target], errors="raise")
    if not np.isfinite(data[list(TARGETS)].to_numpy(dtype=float)).all():
        raise ValueError("The adsorption labels must be finite numbers.")
    return data


def get_pareto_fronts(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples = len(scores)
    dominated_indices = [None] * n_samples
    dominated_by_count = np.zeros(n_samples, dtype=np.int32)
    dominates_count = np.zeros(n_samples, dtype=np.int32)

    for i in range(n_samples):
        diff = scores[i] - scores
        dominates = (diff >= 0).all(axis=1) & (diff > 0).any(axis=1)
        dominated_by = (diff <= 0).all(axis=1) & (diff < 0).any(axis=1)
        dominated_indices[i] = np.flatnonzero(dominates).astype(np.int32)
        dominates_count[i] = len(dominated_indices[i])
        dominated_by_count[i] = int(dominated_by.sum())

    remaining = dominated_by_count.copy()
    ranks = np.zeros(n_samples, dtype=np.int32)
    front = np.flatnonzero(remaining == 0)
    layer = 1

    while len(front):
        ranks[front] = layer
        next_front = []
        for p in front:
            for q in dominated_indices[p]:
                remaining[q] -= 1
                if remaining[q] == 0:
                    next_front.append(q)
        front = np.asarray(next_front, dtype=np.int32)
        layer += 1

    if (ranks == 0).any():
        raise RuntimeError("Pareto ranking did not cover all materials.")
    return ranks, dominates_count, dominated_by_count


def analyze(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_samples = len(data)
    top_count = int(np.ceil(n_samples * 0.01))
    membership = data.copy()
    statistics = [{"Metric": "Dataset_Size", "Target_1": "", "Target_2": "", "Value": n_samples}]

    for target in TARGETS:
        selected = data[target].sort_values(ascending=False, kind="mergesort").index[:top_count]
        membership[f"Top1pct_{target}"] = membership.index.isin(selected).astype(int)
        statistics.extend([
            {"Metric": "Top1pct_Count", "Target_1": target, "Target_2": "", "Value": top_count},
            {"Metric": "Percentile_99_Threshold", "Target_1": target, "Target_2": "", "Value": data[target].quantile(0.99)},
        ])

    spearman = data[list(TARGETS)].corr(method="spearman")
    for first, second in combinations(TARGETS, 2):
        statistics.append({
            "Metric": "Spearman", "Target_1": first,
            "Target_2": second, "Value": spearman.loc[first, second],
        })

    scores = data[list(TARGETS)].to_numpy(dtype=float)
    ranks, dominates_count, dominated_by_count = get_pareto_fronts(scores)
    membership["Dominates_Count"] = dominates_count
    membership["Dominated_By_Count"] = dominated_by_count
    membership["Pareto_Layer"] = ranks
    membership["Pareto_Optimal"] = (ranks == 1).astype(int)

    for layer in (1, 2, 3):
        statistics.append({
            "Metric": "Pareto_Layer_Count", "Target_1": str(layer),
            "Target_2": "", "Value": int((ranks == layer).sum()),
        })
    statistics.append({
        "Metric": "Pareto_First3_Count", "Target_1": "",
        "Target_2": "", "Value": int((ranks <= 3).sum()),
    })

    return membership, pd.DataFrame(statistics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("Manual.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("expected_outputs"))
    args = parser.parse_args()

    membership, statistics = analyze(load_data(args.input))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    membership.to_csv(args.output_dir / "oracle_membership.csv", index=False)
    statistics.to_csv(args.output_dir / "oracle_statistics.csv", index=False)
    print(f"Processed {len(membership)} materials.")
    print(f"Saved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
