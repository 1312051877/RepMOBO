#!/usr/bin/env python3
"""Run one RepMOBO task. Successful outputs: selection events, metrics, summary."""
from __future__ import annotations
import argparse
import ctypes
import gc
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def allowed_cpu_count():
    """Conservative process CPU limit from affinity, scheduler and cgroup quota."""
    limits = [os.cpu_count() or 1]
    try:
        limits.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        if os.name == "nt":
            try:
                kernel = ctypes.windll.kernel32
                kernel.GetCurrentProcess.restype = ctypes.c_void_p
                mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
                kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
                if kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.byref(mask), ctypes.byref(system_mask)):
                    limits.append(bin(mask.value).count("1"))
            except (AttributeError, OSError):
                pass
    for name in ["SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE",
                 "PBS_NCPUS", "NCPUS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        match = re.match(r"^(\d+)", os.environ.get(name, ""))
        if match and int(match.group(1)) > 0:
            limits.append(int(match.group(1)))
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            limits.append(max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        try:
            base = Path("/sys/fs/cgroup/cpu")
            quota = int((base / "cpu.cfs_quota_us").read_text())
            period = int((base / "cpu.cfs_period_us").read_text())
            if quota > 0:
                limits.append(max(1, quota // period))
        except (OSError, ValueError):
            pass
    return max(1, min(value for value in limits if value > 0))


CPU_THREADS = allowed_cpu_count()
for _name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_name] = str(CPU_THREADS)
import numpy as np
import pandas as pd

TORCH_IMPORT_ERROR = None
try:
    import torch
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(1)
except ImportError as exc:
    torch = None
    TORCH_IMPORT_ERROR = exc
GP_IMPORT_ERROR = None
try:
    import gpytorch
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.mlls import SumMarginalLogLikelihood
    from botorch.models import SingleTaskGP, ModelListGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
    from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
    try:
        from botorch.sampling.normal import SobolQMCNormalSampler
    except ImportError:
        from botorch.sampling.samplers import SobolQMCNormalSampler
except ImportError as exc:
    GP_IMPORT_ERROR = exc

ROOT = Path(__file__).resolve().parents[1]
OBJECTIVES = ["CEES_true", "DMMP_true", "EtCl_true"]
SHORT_OBJECTIVES = ["CEES", "DMMP", "EtCl"]
OOM_EXIT_CODE = 86

def now_str() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel_or_abs(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return ROOT / p


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pareto_front_mask_max(Y: np.ndarray) -> np.ndarray:
    n = Y.shape[0]
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        ge = np.all(Y >= Y[i], axis=1)
        gt = np.any(Y > Y[i], axis=1)
        if np.any(ge & gt):
            dominated[i] = True
    return ~dominated


def hv2d_max(points_yz: np.ndarray, ref_yz: np.ndarray) -> float:
    pts = np.asarray(points_yz, dtype=float)
    ref = np.asarray(ref_yz, dtype=float)
    pts = pts[(pts[:, 0] > ref[0]) & (pts[:, 1] > ref[1])]
    if pts.size == 0:
        return 0.0
    ys = np.unique(pts[:, 0])
    bounds = np.concatenate([[ref[0]], np.sort(ys)])
    area = 0.0
    for low, high in zip(bounds[:-1], bounds[1:]):
        if high <= low:
            continue
        active = pts[pts[:, 0] >= high]
        if active.size == 0:
            continue
        max_z = np.max(active[:, 1])
        if max_z > ref[1]:
            area += (high - low) * (max_z - ref[1])
    return float(area)


def hv3d_max(points_xyz: np.ndarray, ref_xyz: np.ndarray) -> float:
    pts = np.asarray(points_xyz, dtype=float)
    ref = np.asarray(ref_xyz, dtype=float)
    pts = pts[np.all(pts > ref, axis=1)]
    if pts.size == 0:
        return 0.0
    xs = np.unique(pts[:, 0])
    bounds = np.concatenate([[ref[0]], np.sort(xs)])
    volume = 0.0
    for low, high in zip(bounds[:-1], bounds[1:]):
        if high <= low:
            continue
        active = pts[pts[:, 0] >= high]
        if active.size == 0:
            continue
        yz_area = hv2d_max(active[:, 1:3], ref[1:3])
        volume += (high - low) * yz_area
    return float(volume)


def mean_nearest_neighbor_distance(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n <= 1:
        return float("nan")
    # Chunked pairwise nearest neighbor to avoid large temporary arrays for selected_set MNND.
    best = np.full(n, np.inf, dtype=float)
    chunk = 512
    for i in range(0, n, chunk):
        Xi = X[i : i + chunk]
        d2 = np.sum((Xi[:, None, :] - X[None, :, :]) ** 2, axis=2)
        rows = np.arange(i, min(i + chunk, n))
        d2[np.arange(len(rows)), rows] = np.inf
        best[i : i + chunk] = np.sqrt(np.min(d2, axis=1))
    return float(np.mean(best))


def standardize_y(Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = Y.mean(axis=0)
    std = Y.std(axis=0, ddof=1)
    std[std < 1e-12] = 1.0
    return (Y - mean) / std, mean, std


def compute_metrics(
    cfg: dict,
    full_df: pd.DataFrame,
    Y_raw: np.ndarray,
    X_latent: np.ndarray,
    queried: Sequence[int],
    batch_indices: Sequence[int],
    round_id: int,
    n_before: int,
    n_after: int,
    oracle_ref: dict,
    timings: dict,
) -> dict:
    q = np.array(sorted(set(queried)), dtype=int)
    Yq = Y_raw[q]
    ref = np.array([oracle_ref["reference_point"][obj] for obj in OBJECTIVES], dtype=float)
    hv = hv3d_max(Yq, ref)
    oracle_hv = float(oracle_ref["oracle_HV"])

    is_front = full_df["is_oracle_front"].to_numpy(bool)
    is_top3 = full_df["is_top3_layer"].to_numpy(bool)

    row = {
        "task_id": cfg["task_id"],
        "method": cfg["method"],
        "seed": int(cfg["seed"]),
        "round": int(round_id),
        "n_evaluated_before": int(n_before),
        "n_evaluated_after": int(n_after),
        "n_evaluated": int(len(q)),
        "batch_size_this_round": int(len(batch_indices)),
        "budget_fraction": float(len(q) / len(full_df)),
        "oracle_reference_id": oracle_ref["oracle_reference_id"],
        "reference_point_id": oracle_ref["reference_point_id"],
        "HV": float(hv),
        "normalized_HV": float(hv / oracle_hv if oracle_hv > 0 else np.nan),
        "pareto_front_capture": float(is_front[q].sum() / max(1, int(is_front.sum()))),
        "top3_layer_capture": float(is_top3[q].sum() / max(1, int(is_top3.sum()))),
    }

    for short in SHORT_OBJECTIVES:
        mask1 = full_df[f"is_top1_{short}"].to_numpy(bool)
        mask5 = full_df[f"is_top5_{short}"].to_numpy(bool)
        row[f"top1_{short}_hit"] = float(mask1[q].sum() / max(1, int(mask1.sum())))
        row[f"top5_{short}_hit"] = float(mask5[q].sum() / max(1, int(mask5.sum())))
        row[f"n_top5_{short}_found"] = int(mask5[q].sum())

    row["top1_mean_hit"] = float(np.mean([row[f"top1_{s}_hit"] for s in SHORT_OBJECTIVES]))
    row["top5_mean_hit"] = float(np.mean([row[f"top5_{s}_hit"] for s in SHORT_OBJECTIVES]))

    c1 = full_df["is_top1_composite"].to_numpy(bool)
    c5 = full_df["is_top5_composite"].to_numpy(bool)
    row["top1_composite_hit"] = float(c1[q].sum() / max(1, int(c1.sum())))
    row["top5_composite_hit"] = float(c5[q].sum() / max(1, int(c5.sum())))

    b = np.array(list(batch_indices), dtype=int)
    row["batch_MNND"] = mean_nearest_neighbor_distance(X_latent[b]) if len(b) > 1 else float("nan")
    row["selected_set_MNND"] = mean_nearest_neighbor_distance(X_latent[q]) if len(q) > 1 else float("nan")

    row["n_oracle_front_found"] = int(is_front[q].sum())
    row["n_top3_layer_found"] = int(is_top3[q].sum())

    row["fit_time_seconds"] = float(timings.get("fit", 0.0))
    row["acquisition_time_seconds"] = float(timings.get("acquisition", 0.0))
    row["metric_time_seconds"] = float(timings.get("metric", 0.0))
    row["walltime_round"] = float(timings.get("round", 0.0))
    row["walltime_total"] = float(timings.get("total", 0.0))
    row["status"] = "ok"
    row["error_message"] = ""
    return row


def make_selection_rows(
    cfg: dict,
    full_df: pd.DataFrame,
    latent_df: pd.DataFrame,
    indices: Sequence[int],
    is_initial: bool,
    selected_round: int,
    selection_type: str,
    n_before: int,
    n_after: int,
    rank_offset: int = 0,
    selection_scores: Optional[Dict[int, float]] = None,
    pred_mean_raw: Optional[np.ndarray] = None,
    pred_std_raw: Optional[np.ndarray] = None,
) -> List[dict]:
    rows = []
    selection_scores = selection_scores or {}
    for j, idx in enumerate(indices, start=1):
        src = full_df.iloc[int(idx)]
        lat = latent_df.iloc[int(idx)]
        row = {
            "task_id": cfg["task_id"],
            "method": cfg["method"],
            "seed": int(cfg["seed"]),
            "is_initial": bool(is_initial),
            "selected_round": int(selected_round),
            "selection_type": selection_type,
            "round": int(selected_round),
            "n_evaluated_before": int(n_before),
            "n_evaluated_after": int(n_after),
            "batch_size_this_round": int(len(indices)),
            "selection_rank_in_batch": int(rank_offset + j),
            "mof_index": int(src["mof_index"]),
            "mof_id": str(src["mof_id"]),
            "CEES_true": float(src["CEES_true"]),
            "DMMP_true": float(src["DMMP_true"]),
            "EtCl_true": float(src["EtCl_true"]),
            "selection_score": selection_scores.get(int(idx), np.nan),
            "selection_score_type": cfg.get("selection_score_type", selection_type),
            "CEES_pred_mean_at_selection": np.nan,
            "DMMP_pred_mean_at_selection": np.nan,
            "EtCl_pred_mean_at_selection": np.nan,
            "CEES_pred_std_at_selection": np.nan,
            "DMMP_pred_std_at_selection": np.nan,
            "EtCl_pred_std_at_selection": np.nan,
            "oracle_pareto_layer": int(src["oracle_pareto_layer"]),
            "is_oracle_front": bool(src["is_oracle_front"]),
            "is_top3_layer": bool(src["is_top3_layer"]),
            "is_top1_CEES": bool(src["is_top1_CEES"]),
            "is_top1_DMMP": bool(src["is_top1_DMMP"]),
            "is_top1_EtCl": bool(src["is_top1_EtCl"]),
            "is_top1_composite": bool(src["is_top1_composite"]),
            "latent_x": float(lat["PCA2_x"]),
            "latent_y": float(lat["PCA2_y"]),
        }
        if pred_mean_raw is not None:
            row["CEES_pred_mean_at_selection"] = float(pred_mean_raw[int(idx), 0])
            row["DMMP_pred_mean_at_selection"] = float(pred_mean_raw[int(idx), 1])
            row["EtCl_pred_mean_at_selection"] = float(pred_mean_raw[int(idx), 2])
        if pred_std_raw is not None:
            row["CEES_pred_std_at_selection"] = float(pred_std_raw[int(idx), 0])
            row["DMMP_pred_std_at_selection"] = float(pred_std_raw[int(idx), 1])
            row["EtCl_pred_std_at_selection"] = float(pred_std_raw[int(idx), 2])
        rows.append(row)
    return rows


def append_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=header, encoding="utf-8-sig")


def build_model(
    train_X_np: np.ndarray,
    train_Y_raw_np: np.ndarray,
    ard: bool,
    device: torch.device,
    dtype: torch.dtype,
    max_fit_iter: int,
) -> Tuple[ModelListGP, np.ndarray, np.ndarray, np.ndarray]:
    Y_std_np, y_mean, y_std = standardize_y(train_Y_raw_np)
    train_X = torch.as_tensor(train_X_np, device=device, dtype=dtype)
    train_Y = torch.as_tensor(Y_std_np, device=device, dtype=dtype)
    d = train_X.shape[-1]

    models = []
    for j in range(train_Y.shape[1]):
        ard_dims = int(d) if ard else None
        covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=ard_dims))
        model_j = SingleTaskGP(train_X, train_Y[:, j : j + 1], covar_module=covar_module)
        models.append(model_j)

    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)

    fit_gpytorch_mll(mll, options={"maxiter": int(max_fit_iter), "disp": False})
    model.eval()
    return model, Y_std_np, y_mean, y_std


def predict_all_raw(
    model: ModelListGP,
    X_np: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    means = []
    stds = []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(0, X_np.shape[0], chunk_size):
            X = torch.as_tensor(X_np[i : i + chunk_size], device=device, dtype=dtype)
            posterior = model.posterior(X)
            mean_std = posterior.mean.detach().cpu().numpy()
            var_std = posterior.variance.clamp_min(1e-12).detach().cpu().numpy()
            means.append(mean_std * y_std.reshape(1, -1) + y_mean.reshape(1, -1))
            stds.append(np.sqrt(var_std) * y_std.reshape(1, -1))
    return np.vstack(means), np.vstack(stds)


def make_sampler(num_samples: int, seed: int):
    try:
        return SobolQMCNormalSampler(sample_shape=torch.Size([int(num_samples)]), seed=int(seed))
    except TypeError:  # Older BoTorch fallback.
        return SobolQMCNormalSampler(num_samples=int(num_samples), seed=int(seed))


def qehvi_scores_for_candidates(
    model: ModelListGP,
    train_Y_std_np: np.ndarray,
    ref_std_np: np.ndarray,
    X_np: np.ndarray,
    candidate_indices: np.ndarray,
    pending_indices: List[int],
    device: torch.device,
    dtype: torch.dtype,
    mc_samples: int,
    acq_chunk_size: int,
    seed: int,
) -> np.ndarray:
    train_Y_std = torch.as_tensor(train_Y_std_np, device=device, dtype=dtype)
    ref_std = torch.as_tensor(ref_std_np, device=device, dtype=dtype)
    partitioning = NondominatedPartitioning(ref_point=ref_std, Y=train_Y_std)
    sampler = make_sampler(mc_samples, seed)
    acq = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_std.tolist(),
        partitioning=partitioning,
        sampler=sampler,
    )

    cand = np.asarray(candidate_indices, dtype=int)
    pending = list(pending_indices)
    out = np.empty(len(cand), dtype=float)
    pending_tensor = None
    if pending:
        pending_tensor = torch.as_tensor(X_np[pending], device=device, dtype=dtype)

    with torch.no_grad():
        for start in range(0, len(cand), acq_chunk_size):
            sub = cand[start : start + acq_chunk_size]
            Xc = torch.as_tensor(X_np[sub], device=device, dtype=dtype)
            if pending_tensor is None:
                Xq = Xc.unsqueeze(1)  # batch x q=1 x d
            else:
                P = pending_tensor.unsqueeze(0).expand(Xc.shape[0], -1, -1)
                Xq = torch.cat([P, Xc.unsqueeze(1)], dim=1)
            vals = acq(Xq).detach().cpu().numpy().reshape(-1)
            out[start : start + len(sub)] = vals
            # Release large qEHVI temporaries between chunks to keep peak GPU memory low.
            del Xc, Xq
            if pending_tensor is not None:
                del P
            if device.type == "cuda":
                torch.cuda.empty_cache()
    del acq, sampler, partitioning, train_Y_std, ref_std
    if pending_tensor is not None:
        del pending_tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def select_random(rng: np.random.Generator, unqueried: np.ndarray, batch_size: int) -> Tuple[List[int], Dict[int, float]]:
    chosen = rng.choice(unqueried, size=batch_size, replace=False).astype(int).tolist()
    return chosen, {idx: np.nan for idx in chosen}


def select_greedy(pred_mean_raw: np.ndarray, unqueried: np.ndarray, batch_size: int) -> Tuple[List[int], Dict[int, float]]:
    # Greedy uses predicted composite score based on min-max normalized posterior mean over the pool.
    M = pred_mean_raw
    lo = M.min(axis=0)
    hi = M.max(axis=0)
    score = ((M - lo) / np.maximum(hi - lo, 1e-12)).mean(axis=1)
    sub_scores = score[unqueried]
    order = np.argsort(-sub_scores, kind="mergesort")[:batch_size]
    chosen = unqueried[order].astype(int).tolist()
    return chosen, {idx: float(score[idx]) for idx in chosen}


def select_qehvi_sequential(
    cfg: dict,
    model: ModelListGP,
    train_Y_std_np: np.ndarray,
    ref_raw_np: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    X_np: np.ndarray,
    pred_mean_raw: np.ndarray,
    pred_std_raw: np.ndarray,
    unqueried: np.ndarray,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    round_id: int,
) -> Tuple[List[int], Dict[int, float]]:
    # Standardized reference point for a model trained on standardized Y.
    ref_std_np = (ref_raw_np - y_mean) / y_std

    # Screen candidates to reduce qEHVI cost while preserving a broad high-value/high-uncertainty set.
    screen_size = int(cfg.get("candidate_screen_size", 2048))
    mc_samples = int(cfg.get("qehvi_mc_samples", 64))
    acq_chunk_size = int(cfg.get("acq_chunk_size", 256))

    chosen: List[int] = []
    scores: Dict[int, float] = {}
    available = np.asarray(unqueried, dtype=int).copy()

    # Pool-level screening score in raw objective space.
    M = pred_mean_raw
    S = pred_std_raw
    lo = M.min(axis=0)
    hi = M.max(axis=0)
    mean_score = ((M - lo) / np.maximum(hi - lo, 1e-12)).mean(axis=1)
    std_score = (S / np.maximum(S.max(axis=0, keepdims=True), 1e-12)).mean(axis=1)
    screening_score = mean_score + 0.25 * std_score

    for k in range(batch_size):
        if available.size == 0:
            break
        n_screen = min(screen_size, available.size)
        order = np.argsort(-screening_score[available], kind="mergesort")[:n_screen]
        cand = available[order]
        vals = qehvi_scores_for_candidates(
            model=model,
            train_Y_std_np=train_Y_std_np,
            ref_std_np=ref_std_np,
            X_np=X_np,
            candidate_indices=cand,
            pending_indices=chosen,
            device=device,
            dtype=dtype,
            mc_samples=mc_samples,
            acq_chunk_size=acq_chunk_size,
            seed=int(cfg["seed"]) + 1000 * int(round_id) + k,
        )
        best_local = int(np.argmax(vals))
        best_idx = int(cand[best_local])
        chosen.append(best_idx)
        scores[best_idx] = float(vals[best_local])
        available = available[available != best_idx]
    return chosen, scores


def load_inputs(cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    full_df = pd.read_csv(rel_or_abs(cfg["full_library_master_file"]))
    full_df["mof_id"] = full_df["mof_id"].astype(str)
    latent_df = pd.read_csv(rel_or_abs(cfg["latent_coordinate_file"]))
    latent_df["mof_id"] = latent_df["mof_id"].astype(str)
    init_df = pd.read_csv(rel_or_abs(cfg["initial_design_file"]))
    init_df["mof_id"] = init_df["mof_id"].astype(str)
    oracle_ref = load_json(rel_or_abs(cfg["oracle_reference_file"]))

    if not full_df["mof_id"].equals(latent_df["mof_id"]):
        raise ValueError("full_library_master and latent coordinates have different mof_id order")
    # Preserve the original Top-5 metrics when optional saved flags were omitted.
    # This is the original preparation rule, evaluated in memory only.
    Y = full_df[OBJECTIVES].to_numpy(dtype=float)
    minima, maxima = Y.min(axis=0), Y.max(axis=0)
    composite = ((Y - minima) / np.where(maxima > minima, maxima - minima, 1.0)).mean(axis=1)
    for name, values in list(zip(SHORT_OBJECTIVES, Y.T)) + [("composite", composite)]:
        column = f"is_top5_{name}"
        if column not in full_df:
            order = np.argsort(-values, kind="mergesort")
            mask = np.zeros(len(full_df), dtype=bool)
            mask[order[:int(math.ceil(len(full_df) * 5.0 / 100.0))]] = True
            full_df[column] = mask
    return full_df, latent_df, init_df, oracle_ref


def run_attempt(cfg, args, out_dir, device):
    seed_everything(int(cfg["seed"]))
    (full_df, latent_df, init_df, oracle_ref) = load_inputs(cfg)
    latent_cols = [c for c in latent_df.columns if c.startswith('latent_')]
    X_np = latent_df[latent_cols].to_numpy(dtype=np.float64)
    Y_raw = full_df[OBJECTIVES].to_numpy(dtype=np.float64)
    ref_raw = np.array([oracle_ref['reference_point'][obj] for obj in OBJECTIVES], dtype=np.float64)
    seed_init = init_df[init_df['seed'].astype(int) == int(cfg['seed'])].copy()
    if len(seed_init) != int(cfg['initial_size']):
        raise ValueError(f"Initial design size mismatch for seed {cfg['seed']}: {len(seed_init)}")
    initial_indices = seed_init.sort_values('init_rank')['mof_index'].astype(int).tolist()
    if len(set(initial_indices)) != len(initial_indices):
        raise ValueError('Duplicate initial indices')
    dtype = torch.double
    selection_path = out_dir / 'selection_events.csv'
    metrics_path = out_dir / 'round_metrics.csv'
    queried: List[int] = list(initial_indices)
    queried_set = set(queried)
    rng = np.random.default_rng(int(cfg['seed']))
    total_start = time.time()
    init_rows = make_selection_rows(cfg=cfg, full_df=full_df, latent_df=latent_df, indices=initial_indices, is_initial=True, selected_round=0, selection_type='initial_precomputed_from_Manual_Scaled', n_before=0, n_after=len(initial_indices))
    append_csv(selection_path, init_rows)
    pred_mean_raw = pred_std_raw = None
    fit_time = acq_time = 0.0
    if cfg.get('has_surrogate', False):
        t0 = time.time()
        (model, train_Y_std_np, y_mean, y_std) = build_model(X_np[queried], Y_raw[queried], bool(cfg['ARD']), device, dtype, int(cfg['max_fit_iter']))
        fit_time = time.time() - t0
        (pred_mean_raw, pred_std_raw) = predict_all_raw(model, X_np, y_mean, y_std, device, dtype, int(args.pred_chunk_size))
    t_metric0 = time.time()
    m0 = compute_metrics(cfg, full_df, Y_raw, X_np, queried, [], 0, 0, len(queried), oracle_ref, {'fit': fit_time, 'acquisition': acq_time, 'metric': 0.0, 'round': 0.0, 'total': time.time() - total_start})
    m0['metric_time_seconds'] = time.time() - t_metric0
    append_csv(metrics_path, [m0])
    budget_points = sorted([int(b) for b in cfg['budget_points'] if int(b) <= int(cfg['max_budget'])])
    max_budget = int(cfg['max_budget'])
    round_id = 0
    while len(queried) < max_budget:
        round_id += 1
        round_start = time.time()
        n_before = len(queried)
        next_milestones = [b for b in budget_points if b > n_before]
        next_target = min(max_budget, next_milestones[0] if next_milestones else max_budget)
        batch_size = min(int(cfg['batch_size']), next_target - n_before)
        if batch_size <= 0:
            break
        unqueried = np.array([i for i in range(len(full_df)) if i not in queried_set], dtype=int)
        fit_time = acq_time = 0.0
        selection_scores: Dict[int, float] = {}
        if cfg['method'] == 'Random':
            t_acq = time.time()
            (batch_indices, selection_scores) = select_random(rng, unqueried, batch_size)
            acq_time = time.time() - t_acq
            pred_mean_raw = pred_std_raw = None
        else:
            t_fit = time.time()
            (model, train_Y_std_np, y_mean, y_std) = build_model(X_np[queried], Y_raw[queried], bool(cfg['ARD']), device, dtype, int(cfg['max_fit_iter']))
            fit_time = time.time() - t_fit
            (pred_mean_raw, pred_std_raw) = predict_all_raw(model, X_np, y_mean, y_std, device, dtype, int(args.pred_chunk_size))
            t_acq = time.time()
            if cfg['method'] == 'Greedy':
                (batch_indices, selection_scores) = select_greedy(pred_mean_raw, unqueried, batch_size)
            else:
                (batch_indices, selection_scores) = select_qehvi_sequential(cfg, model, train_Y_std_np, ref_raw, y_mean, y_std, X_np, pred_mean_raw, pred_std_raw, unqueried, batch_size, device, dtype, round_id)
            acq_time = time.time() - t_acq
        for idx in batch_indices:
            queried.append(int(idx))
            queried_set.add(int(idx))
        n_after = len(queried)
        sel_rows = make_selection_rows(cfg=cfg, full_df=full_df, latent_df=latent_df, indices=batch_indices, is_initial=False, selected_round=round_id, selection_type='random' if cfg['method'] == 'Random' else cfg.get('selection_score_type', 'acquisition'), n_before=n_before, n_after=n_after, selection_scores=selection_scores, pred_mean_raw=pred_mean_raw, pred_std_raw=pred_std_raw)
        append_csv(selection_path, sel_rows)
        t_metric = time.time()
        metric_row = compute_metrics(cfg, full_df, Y_raw, X_np, queried, batch_indices, round_id, n_before, n_after, oracle_ref, {'fit': fit_time, 'acquisition': acq_time, 'metric': 0.0, 'round': time.time() - round_start, 'total': time.time() - total_start})
        metric_row['metric_time_seconds'] = time.time() - t_metric
        metric_row['walltime_round'] = time.time() - round_start
        metric_row['walltime_total'] = time.time() - total_start
        append_csv(metrics_path, [metric_row])
    metrics_df = pd.read_csv(metrics_path)
    last = metrics_df.iloc[-1].to_dict()
    summary = {'task_id': cfg['task_id'], 'status': 'ok', 'final_budget': int(len(queried)), 'final_HV': float(last['HV']), 'final_normalized_HV': float(last['normalized_HV']), 'final_pareto_capture': float(last['pareto_front_capture']), 'final_top3_capture': float(last['top3_layer_capture']), 'final_top5_mean': float(last['top5_mean_hit']), 'final_batch_MNND': float(last['batch_MNND']) if not pd.isna(last['batch_MNND']) else None, 'total_runtime': float(time.time() - total_start), 'error_message': '', 'end_time': now_str()}
    summary["device"] = str(device)
    summary["cpu_threads"] = CPU_THREADS
    summary["parameters"] = cfg
    dump_json(summary, out_dir / "final_summary.json")
    return summary


def choose_device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda:0" if requested == "auto" else requested)
    return torch.device("cpu")


def is_accelerator_oom(error):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        message = str(error).lower()
        kind = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", ())
        if (kind and isinstance(error, kind)) or any(token in message for token in [
            "cuda out of memory", "hip out of memory", "cuda error: out of memory",
            "hip error: out of memory", "cublas_status_alloc_failed", "hipblas_status_alloc_failed",
        ]):
            return True
        error = error.__cause__ or error.__context__
    return False


def load_task_config(args):
    path = rel_or_abs(args.config).resolve()
    if path.parent != (ROOT / "configs").resolve() or path.suffix.lower() != ".json":
        raise ValueError("Choose a JSON directly inside configs/")
    cfg = load_json(path)
    task_id = cfg["task_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", task_id):
        raise ValueError("Invalid task_id")
    if cfg["task_output_dir"] != f"results/tasks/{task_id}":
        raise ValueError("task_output_dir must be results/tasks/<task_id>")
    if cfg["objectives"] != OBJECTIVES:
        raise ValueError("Expected CEES_true, DMMP_true, EtCl_true")
    if cfg["method"] not in ["Random", "Greedy", "Manual_AL", "ReMOBO"]:
        raise ValueError("Unknown acquisition method")
    if args.max_budget_override is not None:
        if not int(cfg["initial_size"]) <= args.max_budget_override <= int(cfg["max_budget"]):
            raise ValueError("Test budget must lie between initial_size and max_budget")
        cfg["max_budget"] = args.max_budget_override
        cfg["budget_points"] = [b for b in cfg["budget_points"] if b <= args.max_budget_override]
    for key in ["full_library_master_file", "latent_coordinate_file", "initial_design_file", "oracle_reference_file"]:
        path = rel_or_abs(cfg[key]).resolve()
        if (ROOT / "data").resolve() not in path.parents or not path.is_file():
            raise ValueError(f"Missing or out-of-package input: {key}")
    return cfg


def run_with_fallback(cfg, args):
    if TORCH_IMPORT_ERROR is not None:
        raise RuntimeError("PyTorch is required for execution") from TORCH_IMPORT_ERROR
    if cfg.get("has_surrogate", False) and GP_IMPORT_ERROR is not None:
        raise RuntimeError("GPyTorch and BoTorch are required for surrogate tasks") from GP_IMPORT_ERROR
    device = choose_device(args.device)
    task_root = (ROOT / "results/tasks").resolve()
    if ROOT.resolve() not in task_root.parents:
        raise ValueError("Result directory resolves outside the package")
    target = task_root / cfg["task_id"]
    if target.exists():
        raise FileExistsError("Task output already exists; refusing to overwrite")
    task_root.mkdir(parents=True, exist_ok=True)
    devices = [str(device)] + (["cpu"] if device.type == "cuda" else [])
    for index, name in enumerate(devices):
        # A fresh process releases all failed accelerator state and resets RNGs.
        with tempfile.TemporaryDirectory(prefix=f".{cfg['task_id']}_", dir=task_root) as temp:
            attempt = Path(temp).resolve()
            if attempt.parent != task_root or target.parent != task_root:
                raise ValueError("Attempt/output path escaped results/tasks")
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--config", args.config,
                       "--device", name, "--pred-chunk-size", str(args.pred_chunk_size),
                       "--_worker", "--_attempt-dir", str(attempt)]
            if args.max_budget_override is not None:
                command += ["--max-budget-override", str(args.max_budget_override)]
            env = os.environ.copy()
            if name == "cpu":
                for variable in ["CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"]:
                    env[variable] = ""
            completed = subprocess.run(command, cwd=ROOT, env=env)
            if completed.returncode == 0:
                expected = {"selection_events.csv", "round_metrics.csv", "final_summary.json"}
                if {p.name for p in attempt.iterdir()} != expected:
                    raise RuntimeError("Worker did not produce the three required outputs")
                summary = load_json(attempt / "final_summary.json")
                if summary.get("status") != "ok" or summary.get("final_budget") != cfg["max_budget"]:
                    raise RuntimeError("Worker did not complete the requested budget")
                summary["cpu_fallback"] = index > 0
                dump_json(summary, attempt / "final_summary.json")
                if target.exists():
                    raise FileExistsError("Output appeared during execution; refusing to overwrite")
                attempt.rename(target)
                print(json.dumps(summary, indent=2))
                return summary
            if completed.returncode == OOM_EXIT_CODE and name.startswith("cuda"):
                print("Accelerator memory exhausted; restarting the entire task on CPU with the same seed and initial design.", flush=True)
                continue
            raise RuntimeError(f"Task failed on {name} (exit code {completed.returncode})")
    raise RuntimeError("No execution attempt succeeded")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="configs/<task_id>.json")
    parser.add_argument("--device", default="auto", help="auto (accelerator then CPU), cpu, or cuda:N")
    parser.add_argument("--max-budget-override", type=int, default=None, help="Optional small test budget; does not edit the config")
    parser.add_argument("--pred-chunk-size", type=int, default=2048)
    parser.add_argument("--check-only", action="store_true", help="Check this config/input interface without running optimization")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_attempt-dir", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.device not in ["auto", "cpu"] and not re.fullmatch(r"cuda:\d+", args.device):
        parser.error("--device must be auto, cpu or cuda:N")
    if args.pred_chunk_size <= 0:
        parser.error("--pred-chunk-size must be positive")
    return args


def main():
    args = parse_args()
    try:
        cfg = load_task_config(args)
        if args.check_only:
            full, latent, initial, oracle = load_inputs(cfg)
            X = latent.filter(regex="^latent_").to_numpy(float)
            Y = full[OBJECTIVES].to_numpy(float)
            if X.shape != (7532, cfg["latent_dim"]) or Y.shape != (7532, 3):
                raise ValueError("Input shapes do not match configuration")
            if not np.isfinite(X).all() or not np.isfinite(Y).all():
                raise ValueError("Non-finite input data")
            print("TASK CONFIGURATION PASSED")
        elif args._worker:
            attempt = Path(args._attempt_dir).resolve()
            task_root = (ROOT / "results/tasks").resolve()
            if attempt.parent != task_root or not attempt.name.startswith(f".{cfg['task_id']}_") or not attempt.is_dir() or any(attempt.iterdir()):
                raise ValueError("Worker requires a fresh isolated attempt directory")
            run_attempt(cfg, args, attempt, choose_device(args.device))
        else:
            run_with_fallback(cfg, args)
    except Exception as error:
        if args._worker and args.device.startswith("cuda") and is_accelerator_oom(error):
            print("Accelerator out of memory.", file=sys.stderr)
            return OOM_EXIT_CODE
        print(f"TASK FAILED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
