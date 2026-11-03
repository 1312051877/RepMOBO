import gc
import json
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import PredefinedSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =========================================================
# 全局配置
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1] / "data"
SPLIT_FILE = BASE_DIR / "fixed_splits.xlsx"

OUTPUT_DIR = BASE_DIR.parent / "results_lgbm"
MODELS_DIR = OUTPUT_DIR / "saved_models"
PRED_DIR = OUTPUT_DIR / "predictions"
TRIALS_DIR = OUTPUT_DIR / "optuna_trials"
FI_DIR = OUTPUT_DIR / "feature_importance"
METRICS_DIR = OUTPUT_DIR / "metrics"
PARAMS_DIR = OUTPUT_DIR / "best_params"

for p in [OUTPUT_DIR, MODELS_DIR, PRED_DIR, TRIALS_DIR, FI_DIR, METRICS_DIR, PARAMS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# 5组特征文件全集
FEATURE_FILES = {
    "Manual": BASE_DIR / "Manual.csv",
    "MOFTransformer": BASE_DIR / "MOFTransformer.csv",
    "UniMOF": BASE_DIR / "unimof.csv",
    "Manual_MOFTransformer": BASE_DIR / "Manual_MOFTransformer_final.csv",
    "Manual_UniMOF": BASE_DIR / "Manual_UniMOF_final.csv",
}

TARGETS = ["CEES", "DMMP", "EtCl"]
META_COLS = ["mof_id", "CEES", "DMMP", "EtCl"]

RANDOM_STATE = 2026
N_TRIALS = 50

# ---------------------------------------------------------
# 笔记本运行资源分配 (单任务内部多线程)
# ---------------------------------------------------------
CV_N_JOBS = 1
OPTUNA_N_JOBS = 1
LGBM_THREADS_PER_FIT = 6
TOTAL_CPU = 6


# ---------------------------------------------------------

# =========================================================
# 工具函数 (已加入强力工程防御机制)
# =========================================================
def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600;
    m = (seconds % 3600) // 60;
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def top_k_hit_rate(y_true, y_pred, k_ratio=0.05):
    k = max(1, int(np.ceil(len(y_true) * k_ratio)))
    true_top_idx = set(np.argsort(y_true)[::-1][:k])
    pred_top_idx = set(np.argsort(y_pred)[::-1][:k])
    return len(true_top_idx & pred_top_idx) / len(true_top_idx)


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def load_all_splits():
    split_map = {}
    with pd.ExcelFile(SPLIT_FILE) as workbook:
        sheet_names = workbook.sheet_names
    for target in TARGETS:
        sheet = "EtCL" if target == "EtCl" and target not in sheet_names and "EtCL" in sheet_names else target
        split_df = pd.read_excel(SPLIT_FILE, sheet_name=sheet)
        split_df["mof_id"] = normalize_id(split_df["mof_id"])
        split_map[target] = split_df
    return split_map


def load_feature_df(feature_name: str, file_path: Path) -> pd.DataFrame:
    print(f"\n[{feature_name}] 正在加载数据...")
    last_err = None

    # 1. 编码回退机制
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
        try:
            df = pd.read_csv(file_path, encoding=enc, low_memory=False)
            print(f"[OK] 读取成功: {file_path.name} | encoding={enc} | shape={df.shape}")
            break
        except Exception as e:
            last_err = e
    else:
        raise RuntimeError(f"无法读取文件: {file_path}\n最后错误: {last_err}")

    # 2. 列完整性与重复ID检查
    # Normalize a legacy header in memory; do not alter the input CSV.
    if "EtCl" not in df.columns and "EtCL" in df.columns:
        df = df.rename(columns={"EtCL": "EtCl"})
    missing = [c for c in META_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{feature_name} 缺少必要列: {missing}")

    df["mof_id"] = normalize_id(df["mof_id"])
    if df["mof_id"].duplicated().any():
        dup_id = df.loc[df["mof_id"].duplicated(), "mof_id"].iloc[0]
        raise ValueError(f"{feature_name} 存在重复 mof_id，例如: {dup_id}")

    return df


def prepare_task_arrays(feature_df: pd.DataFrame, split_df: pd.DataFrame, feature_name: str, target_col: str):
    feature_cols = [c for c in feature_df.columns if c not in META_COLS]
    merged = feature_df.merge(split_df, on="mof_id", how="inner", validate="one_to_one")

    # 3. 严格检查 Merge 后的行数是否流失
    if len(merged) != len(feature_df):
        raise ValueError(f"{feature_name} 与 split 表 merge 后行数不一致: {len(merged)} vs {len(feature_df)}")

    train_df = merged[merged["split"] == "train"].reset_index(drop=True)
    test_df = merged[merged["split"] == "test"].reset_index(drop=True)

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_train = pd.to_numeric(train_df[target_col], errors="raise").to_numpy(dtype=np.float32, copy=True)
    ids_train = train_df["mof_id"].to_numpy()
    fold_train = train_df["cv_fold"].astype(int).to_numpy()

    X_test = test_df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_test = pd.to_numeric(test_df[target_col], errors="raise").to_numpy(dtype=np.float32, copy=True)
    ids_test = test_df["mof_id"].to_numpy()

    cv = PredefinedSplit(test_fold=fold_train)
    return X_train, y_train, ids_train, X_test, y_test, ids_test, feature_cols, cv


def task_done(task_prefix: str) -> bool:
    # 4. 严密的已完成任务校验
    return (
            (METRICS_DIR / f"{task_prefix}__metrics.json").exists() and
            (PRED_DIR / f"{task_prefix}__test_predictions.csv").exists() and
            (PARAMS_DIR / f"{task_prefix}__best_params.json").exists() and
            (MODELS_DIR / f"{task_prefix}__best_model.joblib").exists()
    )


# =========================================================
# Pipeline 构建 (包含资格赛参数空间)
# =========================================================
def build_cv_pipeline(trial):
    model = LGBMRegressor(
        objective="regression", boosting_type="gbdt", random_state=RANDOM_STATE,
        n_jobs=LGBM_THREADS_PER_FIT, verbose=-1,

        n_estimators=trial.suggest_int("n_estimators", 300, 2500),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.10, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 127),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 120),
        min_child_weight=trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 20.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True),
        min_split_gain=trial.suggest_float("min_split_gain", 0.0, 0.5),
        max_bin=trial.suggest_int("max_bin", 127, 255),
    )
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def build_final_pipeline(best_params):
    model = LGBMRegressor(
        objective="regression", boosting_type="gbdt", random_state=RANDOM_STATE,
        n_jobs=TOTAL_CPU, verbose=-1, subsample_freq=1, **best_params
    )
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def get_feature_importance(fitted_pipeline, feature_names):
    model = fitted_pipeline.named_steps["model"]
    return pd.DataFrame({
        "feature": feature_names, "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)


# =========================================================
# 单个任务主函数
# =========================================================
def run_single_task(feature_name, feature_df, split_df, target_col, global_start_time, finished_count, total_tasks):
    task_prefix = f"{target_col}__{feature_name}"

    if task_done(task_prefix):
        print(f"[SKIP] 任务已完整完成: {task_prefix}")
        with open(METRICS_DIR / f"{task_prefix}__metrics.json", "r", encoding="utf-8") as f:
            summary = json.load(f)
        with open(PARAMS_DIR / f"{task_prefix}__best_params.json", "r", encoding="utf-8") as f:
            best_params = json.load(f)
        params_row = {"Model": "LGBM", "Feature_Set": feature_name, "Target": target_col, **best_params}
        return summary, params_row, finished_count + 1

    print("\n" + "-" * 80)
    print(f"执行任务: [{feature_name}] -> 预测 [{target_col}]")
    X_train, y_train, ids_train, X_test, y_test, ids_test, feature_cols, cv = prepare_task_arrays(
        feature_df, split_df, feature_name, target_col
    )

    trial_records = []
    pbar = tqdm(total=N_TRIALS, desc="Optuna Tuning", ncols=100)

    def objective(trial):
        pipe = build_cv_pipeline(trial)
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
            n_jobs=CV_N_JOBS, return_train_score=False
        )
        cv_rmse = -np.mean(scores["test_rmse"])
        cv_r2 = np.mean(scores["test_r2"])

        trial_records.append({"trial": trial.number, "cv_rmse": cv_rmse, "cv_r2": cv_r2, **trial.params})
        pbar.set_postfix(
            {"best_rmse": f"{study.best_value:.4f}" if trial.number > 0 else "-", "curr": f"{cv_rmse:.4f}"})
        pbar.update(1)
        return cv_rmse

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=OPTUNA_N_JOBS)
    pbar.close()

    best_params = study.best_params

    # 全量数据重新训练
    final_pipe = build_final_pipeline(best_params)
    final_pipe.fit(X_train, y_train)
    y_pred = final_pipe.predict(X_test)

    # 保存预测结果
    pd.DataFrame({
        "mof_id": ids_test, "Actual": y_test, "Predicted": y_pred, "Abs_Error": np.abs(y_test - y_pred)
    }).sort_values("Actual", ascending=False).to_csv(PRED_DIR / f"{task_prefix}__test_predictions.csv", index=False,
                                                     encoding="utf-8-sig")

    # 结果评估与保存
    metrics_dict = {
        "Model": "LGBM", "Feature_Set": feature_name, "Target": target_col,
        "N_Train": int(len(y_train)), "N_Test": int(len(y_test)), "N_Features": int(len(feature_cols)),
        "Best_CV_RMSE": float(study.best_value),
        "Test_R2": float(r2_score(y_test, y_pred)),
        "Test_RMSE": float(rmse(y_test, y_pred)),
        "Test_MAE": float(mean_absolute_error(y_test, y_pred)),
        "Test_Spearman": float(spearmanr(y_test, y_pred).correlation),
        "Test_Top1": float(top_k_hit_rate(y_test, y_pred, k_ratio=0.01)),
        "Test_Top5": float(top_k_hit_rate(y_test, y_pred, k_ratio=0.05))
    }
    print(f"--> Test R²: {metrics_dict['Test_R2']:.4f} | Top-5%: {metrics_dict['Test_Top5']:.4f}")

    pd.DataFrame([metrics_dict]).to_csv(METRICS_DIR / f"{task_prefix}__metrics.csv", index=False, encoding="utf-8-sig")
    with open(METRICS_DIR / f"{task_prefix}__metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=2)

    get_feature_importance(final_pipe, feature_cols).to_csv(FI_DIR / f"{task_prefix}__feature_importance.csv",
                                                            index=False, encoding="utf-8-sig")
    pd.DataFrame(trial_records).sort_values("cv_rmse").to_csv(TRIALS_DIR / f"{task_prefix}__optuna_trials.csv",
                                                              index=False, encoding="utf-8-sig")

    with open(PARAMS_DIR / f"{task_prefix}__best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    joblib.dump(final_pipe, MODELS_DIR / f"{task_prefix}__best_model.joblib")

    # 生成供总表使用的 params_row
    params_row = {"Model": "LGBM", "Feature_Set": feature_name, "Target": target_col, **best_params}

    # 垃圾回收
    del X_train, y_train, X_test, y_test, final_pipe
    gc.collect()

    finished_count += 1
    eta = ((time.time() - global_start_time) / finished_count) * (total_tasks - finished_count)
    print(f"总进度: {finished_count}/{total_tasks} | 预计剩余: {format_seconds(eta)}")

    return metrics_dict, params_row, finished_count


# =========================================================
# 执行入口
# =========================================================
def main():
    print("=" * 80)
    print("🚀 LGBM Phase I Benchmark (Final Robust Edition)")
    print("=" * 80)

    split_map = load_all_splits()
    total_tasks = len(FEATURE_FILES) * len(TARGETS)
    finished_count = 0
    global_start_time = time.time()

    all_summaries = []
    all_best_params_rows = []

    for feature_name, file_path in FEATURE_FILES.items():
        feature_df = load_feature_df(feature_name, file_path)
        for target_col in TARGETS:
            summary, params_row, finished_count = run_single_task(
                feature_name, feature_df, split_map[target_col], target_col,
                global_start_time, finished_count, total_tasks
            )
            all_summaries.append(summary)
            all_best_params_rows.append(params_row)

        del feature_df
        gc.collect()

    # 1. 整体指标汇总
    summary_df = pd.DataFrame(all_summaries).sort_values(["Target", "Test_R2"], ascending=[True, False])
    summary_df.to_csv(OUTPUT_DIR / "ALL_RESULTS_SUMMARY.csv", index=False, encoding="utf-8-sig")
    summary_df.to_excel(OUTPUT_DIR / "ALL_RESULTS_SUMMARY.xlsx", index=False)

    # 2. 整体参数汇总 (ChatGPT 第5点建议)
    params_df = pd.DataFrame(all_best_params_rows).sort_values(["Target", "Feature_Set"])
    params_df.to_csv(OUTPUT_DIR / "ALL_BEST_PARAMS.csv", index=False, encoding="utf-8-sig")
    params_df.to_excel(OUTPUT_DIR / "ALL_BEST_PARAMS.xlsx", index=False)

    print("\n🎉 全部 15 个任务已完成！总耗时:", format_seconds(time.time() - global_start_time))


if __name__ == "__main__":
    main()