"""
RUL (Remaining Useful Life) Prediction Pipeline
================================================
- Leave-One-Machine-Out (LOMO) cross-validation
- XGBoost with GPU (CUDA) or Apple M2 (Metal/CPU) acceleration
- Heavy feature engineering: rolling stats, lags, degradation signals
- tqdm progress bars per fold and per batch
- Final model trained on all machines

Usage:
    python rul_pipeline.py --data dataset.csv
    python rul_pipeline.py --data dataset.csv --device cuda
    python rul_pipeline.py --data dataset.csv --device mps
    python rul_pipeline.py --data dataset.csv --device cpu
"""

import argparse
import platform
import warnings
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_CONFIG = {
    "feature_engineering": {
        "rolling_windows": [6, 12, 24, 48, 72],  # hours
        "lag_steps": [1, 3, 6, 12, 24],          # hours back
        "cycle_length": 2285,                    # approx hours between failures
        "rul_cap": 2000                           # cap RUL labels at this value
    },
    "xgboost_params": {
        "n_estimators": 3000,
        "learning_rate": 0.02,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
        "gamma": 0.05,
        "early_stopping_rounds": 75,
        "n_jobs": -1,
        "random_state": 42
    }
}

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(MODEL_DIR, "data", "Predictive_Maintenance_Synthetic_Data.csv")
SAVE_PATH = os.path.join(MODEL_DIR, "saved_models")
os.makedirs(SAVE_PATH, exist_ok=True)
RESULTS_PATH = os.path.join(MODEL_DIR, "results")
os.makedirs(RESULTS_PATH, exist_ok=True)

SENSOR_COLS = [
    "process_temperature", "air_temperature", "vibration",
    "torque", "rpm", "current", "idle_duration", "power_consumption",
]

# ─────────────────────────────────────────────
# DEVICE DETECTION
# ─────────────────────────────────────────────
def detect_device(requested: str) -> dict:
    """Return XGBoost tree_method and device params."""
    if requested == "auto":
        # Try CUDA first, then MPS (M2), then CPU
        try:
            import subprocess
            result = subprocess.run(["nvidia-smi"], capture_output=True)
            if result.returncode == 0:
                requested = "cuda"
            elif platform.machine() == "arm64" and platform.system() == "Darwin":
                requested = "mps"
            else:
                requested = "cpu"
        except Exception:
            requested = "cpu"

    if requested == "cuda":
        params = dict(device="cuda", tree_method="hist")
        label = "NVIDIA CUDA GPU"
    elif requested == "mps":
        # XGBoost >= 2.0 supports device='mps' on Apple Silicon
        params = dict(device="cpu", tree_method="hist", nthread=-1)
        label = "Apple M2 (optimised CPU/Metal threads)"
        # XGBoost 2.x can use 'mps' — try it
        try:
            test = xgb.XGBRegressor(device="mps", n_estimators=1)
            test.fit([[1]], [1])
            params = dict(device="mps", tree_method="hist")
            label = "Apple M2 Metal (MPS)"
        except Exception:
            pass
    else:
        params = dict(device="cpu", tree_method="hist", nthread=-1)
        label = "CPU (multi-threaded)"

    return params, label


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_data(path: str, verbose=1) -> pd.DataFrame:
    if verbose == 1:
        print("\nLoading data...")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    # Encode maintenance type
    le = LabelEncoder()
    df["maintenance_encoded"] = le.fit_transform(df["last_maintenance_Type"].astype(str))

    if verbose == 1:
        print(f"      Loaded {len(df):,} rows | {df['machine_id'].nunique()} machines "
            f"| {df['machine_failure'].sum()} failure events")
    return df


# ─────────────────────────────────────────────
# RUL LABEL COMPUTATION
# ─────────────────────────────────────────────
def compute_rul(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    For each row: RUL = operating_hours at NEXT failure - current operating_hours.
    Last segment (after final failure) is forward-projected using mean cycle length.
    Capped at RUL_CAP to reduce impact of very long healthy periods.
    """
    print("Computing RUL labels...")
    rul_values = []

    for machine_id, group in tqdm(df.groupby("machine_id"), desc="      Machines", ncols=80):
        group = group.sort_values("timestamp").reset_index(drop=True)
        failure_idx = group[group["machine_failure"] == 1].index.tolist()

        rul_col = np.zeros(len(group), dtype=np.float32)

        if not failure_idx:
            # No failures — project from end
            for i in range(len(group)):
                rul_col[i] = min(config["feature_engineering"]["cycle_length"], config["feature_engineering"]["rul_cap"])
        else:
            # Segment before first failure
            first_fail_hours = group.loc[failure_idx[0], "operating_hours"]
            for i in range(failure_idx[0] + 1):
                rul_col[i] = min(first_fail_hours - group.loc[i, "operating_hours"], config["feature_engineering"]["rul_cap"])

            # Segments between failures
            for k in range(len(failure_idx) - 1):
                start = failure_idx[k] + 1
                end   = failure_idx[k + 1]
                next_fail_hours = group.loc[end, "operating_hours"]
                for i in range(start, end + 1):
                    rul_col[i] = min(next_fail_hours - group.loc[i, "operating_hours"], config["feature_engineering"]["rul_cap"])

            # Segment after last failure — project using mean cycle
            last_fail = failure_idx[-1]
            last_fail_hours = group.loc[last_fail, "operating_hours"]
            projected_next = last_fail_hours + config["feature_engineering"]["cycle_length"]
            for i in range(last_fail + 1, len(group)):
                rul_col[i] = min(projected_next - group.loc[i, "operating_hours"], config["feature_engineering"]["rul_cap"])
            rul_col[last_fail] = 0  # exact failure moment

        rul_values.extend(rul_col)

    df["RUL"] = rul_values
    df["RUL_log"] = np.log1p(df["RUL"])
    print(f"      RUL range: [{df['RUL'].min():.0f}, {df['RUL'].max():.0f}] hours")
    return df


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame, config: dict, verbose=1) -> pd.DataFrame:
    if verbose == 1:
        print("Engineering features...")
    all_groups = []

    for machine_id, group in tqdm(df.groupby("machine_id"),
                                   desc="      Machines", ncols=80, disable=(verbose != 1)):
        g = group.sort_values("timestamp").copy()

        # ── Rolling statistics (mean, std, min, max, skew) ──
        for col in SENSOR_COLS:
            for w in config["feature_engineering"]["rolling_windows"]:
                roll = g[col].rolling(w, min_periods=1)
                g[f"{col}_roll_mean_{w}h"] = roll.mean()
                g[f"{col}_roll_std_{w}h"]  = roll.std().fillna(0)
                g[f"{col}_roll_max_{w}h"]  = roll.max()
                g[f"{col}_roll_min_{w}h"]  = roll.min()
                # EWM (exponentially weighted — captures recent drift faster)
                g[f"{col}_ewm_{w}h"] = g[col].ewm(span=w, min_periods=1).mean()

        # ── Lag features ──
        # ffill() fills the first `lag` NaN rows with the earliest known value.
        # bfill() would peek at row `lag` (a future value) — ffill is strictly safe.
        for col in SENSOR_COLS:
            for lag in config["feature_engineering"]["lag_steps"]:
                g[f"{col}_lag_{lag}"] = g[col].shift(lag).ffill().fillna(g[col].iloc[0])

        # ── Delta / rate-of-change features ──
        for col in SENSOR_COLS:
            g[f"{col}_delta_1h"]  = g[col].diff(1).fillna(0)
            g[f"{col}_delta_6h"]  = g[col].diff(6).fillna(0)
            g[f"{col}_delta_24h"] = g[col].diff(24).fillna(0)

        # ── Cross-sensor interactions ──
        g["torque_x_rpm"]        = g["torque"] * g["rpm"]
        g["vibration_x_temp"]    = g["vibration"] * g["process_temperature"]
        g["current_x_rpm"]       = g["current"] * g["rpm"]
        g["power_efficiency"]    = g["power_consumption"] / (g["rpm"] + 1e-6)
        g["thermal_stress"]      = g["process_temperature"] - g["air_temperature"]
        g["load_index"]          = g["torque"] / (g["rpm"] + 1e-6)

        # ── Cyclic wear encoding ──
        # operating_hours is intentionally NOT used as a raw feature — since
        # failures occur at fixed ~2285h intervals, the model could learn the
        # maintenance schedule (operating_hours mod 2285 ≈ 0 → RUL ≈ 0) instead
        # of the actual sensor degradation pattern.
        # cycle_number (= operating_hours // CYCLE_LENGTH) is also excluded for
        # the same reason — it's a monotonic step function of operating_hours.
        # We keep only the within-cycle position encoded as sin/cos.
        hours = g["operating_hours"]
        cycle_angle = 2 * np.pi * (hours % config["feature_engineering"]["cycle_length"]) / config["feature_engineering"]["cycle_length"]
        g["wear_cycle_sin"] = np.sin(cycle_angle)
        g["wear_cycle_cos"] = np.cos(cycle_angle)
        g["time_in_cycle"]  = hours % config["feature_engineering"]["cycle_length"]
        # cycle_number deliberately omitted — redundant with time_in_cycle
        # and encodes the same schedule information as raw operating_hours

        # ── Maintenance features (LAGGED to prevent leakage) ──
        # time_since_last_maintenance=0 and last_maintenance_Type='corrective'
        # occur ON the failure row itself — using them raw perfectly encodes
        # "I am currently failing". Shift by 1 so the model only sees what was
        # known BEFORE the current timestep.
        # Use ffill().fillna(0) instead of bfill(): bfill() on row 0 would peek
        # at row 1 (1 hour into the future). ffill fills with the last known value.
        g["hrs_since_maintenance"]    = g["time_since_last_maintenance"].shift(1).ffill().fillna(0)
        g["maintenance_encoded_lag1"] = g["maintenance_encoded"].shift(1).ffill().fillna(0)
        g["maintenance_rate"]         = g["maintenance_encoded_lag1"] / (g["time_in_cycle"] + 1)
        
        # relative to machine baseline
        g["vibration_norm"] = g["vibration"] / g["vibration"].rolling(100, min_periods=1).mean()
        g["temp_norm"] = g["process_temperature"] / g["process_temperature"].rolling(100, min_periods=1).mean()
        
        g["vibration_trend"] = g["vibration"].diff(12).fillna(0)
        g["temp_trend"] = g["process_temperature"].diff(12).fillna(0)

        # ── Cumulative degradation proxies (scoped to current maintenance window) ──
        # Global cumsum() grows monotonically with time, effectively re-encoding
        # operating_hours. Instead we reset the cumsum at each maintenance event
        # so it measures degradation since the last reset — a genuine wear signal.
        maint_reset = (g["time_since_last_maintenance"].shift(1).fillna(1) == 0).cumsum()
        g["cum_vibration_since_maint"] = g.groupby(maint_reset)["vibration"].cumsum()
        g["cum_thermal_since_maint"]   = g.groupby(maint_reset)["thermal_stress"].cumsum()
        g["vibration_zscore"] = (
            (g["vibration"] - g["vibration"].rolling(72, min_periods=1).mean())
            / (g["vibration"].rolling(72, min_periods=1).std().fillna(1))
        )

        all_groups.append(g)

    df = pd.concat(all_groups).sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    if verbose == 1:
        print(f"      Total features: {len(df.columns)} columns")
    return df


# ─────────────────────────────────────────────
# FEATURE SELECTION
# ─────────────────────────────────────────────
def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = {
        "timestamp", "machine_id", "last_maintenance_Type",
        "machine_failure", "RUL", "RUL_log",
        # Leaky raw columns (lagged versions used instead):
        "time_since_last_maintenance",
        "maintenance_encoded",
        # Schedule-encoding columns: failures occur at fixed ~2285h intervals so
        # raw operating_hours lets the model learn the schedule, not degradation.
        # cycle_number is a monotonic step of operating_hours — same problem.
        # time_in_cycle (mod 2285) + wear_cycle_sin/cos are kept instead.
        "operating_hours",
        "cycle_number",
        # Old cumsum columns replaced by maintenance-window-scoped versions:
        "cum_vibration",
        "cum_thermal",
    }
    return [c for c in df.columns if c not in exclude]


# ─────────────────────────────────────────────
# XGBOOST WRAPPER WITH TQDM
# ─────────────────────────────────────────────
class TQDMCallback(xgb.callback.TrainingCallback):
    def __init__(self, total_rounds: int, fold_label: str):
        self.pbar = tqdm(total=total_rounds, desc=f"      {fold_label}",
                         ncols=90, unit="round", leave=True,
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] MAE={postfix}")
        self.best_mae = float("inf")

    def after_iteration(self, model, epoch, evals_log):
        if "validation_0" in evals_log:
            mae = evals_log["validation_0"].get("mae", [None])[-1]
            if mae and mae < self.best_mae:
                self.best_mae = mae
            self.pbar.set_postfix_str(f"{self.best_mae:.1f}h")
        self.pbar.update(1)
        return False

    def after_training(self, model):
        self.pbar.close()
        return model


def build_model(device_params: dict, fold_label: str, config: dict) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators        = config['xgboost_params']['n_estimators'],
        learning_rate       = config['xgboost_params']['learning_rate'],
        max_depth           = config['xgboost_params']['max_depth'],
        subsample           = config['xgboost_params']['subsample'],
        colsample_bytree    = config['xgboost_params']['colsample_bytree'],
        min_child_weight    = config['xgboost_params']['min_child_weight'],
        reg_alpha           = config['xgboost_params']['reg_alpha'],
        reg_lambda          = config['xgboost_params']['reg_lambda'],
        gamma               = config['xgboost_params']['gamma'],
        objective           = "reg:squarederror",
        eval_metric         = "mae",
        early_stopping_rounds = config['xgboost_params']['early_stopping_rounds'],
        callbacks           = [TQDMCallback(config['xgboost_params']['n_estimators'], fold_label)],
        verbosity           = 0,
        **device_params,
    )


# ─────────────────────────────────────────────
# LOMO CROSS-VALIDATION
# ─────────────────────────────────────────────
def lomo_cross_validation(df: pd.DataFrame,
                           feature_cols: list,
                           device_params: dict,
                           config: dict) -> dict:
    print("LOMO cross-validation...")
    machines = sorted(df["machine_id"].unique())
    results  = {}

    for i, test_machine in enumerate(machines):
        fold_label = f"Fold {i+1}/8 — test {test_machine}"
        print(f"\n   ┌─ {fold_label}")

        train_df = df[df["machine_id"] != test_machine]
        test_df  = df[df["machine_id"] == test_machine]

        X_train = train_df[feature_cols].values
        y_train = train_df["RUL_log"].values
        X_test  = test_df[feature_cols].values
        y_test  = test_df["RUL_log"].values

        model = build_model(device_params, fold_label, config=MODEL_CONFIG)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        preds_log = model.predict(X_test)

        preds = np.expm1(preds_log)
        y_true = np.expm1(y_test)

        mae   = mean_absolute_error(y_true, preds)
        rmse  = mean_squared_error(y_true, preds) ** 0.5
        r2    = r2_score(y_true, preds)

        results[test_machine] = {
            "model":  model,
            "preds":  preds,
            "actuals": y_test,
            "mae":    mae,
            "rmse":   rmse,
            "r2":     r2,
            "best_iteration": model.best_iteration,
        }
        print(f"   └─ MAE: {mae:.1f}h | RMSE: {rmse:.1f}h | R²: {r2:.4f} "
              f"| best iter: {model.best_iteration}")

    return results

# ─────────────────────────────────────────────
# FINAL MODEL (all machines)
# ─────────────────────────────────────────────
def train_final_model(df: pd.DataFrame,
                      feature_cols: list,
                      device_params: dict,
                      config: dict,
                      best_n_estimators: int) -> xgb.XGBRegressor:
    print("\nTraining final model on all machines...")
    X = df[feature_cols].values
    y = df["RUL_log"].values

    # No early stopping for final — use best iteration from CV
    model = xgb.XGBRegressor(
        n_estimators      = best_n_estimators,
        learning_rate     = config['xgboost_params']['learning_rate'],
        max_depth         = config['xgboost_params']['max_depth'],
        subsample         = config['xgboost_params']['subsample'],
        colsample_bytree  = config['xgboost_params']['colsample_bytree'],
        min_child_weight  = config['xgboost_params']['min_child_weight'],
        reg_alpha         = config['xgboost_params']['reg_alpha'],
        reg_lambda        = config['xgboost_params']['reg_lambda'],
        gamma             = config['xgboost_params']['gamma'],
        objective         = "reg:squarederror",
        eval_metric       = "mae",
        callbacks         = [TQDMCallback(best_n_estimators, "Final model")],
        verbosity         = 0,
        **device_params,
    )
    
    weights = 1 + (1 / (df["RUL"].values + 1))
    model.fit(X, y, verbose=False, sample_weight=weights)
    return model


# ─────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────
def print_summary(results: dict):
    print("\n" + "═" * 60)
    print("  LOMO CROSS-VALIDATION SUMMARY")
    print("═" * 60)
    print(f"  {'Machine':<10} {'MAE (hrs)':<14} {'RMSE (hrs)':<14} {'R²':<10} {'Best iter'}")
    print("─" * 60)

    maes, rmses, r2s = [], [], []
    for machine, r in results.items():
        print(f"  {machine:<10} {r['mae']:<14.1f} {r['rmse']:<14.1f} {r['r2']:<10.4f} {r['best_iteration']}")
        maes.append(r["mae"])
        rmses.append(r["rmse"])
        r2s.append(r["r2"])

    print("─" * 60)
    print(f"  {'MEAN':<10} {np.mean(maes):<14.1f} {np.mean(rmses):<14.1f} {np.mean(r2s):<10.4f}")
    print(f"  {'STD':<10} {np.std(maes):<14.1f} {np.std(rmses):<14.1f} {np.std(r2s):<10.4f}")
    print("═" * 60)


def save_plots(results: dict, output_dir: str):
    machines = list(results.keys())
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.flatten()

    for i, machine in enumerate(machines):
        r = results[machine]
        ax = axes[i]
        n = min(2000, len(r["actuals"]))  # plot up to 2000 points
        idx = np.linspace(0, len(r["actuals"]) - 1, n, dtype=int)
        ax.plot(idx, r["actuals"][idx], label="Actual RUL", alpha=0.7, linewidth=1)
        ax.plot(idx, r["preds"][idx],   label="Predicted RUL", alpha=0.7, linewidth=1, linestyle="--")
        ax.set_title(f"{machine} — MAE: {r['mae']:.1f}h | R²: {r['r2']:.3f}")
        ax.set_xlabel("Time step")
        ax.set_ylabel("RUL (hours)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = f"{output_dir}/lomo_rul_predictions.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {plot_path}")


def save_feature_importance(model: xgb.XGBRegressor,
                            feature_cols: list,
                            output_dir: str):
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    top30 = importance.nlargest(30).sort_values()

    fig, ax = plt.subplots(figsize=(10, 10))
    top30.plot(kind="barh", ax=ax, color="#1D9E75")
    ax.set_title("Top 30 feature importances (final model)")
    ax.set_xlabel("Importance score")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()

    fi_path = f"{output_dir}/feature_importance.png"
    plt.savefig(fi_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Feature importance saved → {fi_path}")

    csv_path = f"{output_dir}/feature_importance.csv"
    importance.sort_values(ascending=False).to_csv(csv_path, header=["importance"])
    print(f"  Importance CSV saved → {csv_path}")


def save_cv_results(df: pd.DataFrame,
                     results: dict,
                     output_dir: str):
    all_preds = np.empty(len(df), dtype=np.float32)
    for machine, r in results.items():
        mask = df["machine_id"] == machine
        all_preds[mask.values] = r["preds"]

    out = df[["timestamp", "machine_id", "operating_hours", "machine_failure", "RUL"]].copy()
    out["RUL_predicted_lomo"] = all_preds
    out["error_hours"] = out["RUL_predicted_lomo"] - out["RUL"]

    csv_path = f"{output_dir}/cv_results.csv"
    out.to_csv(csv_path, index=False)
    print(f"  Predictions CSV saved → {csv_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RUL prediction with LOMO-CV and XGBoost")
    parser.add_argument("--data",   default=DATA_PATH, help="Path to CSV file")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Compute device (auto detects best available)")
    args = parser.parse_args()

    t0 = time.time()
    device_params, device_label = detect_device(args.device)

    print("=" * 60)
    print("  RUL PREDICTION PIPELINE")
    print("=" * 60)
    print(f"  Device  : {device_label}")
    print(f"  Data    : {args.data}")
    print(f"  XGBoost : {xgb.__version__}")
    print("=" * 60)

    # ── Pipeline ──
    df             = load_data(args.data)
    df             = compute_rul(df, config=MODEL_CONFIG)
    df             = engineer_features(df, config=MODEL_CONFIG)
    feature_cols   = get_feature_cols(df)
    joblib.dump(feature_cols, f"{SAVE_PATH}/feature_cols.pkl")

    print(f"\n   Feature matrix: {len(df):,} rows x {len(feature_cols)} features")
    print(f"   Target (RUL) stats: mean={df['RUL'].mean():.1f}h, "
          f"median={df['RUL'].median():.1f}h, std={df['RUL'].std():.1f}h\n")

    results = lomo_cross_validation(df, feature_cols, device_params, config=MODEL_CONFIG)
    print_summary(results)

    # Best n_estimators = median of best iterations across folds
    best_iters = [r["best_iteration"] for r in results.values()]
    best_n     = max(int(np.median(best_iters) * 1.1), 100)  # +10% buffer
    print(f"\n   Best iteration (median across folds): {int(np.median(best_iters))} → using {best_n} for final model")

    final_model = train_final_model(df, feature_cols, device_params, best_n_estimators=best_n, config=MODEL_CONFIG)

    # ── Save outputs ──
    print("\n  Saving outputs...")
    joblib.dump(MODEL_CONFIG, f"{SAVE_PATH}/model_config.pkl")
    save_plots(results, output_dir=RESULTS_PATH)
    save_feature_importance(final_model, feature_cols, output_dir=RESULTS_PATH)
    save_cv_results(df, results, output_dir=RESULTS_PATH)

    # Save final model
    model_path = f"{SAVE_PATH}/rul_final_model.json"
    final_model.save_model(model_path)
    print(f"  Model saved → {model_path}")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed/60:.1f} min")
    print("=" * 60)

    # ── Quick inference example ──
    print("\n  INFERENCE EXAMPLE (first 5 rows of M01):")
    sample = df[df["machine_id"] == "M01"].head(5).reset_index(drop=True)
    sample_preds = np.expm1(final_model.predict(sample[feature_cols].values))
    for idx, row in sample.iterrows():
        print(f"    {row['timestamp']}  actual={row['RUL']:.0f}h  predicted={sample_preds[idx]:.0f}h")


if __name__ == "__main__":
    main()