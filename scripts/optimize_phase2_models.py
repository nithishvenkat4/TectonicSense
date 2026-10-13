"""TectonicSense - Hyperparameter Optimization Script (Optuna)

Phase 2 — Machine Learning
Optimizes the top candidate gradient boosting models:
  1. XGBoost
  2. LightGBM

Using Optuna (TPESampler, seed=42).
Primary objective: Maximize Validation Macro F1.
NO test set data leakage.

Outputs:
  - results/phase_2_machine_learning/hyperparameter_tuning/optuna_results.csv
  - results/phase_2_machine_learning/hyperparameter_tuning/best_parameters.json
  - results/phase_2_machine_learning/hyperparameter_tuning/optimization_summary.json
  - results/phase_2_machine_learning/hyperparameter_tuning/optimization_history.png
  - results/phase_2_machine_learning/hyperparameter_tuning/tuned_vs_untuned_comparison.csv
"""

from pathlib import Path
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

import xgboost as xgb
import lightgbm as lgb
import optuna

# Deterministic seed
RANDOM_STATE = 42
N_TRIALS_PER_MODEL = 30

# Project paths
ROOT_DIR = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT_DIR / "data" / "processed" / "splits"
ADVANCED_RESULTS_JSON = ROOT_DIR / "results" / "phase_2_machine_learning" / "advanced_models" / "advanced_model_results.json"
TUNING_DIR = ROOT_DIR / "results" / "phase_2_machine_learning" / "hyperparameter_tuning"

TARGET_COL = "boundary_class"


def load_data():
    """Load train and validation splits."""
    print("=" * 70)
    print("LOADING DATA FOR HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)

    train_path = SPLITS_DIR / "train.csv"
    val_path = SPLITS_DIR / "validation.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    feature_cols = [c for c in train_df.columns if c != TARGET_COL]

    # Verify no leakage columns
    forbidden = ["event_id", "boundary_id", "boundary_step_id"]
    for col in forbidden:
        if col in feature_cols:
            raise ValueError(f"Leakage hazard: '{col}' found in features!")

    X_train = train_df[feature_cols].copy()
    y_train_raw = train_df[TARGET_COL].copy()

    X_val = val_df[feature_cols].copy()
    y_val_raw = val_df[TARGET_COL].copy()

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)

    classes = le.classes_.tolist()
    print(f"Train samples: {len(X_train):,}, Validation samples: {len(X_val):,}")
    print(f"Target classes ({len(classes)}): {classes}")

    return X_train, y_train, X_val, y_val, feature_cols, classes, le


def compute_metrics(y_true, y_pred, elapsed_train=0.0, elapsed_infer=0.0):
    """Compute standard multiclass metrics focusing on Macro F1."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "training_time_sec": round(float(elapsed_train), 2),
        "inference_time_sec": round(float(elapsed_infer), 4),
    }


def optimize_xgboost(X_train, y_train, X_val, y_val, n_trials=N_TRIALS_PER_MODEL):
    """Run Optuna study for XGBoost maximizing Validation Macro F1."""
    print("\n" + "=" * 70)
    print(f"STARTING OPTUNA STUDY: XGBOOST ({n_trials} Trials)")
    print("=" * 70)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="xgboost_optimization")

    trial_records = []

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 220, step=20),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.25, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "tree_method": "hist"
        }

        t_start = time.time()
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_train, y_train)
        fit_duration = time.time() - t_start

        preds = clf.predict(X_val)
        macro_f1 = f1_score(y_val, preds, average="macro", zero_division=0)

        trial_records.append({
            "model": "XGBoost",
            "trial_number": trial.number,
            "macro_f1": float(macro_f1),
            "fit_time_sec": round(fit_duration, 2),
            **params
        })

        if trial.number % 5 == 0 or trial.number == n_trials - 1:
            print(f"[XGBoost Trial {trial.number:2d}/{n_trials}] Val Macro F1: {macro_f1:.6f} (Fit: {fit_duration:.1f}s)")

        return macro_f1

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    total_opt_time = time.time() - t0

    print(f"\nXGBoost Optimization Finished in {total_opt_time:.1f}s")
    print(f"Best Trial: #{study.best_trial.number} with Val Macro F1 = {study.best_value:.6f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study, trial_records, total_opt_time


def optimize_lightgbm(X_train, y_train, X_val, y_val, n_trials=N_TRIALS_PER_MODEL):
    """Run Optuna study for LightGBM maximizing Validation Macro F1."""
    print("\n" + "=" * 70)
    print(f"STARTING OPTUNA STUDY: LIGHTGBM ({n_trials} Trials)")
    print("=" * 70)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="lightgbm_optimization")

    trial_records = []

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 220, step=20),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100, step=10),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.25, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60, step=10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "objective": "multiclass",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1
        }

        t_start = time.time()
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_train, y_train)
        fit_duration = time.time() - t_start

        preds = clf.predict(X_val)
        macro_f1 = f1_score(y_val, preds, average="macro", zero_division=0)

        trial_records.append({
            "model": "LightGBM",
            "trial_number": trial.number,
            "macro_f1": float(macro_f1),
            "fit_time_sec": round(fit_duration, 2),
            **params
        })

        if trial.number % 5 == 0 or trial.number == n_trials - 1:
            print(f"[LightGBM Trial {trial.number:2d}/{n_trials}] Val Macro F1: {macro_f1:.6f} (Fit: {fit_duration:.1f}s)")

        return macro_f1

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    total_opt_time = time.time() - t0

    print(f"\nLightGBM Optimization Finished in {total_opt_time:.1f}s")
    print(f"Best Trial: #{study.best_trial.number} with Val Macro F1 = {study.best_value:.6f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study, trial_records, total_opt_time


def evaluate_final_tuned_model(model_name, model, X_train, y_train, X_val, y_val, classes):
    """Train the final tuned model and calculate full metrics."""
    print(f"\nEvaluating final tuned {model_name}...")
    t_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t_start

    t_infer_start = time.time()
    val_preds = model.predict(X_val)
    if val_preds.ndim > 1 and val_preds.shape[1] > 1:
        val_preds = np.argmax(val_preds, axis=1)
    val_preds = np.asarray(val_preds).ravel().astype(int)
    val_infer_time = time.time() - t_infer_start

    # Train predictions
    train_preds = model.predict(X_train)
    if train_preds.ndim > 1 and train_preds.shape[1] > 1:
        train_preds = np.argmax(train_preds, axis=1)
    train_preds = np.asarray(train_preds).ravel().astype(int)

    train_metrics = compute_metrics(y_train, train_preds, train_time, 0.0)
    val_metrics = compute_metrics(y_val, val_preds, train_time, val_infer_time)
    val_report = classification_report(y_val, val_preds, target_names=classes, zero_division=0, output_dict=True)

    return {
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "validation_report": val_report,
        "fitted_model": model
    }


def plot_optimization_history(all_trials_df):
    """Plot Optuna optimization convergence history for all models."""
    plt.figure(figsize=(11, 5.5), dpi=300)
    sns.set_theme(style="whitegrid")

    for model_name in all_trials_df["model"].unique():
        sub_df = all_trials_df[all_trials_df["model"] == model_name].sort_values("trial_number")
        running_best = sub_df["macro_f1"].cummax()
        plt.plot(sub_df["trial_number"], sub_df["macro_f1"], "o--", alpha=0.4, label=f"{model_name} (Trial Score)")
        plt.plot(sub_df["trial_number"], running_best, "-", linewidth=2.5, label=f"{model_name} (Running Best Macro F1)")

    plt.title("TectonicSense Phase 2: Optuna Hyperparameter Optimization Trajectory", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Trial Number", fontsize=11, fontweight="semibold")
    plt.ylabel("Validation Macro F1 Score", fontsize=11, fontweight="semibold")
    plt.legend(frameon=True, facecolor="white", edgecolor="gray")
    plt.tight_layout()

    fig_path = TUNING_DIR / "optimization_history.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved optimization history figure: {fig_path}")


def main():
    TUNING_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, feature_cols, classes, le = load_data()

    # Load untuned advanced results for comparison
    untuned_results = {}
    if ADVANCED_RESULTS_JSON.exists():
        with open(ADVANCED_RESULTS_JSON, "r") as f:
            untuned_results = json.load(f)

    # 1. Run XGBoost Optimization
    xgb_study, xgb_trials, xgb_duration = optimize_xgboost(X_train, y_train, X_val, y_val, n_trials=N_TRIALS_PER_MODEL)

    # 2. Run LightGBM Optimization
    lgb_study, lgb_trials, lgb_duration = optimize_lightgbm(X_train, y_train, X_val, y_val, n_trials=N_TRIALS_PER_MODEL)

    # Combine trials
    all_trials = xgb_trials + lgb_trials
    trials_df = pd.DataFrame(all_trials)
    trials_csv_path = TUNING_DIR / "optuna_results.csv"
    trials_df.to_csv(trials_csv_path, index=False)
    print(f"\nSaved trial history CSV: {trials_csv_path}")

    # Plot history
    plot_optimization_history(trials_df)

    # Best parameters dictionary
    best_params = {
        "XGBoost": {
            "best_params": xgb_study.best_params,
            "best_val_macro_f1": round(float(xgb_study.best_value), 6),
            "best_trial_number": int(xgb_study.best_trial.number),
            "total_trials": len(xgb_study.trials),
            "optimization_time_sec": round(xgb_duration, 2)
        },
        "LightGBM": {
            "best_params": lgb_study.best_params,
            "best_val_macro_f1": round(float(lgb_study.best_value), 6),
            "best_trial_number": int(lgb_study.best_trial.number),
            "total_trials": len(lgb_study.trials),
            "optimization_time_sec": round(lgb_duration, 2)
        }
    }
    best_params_path = TUNING_DIR / "best_parameters.json"
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"Saved best parameters JSON: {best_params_path}")

    # Retrain tuned models with best parameters
    print("\n" + "=" * 70)
    print("RETRAINING TUNED MODELS WITH BEST HYPERPARAMETERS")
    print("=" * 70)

    # Tuned XGBoost
    best_xgb_clf = xgb.XGBClassifier(
        **xgb_study.best_params,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist"
    )
    tuned_xgb_res = evaluate_final_tuned_model("XGBoost (Tuned)", best_xgb_clf, X_train, y_train, X_val, y_val, classes)

    # Tuned LightGBM
    best_lgb_clf = lgb.LGBMClassifier(
        **lgb_study.best_params,
        objective="multiclass",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    tuned_lgb_res = evaluate_final_tuned_model("LightGBM (Tuned)", best_lgb_clf, X_train, y_train, X_val, y_val, classes)

    # Build Tuned vs Untuned Comparison Table
    comparison_rows = []

    # Untuned XGBoost
    if "XGBoost" in untuned_results:
        u_xgb = untuned_results["XGBoost"]["validation"]
        comparison_rows.append({
            "Model": "XGBoost (Default)",
            "Status": "Untuned",
            "Accuracy": u_xgb["accuracy"],
            "Macro Precision": u_xgb["macro_precision"],
            "Macro Recall": u_xgb["macro_recall"],
            "Macro F1": u_xgb["macro_f1"],
            "Weighted F1": u_xgb["weighted_f1"],
            "Training Time": u_xgb["training_time_sec"],
            "Inference Time": u_xgb["inference_time_sec"]
        })

    # Tuned XGBoost
    comparison_rows.append({
        "Model": "XGBoost (Tuned)",
        "Status": "Tuned (Optuna)",
        "Accuracy": tuned_xgb_res["validation_metrics"]["accuracy"],
        "Macro Precision": tuned_xgb_res["validation_metrics"]["macro_precision"],
        "Macro Recall": tuned_xgb_res["validation_metrics"]["macro_recall"],
        "Macro F1": tuned_xgb_res["validation_metrics"]["macro_f1"],
        "Weighted F1": tuned_xgb_res["validation_metrics"]["weighted_f1"],
        "Training Time": tuned_xgb_res["validation_metrics"]["training_time_sec"],
        "Inference Time": tuned_xgb_res["validation_metrics"]["inference_time_sec"]
    })

    # Untuned LightGBM
    if "LightGBM" in untuned_results:
        u_lgb = untuned_results["LightGBM"]["validation"]
        comparison_rows.append({
            "Model": "LightGBM (Default)",
            "Status": "Untuned",
            "Accuracy": u_lgb["accuracy"],
            "Macro Precision": u_lgb["macro_precision"],
            "Macro Recall": u_lgb["macro_recall"],
            "Macro F1": u_lgb["macro_f1"],
            "Weighted F1": u_lgb["weighted_f1"],
            "Training Time": u_lgb["training_time_sec"],
            "Inference Time": u_lgb["inference_time_sec"]
        })

    # Tuned LightGBM
    comparison_rows.append({
        "Model": "LightGBM (Tuned)",
        "Status": "Tuned (Optuna)",
        "Accuracy": tuned_lgb_res["validation_metrics"]["accuracy"],
        "Macro Precision": tuned_lgb_res["validation_metrics"]["macro_precision"],
        "Macro Recall": tuned_lgb_res["validation_metrics"]["macro_recall"],
        "Macro F1": tuned_lgb_res["validation_metrics"]["macro_f1"],
        "Weighted F1": tuned_lgb_res["validation_metrics"]["weighted_f1"],
        "Training Time": tuned_lgb_res["validation_metrics"]["training_time_sec"],
        "Inference Time": tuned_lgb_res["validation_metrics"]["inference_time_sec"]
    })

    tuned_df = pd.DataFrame(comparison_rows).sort_values("Macro F1", ascending=False).reset_index(drop=True)
    tuned_csv_path = TUNING_DIR / "tuned_vs_untuned_comparison.csv"
    tuned_df.to_csv(tuned_csv_path, index=False)

    print("\n" + "=" * 70)
    print("TUNED VS UNTUNED COMPARISON (VALIDATION)")
    print("=" * 70)
    print(tuned_df.to_string(index=False))

    # Optimization summary JSON
    summary_data = {
        "trials_executed_per_model": N_TRIALS_PER_MODEL,
        "total_trials_executed": len(all_trials),
        "primary_metric": "Macro F1 (Validation)",
        "best_parameters": best_params,
        "tuned_models_evaluation": {
            "XGBoost_Tuned": {
                "train_metrics": tuned_xgb_res["train_metrics"],
                "validation_metrics": tuned_xgb_res["validation_metrics"],
                "validation_report": tuned_xgb_res["validation_report"]
            },
            "LightGBM_Tuned": {
                "train_metrics": tuned_lgb_res["train_metrics"],
                "validation_metrics": tuned_lgb_res["validation_metrics"],
                "validation_report": tuned_lgb_res["validation_report"]
            }
        },
        "tuned_vs_untuned_summary": tuned_df.to_dict(orient="records")
    }

    summary_path = TUNING_DIR / "optimization_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"\nSaved optimization summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
