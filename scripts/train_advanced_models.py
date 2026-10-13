"""TectonicSense - Advanced Models Training Script

Phase 2 — Machine Learning
Evaluates:
  1. XGBoost (XGBClassifier)
  2. LightGBM (LGBMClassifier)
  3. CatBoost (CatBoostClassifier)
  4. HistGradientBoostingClassifier

Outputs:
  - results/phase_2_machine_learning/advanced_models/advanced_model_results.json
  - results/phase_2_machine_learning/advanced_models/advanced_model_comparison.csv
"""

from pathlib import Path
import json
import time
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Deterministic seed
RANDOM_STATE = 42

# Project paths
ROOT_DIR = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT_DIR / "data" / "processed" / "splits"
RESULTS_DIR = ROOT_DIR / "results" / "phase_2_machine_learning" / "advanced_models"

TARGET_COL = "boundary_class"


def load_data():
    """Load train and validation splits."""
    print("=" * 70)
    print("LOADING DATA SPLITS FOR ADVANCED GRADIENT BOOSTING MODELS")
    print("=" * 70)

    train_path = SPLITS_DIR / "train.csv"
    val_path = SPLITS_DIR / "validation.csv"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Missing split files in {SPLITS_DIR}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    print(f"Loaded Train:      {len(train_df):,} rows x {train_df.shape[1]} cols")
    print(f"Loaded Validation: {len(val_df):,} rows x {val_df.shape[1]} cols")

    return train_df, val_df


def prepare_features_and_target(train_df, val_df):
    """Separate features and target, label-encode the multiclass target strictly on train."""
    feature_cols = [c for c in train_df.columns if c != TARGET_COL]

    # Leakage check
    forbidden = ["event_id", "boundary_id", "boundary_step_id"]
    for col in forbidden:
        if col in feature_cols:
            raise ValueError(f"Leakage hazard: '{col}' detected in features!")

    X_train = train_df[feature_cols].copy()
    y_train_raw = train_df[TARGET_COL].copy()

    X_val = val_df[feature_cols].copy()
    y_val_raw = val_df[TARGET_COL].copy()

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)

    classes = le.classes_.tolist()
    print(f"\nTarget classes ({len(classes)}): {classes}")
    for idx, c in enumerate(classes):
        print(f"  Class {idx} -> {c}")

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


def train_and_eval(model_name, model, X_train, y_train, X_val, y_val, classes):
    """Fit model on train, evaluate on train and validation, record times."""
    print(f"\n--- Training {model_name} ---")
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    print(f"Training completed in {train_time:.2f}s")

    # Train inference
    start_infer_train = time.time()
    train_preds = model.predict(X_train)
    if train_preds.ndim > 1 and train_preds.shape[1] > 1:
        train_preds = np.argmax(train_preds, axis=1)
    train_preds = np.asarray(train_preds).ravel().astype(int)
    train_infer_time = time.time() - start_infer_train

    # Val inference
    start_infer_val = time.time()
    val_preds = model.predict(X_val)
    if val_preds.ndim > 1 and val_preds.shape[1] > 1:
        val_preds = np.argmax(val_preds, axis=1)
    val_preds = np.asarray(val_preds).ravel().astype(int)
    val_infer_time = time.time() - start_infer_val
    print(f"Validation inference completed in {val_infer_time:.4f}s")

    train_metrics = compute_metrics(y_train, train_preds, train_time, train_infer_time)
    val_metrics = compute_metrics(y_val, val_preds, train_time, val_infer_time)

    val_report = classification_report(
        y_val,
        val_preds,
        target_names=classes,
        zero_division=0,
        output_dict=True
    )

    print(f"[{model_name}] Val Acc: {val_metrics['accuracy']:.4f} | Val Macro F1: {val_metrics['macro_f1']:.4f} | Val Weighted F1: {val_metrics['weighted_f1']:.4f}")

    return {
        "model_name": model_name,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "validation_classification_report": val_report,
        "fitted_model": model
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_data()
    X_train, y_train, X_val, y_val, feature_cols, classes, le = prepare_features_and_target(train_df, val_df)

    print(f"\nPredictive features count: {len(feature_cols)}")

    results = {}
    comparison_rows = []

    # 1. XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist"
    )
    xgb_res = train_and_eval("XGBoost", xgb_model, X_train, y_train, X_val, y_val, classes)
    results["XGBoost"] = {
        "train": xgb_res["train_metrics"],
        "validation": xgb_res["validation_metrics"],
        "class_report": xgb_res["validation_classification_report"]
    }
    comparison_rows.append({
        "Model": "XGBoost",
        "Accuracy": xgb_res["validation_metrics"]["accuracy"],
        "Macro Precision": xgb_res["validation_metrics"]["macro_precision"],
        "Macro Recall": xgb_res["validation_metrics"]["macro_recall"],
        "Macro F1": xgb_res["validation_metrics"]["macro_f1"],
        "Weighted F1": xgb_res["validation_metrics"]["weighted_f1"],
        "Training Time (s)": xgb_res["validation_metrics"]["training_time_sec"],
        "Inference Time (s)": xgb_res["validation_metrics"]["inference_time_sec"]
    })

    # 2. LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.1,
        num_leaves=31,
        objective="multiclass",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    lgb_res = train_and_eval("LightGBM", lgb_model, X_train, y_train, X_val, y_val, classes)
    results["LightGBM"] = {
        "train": lgb_res["train_metrics"],
        "validation": lgb_res["validation_metrics"],
        "class_report": lgb_res["validation_classification_report"]
    }
    comparison_rows.append({
        "Model": "LightGBM",
        "Accuracy": lgb_res["validation_metrics"]["accuracy"],
        "Macro Precision": lgb_res["validation_metrics"]["macro_precision"],
        "Macro Recall": lgb_res["validation_metrics"]["macro_recall"],
        "Macro F1": lgb_res["validation_metrics"]["macro_f1"],
        "Weighted F1": lgb_res["validation_metrics"]["weighted_f1"],
        "Training Time (s)": lgb_res["validation_metrics"]["training_time_sec"],
        "Inference Time (s)": lgb_res["validation_metrics"]["inference_time_sec"]
    })

    # 3. CatBoost
    cb_model = cb.CatBoostClassifier(
        iterations=200,
        learning_rate=0.1,
        depth=6,
        loss_function="MultiClass",
        random_seed=RANDOM_STATE,
        verbose=False,
        thread_count=-1
    )
    cb_res = train_and_eval("CatBoost", cb_model, X_train, y_train, X_val, y_val, classes)
    results["CatBoost"] = {
        "train": cb_res["train_metrics"],
        "validation": cb_res["validation_metrics"],
        "class_report": cb_res["validation_classification_report"]
    }
    comparison_rows.append({
        "Model": "CatBoost",
        "Accuracy": cb_res["validation_metrics"]["accuracy"],
        "Macro Precision": cb_res["validation_metrics"]["macro_precision"],
        "Macro Recall": cb_res["validation_metrics"]["macro_recall"],
        "Macro F1": cb_res["validation_metrics"]["macro_f1"],
        "Weighted F1": cb_res["validation_metrics"]["weighted_f1"],
        "Training Time (s)": cb_res["validation_metrics"]["training_time_sec"],
        "Inference Time (s)": cb_res["validation_metrics"]["inference_time_sec"]
    })

    # 4. HistGradientBoostingClassifier
    hgb_model = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.1,
        max_leaf_nodes=31,
        random_state=RANDOM_STATE
    )
    hgb_res = train_and_eval("HistGradientBoosting", hgb_model, X_train, y_train, X_val, y_val, classes)
    results["HistGradientBoosting"] = {
        "train": hgb_res["train_metrics"],
        "validation": hgb_res["validation_metrics"],
        "class_report": hgb_res["validation_classification_report"]
    }
    comparison_rows.append({
        "Model": "HistGradientBoosting",
        "Accuracy": hgb_res["validation_metrics"]["accuracy"],
        "Macro Precision": hgb_res["validation_metrics"]["macro_precision"],
        "Macro Recall": hgb_res["validation_metrics"]["macro_recall"],
        "Macro F1": hgb_res["validation_metrics"]["macro_f1"],
        "Weighted F1": hgb_res["validation_metrics"]["weighted_f1"],
        "Training Time (s)": hgb_res["validation_metrics"]["training_time_sec"],
        "Inference Time (s)": hgb_res["validation_metrics"]["inference_time_sec"]
    })

    # Save results json
    json_path = RESULTS_DIR / "advanced_model_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved advanced model results json to: {json_path}")

    # Save comparison csv
    comp_df = pd.DataFrame(comparison_rows)
    comp_df = comp_df.sort_values(by="Macro F1", ascending=False)
    csv_path = RESULTS_DIR / "advanced_model_comparison.csv"
    comp_df.to_csv(csv_path, index=False)
    print(f"Saved advanced model comparison csv to: {csv_path}")

    print("\n" + "=" * 70)
    print("ADVANCED MODELS COMPARISON SUMMARY (VALIDATION)")
    print("=" * 70)
    print(comp_df.to_string(index=False))


if __name__ == "__main__":
    main()
