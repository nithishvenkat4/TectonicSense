"""TectonicSense - Baseline Model Training Script

Phase 2 — Machine Learning
Evaluates:
  1. DummyClassifier (sanity check)
  2. LogisticRegression (with StandardScaler in Pipeline)
  3. RandomForestClassifier (nonlinear ensemble)

Outputs:
  - results/phase_2_machine_learning/baseline/baseline_results.json
  - results/phase_2_machine_learning/baseline/baseline_comparison.csv
  - models/phase_2_baseline/random_forest_baseline.joblib
"""

from pathlib import Path
import json
import time
import pandas as pd
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
import joblib

# Deterministic seed
RANDOM_STATE = 42

# Project paths
ROOT_DIR = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT_DIR / "data" / "processed" / "splits"
RESULTS_DIR = ROOT_DIR / "results" / "phase_2_machine_learning" / "baseline"
MODELS_DIR = ROOT_DIR / "models" / "phase_2_baseline"

TARGET_COL = "boundary_class"


def load_data():
    """Load train and validation splits."""
    print("=" * 70)
    print("LOADING DATA SPLITS FOR BASELINE MODELS")
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


def prepare_features(df):
    """Separate features (X) and target (y). Exclude any leakage columns."""
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    
    # Explicitly check for identifiers
    forbidden = ["event_id", "boundary_id", "boundary_step_id"]
    for col in forbidden:
        if col in feature_cols:
            raise ValueError(f"Leakage hazard: '{col}' found in feature list!")

    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    return X, y, feature_cols


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


def train_and_eval(model_name, model, X_train, y_train, X_val, y_val):
    """Fit model on train, evaluate on train and validation, record times."""
    print(f"\n--- Training {model_name} ---")
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    print(f"Training completed in {train_time:.2f}s")

    # Train inference
    start_infer_train = time.time()
    train_preds = model.predict(X_train)
    train_infer_time = time.time() - start_infer_train

    # Val inference
    start_infer_val = time.time()
    val_preds = model.predict(X_val)
    val_infer_time = time.time() - start_infer_val
    print(f"Validation inference completed in {val_infer_time:.4f}s")

    train_metrics = compute_metrics(y_train, train_preds, train_time, train_infer_time)
    val_metrics = compute_metrics(y_val, val_preds, train_time, val_infer_time)

    # Detailed report on validation
    val_report = classification_report(y_val, val_preds, zero_division=0, output_dict=True)

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
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_data()
    X_train, y_train, feature_cols = prepare_features(train_df)
    X_val, y_val, _ = prepare_features(val_df)

    print(f"\nTotal features: {len(feature_cols)}")
    print(f"Features: {feature_cols}")

    results = {}
    comparison_rows = []

    # 1. Dummy Classifier (Baseline Sanity Check)
    dummy_model = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy_res = train_and_eval("DummyClassifier", dummy_model, X_train, y_train, X_val, y_val)
    results["DummyClassifier"] = {
        "train": dummy_res["train_metrics"],
        "validation": dummy_res["validation_metrics"],
        "class_report": dummy_res["validation_classification_report"]
    }
    comparison_rows.append({
        "Model": "DummyClassifier",
        "Accuracy": dummy_res["validation_metrics"]["accuracy"],
        "Macro Precision": dummy_res["validation_metrics"]["macro_precision"],
        "Macro Recall": dummy_res["validation_metrics"]["macro_recall"],
        "Macro F1": dummy_res["validation_metrics"]["macro_f1"],
        "Weighted F1": dummy_res["validation_metrics"]["weighted_f1"],
        "Training Time (s)": dummy_res["validation_metrics"]["training_time_sec"],
        "Inference Time (s)": dummy_res["validation_metrics"]["inference_time_sec"]
    })

    # 2. Logistic Regression (Linear Baseline with StandardScaler Pipeline)
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])
    lr_res = train_and_eval("LogisticRegression", lr_pipeline, X_train, y_train, X_val, y_val)
    results["LogisticRegression"] = {
        "train": lr_res["train_metrics"],
        "validation": lr_res["validation_metrics"],
        "class_report": lr_res["validation_classification_report"]
    }
    comparison_rows.append({
        "Model": "LogisticRegression",
        "Accuracy": lr_res["validation_metrics"]["accuracy"],
        "Macro Precision": lr_res["validation_metrics"]["macro_precision"],
        "Macro Recall": lr_res["validation_metrics"]["macro_recall"],
        "Macro F1": lr_res["validation_metrics"]["macro_f1"],
        "Weighted F1": lr_res["validation_metrics"]["weighted_f1"],
        "Training Time (s)": lr_res["validation_metrics"]["training_time_sec"],
        "Inference Time (s)": lr_res["validation_metrics"]["inference_time_sec"]
    })

    # 3. Random Forest (Nonlinear Ensemble Baseline)
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=25,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_res = train_and_eval("RandomForestClassifier", rf_model, X_train, y_train, X_val, y_val)
    results["RandomForestClassifier"] = {
        "train": rf_res["train_metrics"],
        "validation": rf_res["validation_metrics"],
        "class_report": rf_res["validation_classification_report"]
    }
    comparison_rows.append({
        "Model": "RandomForestClassifier",
        "Accuracy": rf_res["validation_metrics"]["accuracy"],
        "Macro Precision": rf_res["validation_metrics"]["macro_precision"],
        "Macro Recall": rf_res["validation_metrics"]["macro_recall"],
        "Macro F1": rf_res["validation_metrics"]["macro_f1"],
        "Weighted F1": rf_res["validation_metrics"]["weighted_f1"],
        "Training Time (s)": rf_res["validation_metrics"]["training_time_sec"],
        "Inference Time (s)": rf_res["validation_metrics"]["inference_time_sec"]
    })

    # Save RandomForest baseline model
    rf_path = MODELS_DIR / "random_forest_baseline.joblib"
    joblib.dump(rf_res["fitted_model"], rf_path, compress=3)
    print(f"\nSaved RandomForest baseline model to: {rf_path}")

    # Save results json
    json_path = RESULTS_DIR / "baseline_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved baseline results json to: {json_path}")

    # Save comparison csv
    comp_df = pd.DataFrame(comparison_rows)
    comp_df = comp_df.sort_values(by="Macro F1", ascending=False)
    csv_path = RESULTS_DIR / "baseline_comparison.csv"
    comp_df.to_csv(csv_path, index=False)
    print(f"Saved baseline comparison csv to: {csv_path}")

    print("\n" + "=" * 70)
    print("BASELINE MODELS COMPARISON SUMMARY (VALIDATION)")
    print("=" * 70)
    print(comp_df.to_string(index=False))


if __name__ == "__main__":
    main()
