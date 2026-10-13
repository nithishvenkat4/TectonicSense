"""TectonicSense - Best Candidate Model Serialization Script

Phase 2 — Machine Learning
Selects the overall best Phase 2 candidate model based on:
  1. Macro F1 (primary priority)
  2. Macro Recall
  3. Weighted F1
  4. Accuracy
  5. Training Latency / Computational Cost

Packages candidate model and exports:
  - models/phase_2_best_candidate/best_candidate_model.joblib
  - models/phase_2_best_candidate/model_metadata.json
  - models/phase_2_best_candidate/feature_names.json
  - models/phase_2_best_candidate/target_class_mapping.json
  - results/phase_2_machine_learning/metrics/best_candidate_validation_metrics.json
  - results/phase_2_machine_learning/metrics/final_model_selection_comparison.csv
"""

from pathlib import Path
import json
import time
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
import joblib

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

RANDOM_STATE = 42

ROOT_DIR = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT_DIR / "data" / "processed" / "splits"
BASELINE_JSON = ROOT_DIR / "results" / "phase_2_machine_learning" / "baseline" / "baseline_results.json"
ADVANCED_JSON = ROOT_DIR / "results" / "phase_2_machine_learning" / "advanced_models" / "advanced_model_results.json"
TUNING_SUMMARY_JSON = ROOT_DIR / "results" / "phase_2_machine_learning" / "hyperparameter_tuning" / "optimization_summary.json"
BEST_PARAMS_JSON = ROOT_DIR / "results" / "phase_2_machine_learning" / "hyperparameter_tuning" / "best_parameters.json"

MODEL_OUT_DIR = ROOT_DIR / "models" / "phase_2_best_candidate"
METRICS_OUT_DIR = ROOT_DIR / "results" / "phase_2_machine_learning" / "metrics"

TARGET_COL = "boundary_class"


def load_data():
    """Load train and val data."""
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")

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
    return X_train, y_train, X_val, y_val, feature_cols, classes, le


def main():
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SELECTING AND SERIALIZING BEST PHASE 2 CANDIDATE MODEL")
    print("=" * 70)

    # 1. Gather all candidates (Baselines, Advanced, Tuned)
    all_rows = []

    # Baselines
    if BASELINE_JSON.exists():
        with open(BASELINE_JSON, "r") as f:
            b_data = json.load(f)
            for m_name, d in b_data.items():
                val = d["validation"]
                all_rows.append({
                    "Model": m_name,
                    "Category": "Baseline",
                    "Accuracy": val["accuracy"],
                    "Macro Precision": val["macro_precision"],
                    "Macro Recall": val["macro_recall"],
                    "Macro F1": val["macro_f1"],
                    "Weighted F1": val["weighted_f1"],
                    "Training Time": val["training_time_sec"],
                    "Inference Time": val["inference_time_sec"]
                })

    # Advanced
    if ADVANCED_JSON.exists():
        with open(ADVANCED_JSON, "r") as f:
            a_data = json.load(f)
            for m_name, d in a_data.items():
                val = d["validation"]
                all_rows.append({
                    "Model": m_name,
                    "Category": "Advanced Untuned",
                    "Accuracy": val["accuracy"],
                    "Macro Precision": val["macro_precision"],
                    "Macro Recall": val["macro_recall"],
                    "Macro F1": val["macro_f1"],
                    "Weighted F1": val["weighted_f1"],
                    "Training Time": val["training_time_sec"],
                    "Inference Time": val["inference_time_sec"]
                })

    # Tuned
    tuned_models_info = {}
    if TUNING_SUMMARY_JSON.exists():
        with open(TUNING_SUMMARY_JSON, "r") as f:
            t_data = json.load(f)
            tuned_evals = t_data.get("tuned_models_evaluation", {})
            for t_name, d in tuned_evals.items():
                val = d["validation_metrics"]
                display_name = t_name.replace("_", " ")
                all_rows.append({
                    "Model": display_name,
                    "Category": "Tuned (Optuna)",
                    "Accuracy": val["accuracy"],
                    "Macro Precision": val["macro_precision"],
                    "Macro Recall": val["macro_recall"],
                    "Macro F1": val["macro_f1"],
                    "Weighted F1": val["weighted_f1"],
                    "Training Time": val["training_time_sec"],
                    "Inference Time": val["inference_time_sec"]
                })
                tuned_models_info[t_name] = d

    comparison_df = pd.DataFrame(all_rows)
    comparison_df = comparison_df.sort_values(
        by=["Macro F1", "Macro Recall", "Weighted F1", "Accuracy", "Training Time"],
        ascending=[False, False, False, False, True]
    ).reset_index(drop=True)

    comp_csv = METRICS_OUT_DIR / "final_model_selection_comparison.csv"
    comparison_df.to_csv(comp_csv, index=False)
    print("\nFINAL MULTI-MODEL RANKING TABLE (VALIDATION SET):")
    print(comparison_df.to_string(index=False))

    best_record = comparison_df.iloc[0]
    best_model_name = best_record["Model"]
    print(f"\n>>> BEST PHASE 2 CANDIDATE: {best_model_name} (Macro F1 = {best_record['Macro F1']:.4f})")

    # Load data for final fitting and serialization
    X_train, y_train, X_val, y_val, feature_cols, classes, le = load_data()

    # Load best hyperparameters if available
    best_params = {}
    if BEST_PARAMS_JSON.exists():
        with open(BEST_PARAMS_JSON, "r") as f:
            best_params = json.load(f)

    # Initialize best model based on selection
    if "XGBoost" in best_model_name:
        params = best_params.get("XGBoost", {}).get("best_params", {})
        candidate_model = xgb.XGBClassifier(
            **params,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist"
        )
        model_type = "XGBoost"
    elif "LightGBM" in best_model_name:
        params = best_params.get("LightGBM", {}).get("best_params", {})
        candidate_model = lgb.LGBMClassifier(
            **params,
            objective="multiclass",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )
        model_type = "LightGBM"
    elif "HistGradientBoosting" in best_model_name:
        params = {"max_iter": 150, "learning_rate": 0.1, "max_leaf_nodes": 31}
        candidate_model = HistGradientBoostingClassifier(
            **params,
            random_state=RANDOM_STATE
        )
        model_type = "HistGradientBoostingClassifier"
    else:
        params = {"n_estimators": 100, "max_depth": 25}
        candidate_model = RandomForestClassifier(
            **params,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model_type = "RandomForestClassifier"

    # Train and evaluate candidate model on train and val
    print(f"\nFitting best candidate model ({model_type}) on training data...")
    t_start = time.time()
    candidate_model.fit(X_train, y_train)
    fit_duration = time.time() - t_start

    t_infer = time.time()
    val_preds = candidate_model.predict(X_val)
    if val_preds.ndim > 1 and val_preds.shape[1] > 1:
        val_preds = np.argmax(val_preds, axis=1)
    val_preds = np.asarray(val_preds).ravel().astype(int)
    infer_duration = time.time() - t_infer

    # Validation metrics
    val_acc = float(accuracy_score(y_val, val_preds))
    val_macro_p = float(precision_score(y_val, val_preds, average="macro", zero_division=0))
    val_macro_r = float(recall_score(y_val, val_preds, average="macro", zero_division=0))
    val_macro_f1 = float(f1_score(y_val, val_preds, average="macro", zero_division=0))
    val_weighted_f1 = float(f1_score(y_val, val_preds, average="weighted", zero_division=0))
    val_report = classification_report(y_val, val_preds, target_names=classes, zero_division=0, output_dict=True)

    val_metrics = {
        "accuracy": round(val_acc, 4),
        "macro_precision": round(val_macro_p, 4),
        "macro_recall": round(val_macro_r, 4),
        "macro_f1": round(val_macro_f1, 4),
        "weighted_f1": round(val_weighted_f1, 4),
        "training_time_sec": round(fit_duration, 2),
        "inference_time_sec": round(infer_duration, 4)
    }

    # Save model artifact
    model_save_path = MODEL_OUT_DIR / "best_candidate_model.joblib"
    joblib.dump(candidate_model, model_save_path, compress=3)
    print(f"Serialized model artifact to: {model_save_path}")

    # Save label encoder
    encoder_save_path = MODEL_OUT_DIR / "label_encoder.joblib"
    joblib.dump(le, encoder_save_path)
    print(f"Saved label encoder to: {encoder_save_path}")

    # Save feature names
    feature_names_path = MODEL_OUT_DIR / "feature_names.json"
    with open(feature_names_path, "w") as f:
        json.dump({"features": feature_cols, "count": len(feature_cols)}, f, indent=4)
    print(f"Saved feature names to: {feature_names_path}")

    # Save target mapping
    mapping = {int(idx): cls_name for idx, cls_name in enumerate(classes)}
    mapping_path = MODEL_OUT_DIR / "target_class_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=4)
    print(f"Saved target class mapping to: {mapping_path}")

    # Save validation metrics
    val_metrics_path = METRICS_OUT_DIR / "best_candidate_validation_metrics.json"
    full_metrics = {
        "model_name": best_model_name,
        "model_type": model_type,
        "metrics": val_metrics,
        "classification_report": val_report
    }
    with open(val_metrics_path, "w") as f:
        json.dump(full_metrics, f, indent=4)
    print(f"Saved best candidate validation metrics to: {val_metrics_path}")

    # Save metadata JSON
    metadata = {
        "model_name": best_model_name,
        "model_type": model_type,
        "status": "BEST PHASE 2 CANDIDATE",
        "random_seed": RANDOM_STATE,
        "features": feature_cols,
        "feature_count": len(feature_cols),
        "target_column": TARGET_COL,
        "target_classes": classes,
        "target_class_mapping": mapping,
        "hyperparameters": params,
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "validation_metrics": val_metrics,
        "rationale": (
            f"{best_model_name} demonstrated the highest Macro F1 score ({val_metrics['macro_f1']:.4f}) "
            f"across all evaluated models, balanced with high precision ({val_metrics['macro_precision']:.4f}) "
            f"and recall ({val_metrics['macro_recall']:.4f}) across all seven imbalanced tectonic classes, "
            f"while maintaining low inference latency ({val_metrics['inference_time_sec']:.4f}s)."
        )
    }
    metadata_path = MODEL_OUT_DIR / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved candidate metadata to: {metadata_path}")

    print("\n" + "=" * 70)
    print("BEST CANDIDATE MODEL SERIALIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
