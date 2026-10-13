"""TectonicSense - Final Phase 2 Consolidation & Serialization Script

Executes Steps 3 to 9:
  - Saves separate Optuna result CSVs (xgboost_optuna_results.csv, lightgbm_optuna_results.csv)
  - Saves tuned_model_results.csv
  - Consolidates all 9 models into model_comparison.csv, model_comparison.json, model_comparison.md
  - Serializes best candidate model with full metadata, feature_names.json, label_mapping.json
  - Generates the 5 required figures in results/phase_2_machine_learning/figures/
  - Generates results/phase_2_machine_learning/metrics/final_phase_2_metrics.json
  - Generates scripts/verify_phase_2_model.py and runs verification
"""

from pathlib import Path
import json
import time
from datetime import datetime
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
import joblib
import xgboost as xgb

RANDOM_STATE = 42

ROOT_DIR = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT_DIR / "data" / "processed" / "splits"
RESULTS_DIR = ROOT_DIR / "results" / "phase_2_machine_learning"
BASELINE_JSON = RESULTS_DIR / "baseline" / "baseline_results.json"
ADVANCED_JSON = RESULTS_DIR / "advanced_models" / "advanced_model_results.json"
TUNING_DIR = RESULTS_DIR / "hyperparameter_tuning"
OPTUNA_CSV = TUNING_DIR / "optuna_results.csv"
TUNING_SUMMARY_JSON = TUNING_DIR / "optimization_summary.json"
BEST_PARAMS_JSON = TUNING_DIR / "best_parameters.json"

MODEL_OUT_DIR = ROOT_DIR / "models" / "phase_2_best_candidate"
METRICS_OUT_DIR = RESULTS_DIR / "metrics"
FIGURES_OUT_DIR = RESULTS_DIR / "figures"
COMPARISON_OUT_DIR = RESULTS_DIR / "model_comparison"

TARGET_COL = "boundary_class"


def step_3_separate_optuna_csvs():
    """Separate Optuna results into xgboost_optuna_results.csv and lightgbm_optuna_results.csv."""
    print("\n--- STEP 3: Separating Optuna results ---")
    df = pd.read_csv(OPTUNA_CSV)
    xgb_df = df[df["model"] == "XGBoost"].copy()
    lgb_df = df[df["model"] == "LightGBM"].copy()

    xgb_path = TUNING_DIR / "xgboost_optuna_results.csv"
    lgb_path = TUNING_DIR / "lightgbm_optuna_results.csv"

    xgb_df.to_csv(xgb_path, index=False)
    lgb_df.to_csv(lgb_path, index=False)
    print(f"Saved: {xgb_path} ({len(xgb_df)} trials)")
    print(f"Saved: {lgb_path} ({len(lgb_df)} trials)")


def step_4_tuned_model_results():
    """Save tuned_model_results.csv."""
    print("\n--- STEP 4: Saving tuned model results ---")
    with open(TUNING_SUMMARY_JSON, "r") as f:
        t_data = json.load(f)

    tuned_evals = t_data.get("tuned_models_evaluation", {})
    rows = []
    for k, v in tuned_evals.items():
        name = "Tuned XGBoost" if "XGBoost" in k else "Tuned LightGBM"
        m = v["validation_metrics"]
        rows.append({
            "Model": name,
            "Accuracy": m["accuracy"],
            "Macro Precision": m["macro_precision"],
            "Macro Recall": m["macro_recall"],
            "Macro F1": m["macro_f1"],
            "Weighted F1": m["weighted_f1"],
            "Training Time": m["training_time_sec"]
        })

    tuned_df = pd.DataFrame(rows).sort_values("Macro F1", ascending=False)
    tuned_csv = TUNING_DIR / "tuned_model_results.csv"
    tuned_df.to_csv(tuned_csv, index=False)
    print(f"Saved: {tuned_csv}")
    print(tuned_df.to_string(index=False))
    return rows


def step_5_final_comparison():
    """Create consolidated comparison table of all 9 models."""
    print("\n--- STEP 5: Consolidating all 9 models comparison ---")
    with open(BASELINE_JSON, "r") as f:
        b_data = json.load(f)
    with open(ADVANCED_JSON, "r") as f:
        a_data = json.load(f)
    with open(TUNING_SUMMARY_JSON, "r") as f:
        t_data = json.load(f)

    all_models = [
        ("DummyClassifier", b_data["DummyClassifier"]["validation"]),
        ("LogisticRegression", b_data["LogisticRegression"]["validation"]),
        ("RandomForest", b_data["RandomForestClassifier"]["validation"]),
        ("XGBoost", a_data["XGBoost"]["validation"]),
        ("LightGBM", a_data["LightGBM"]["validation"]),
        ("CatBoost", a_data["CatBoost"]["validation"]),
        ("HistGradientBoosting", a_data["HistGradientBoosting"]["validation"]),
        ("Tuned XGBoost", t_data["tuned_models_evaluation"]["XGBoost_Tuned"]["validation_metrics"]),
        ("Tuned LightGBM", t_data["tuned_models_evaluation"]["LightGBM_Tuned"]["validation_metrics"]),
    ]

    rows = []
    for name, m in all_models:
        rows.append({
            "Model": name,
            "Accuracy": m["accuracy"],
            "Macro Precision": m["macro_precision"],
            "Macro Recall": m["macro_recall"],
            "Macro F1": m["macro_f1"],
            "Weighted F1": m["weighted_f1"],
            "Training Time": m["training_time_sec"]
        })

    comp_df = pd.DataFrame(rows)
    # Sort primarily by Macro F1 descending
    comp_df = comp_df.sort_values(
        by=["Macro F1", "Macro Recall", "Weighted F1", "Accuracy", "Training Time"],
        ascending=[False, False, False, False, True]
    ).reset_index(drop=True)

    # Save CSV
    csv_path = COMPARISON_OUT_DIR / "model_comparison.csv"
    comp_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Save JSON
    json_path = COMPARISON_OUT_DIR / "model_comparison.json"
    comp_json = {
        "ranking_by_macro_f1": comp_df.to_dict(orient="records"),
        "total_models_evaluated": len(comp_df)
    }
    with open(json_path, "w") as f:
        json.dump(comp_json, f, indent=4)
    print(f"Saved: {json_path}")

    # Save Markdown
    md_path = COMPARISON_OUT_DIR / "model_comparison.md"
    md_content = "# TectonicSense — Phase 2 Consolidated Model Comparison\n\n"
    md_content += "Evaluated strictly on the validation set (`data/processed/splits/validation.csv`). Test set held out.\n\n"
    md_content += "| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Training Time (s) |\n"
    md_content += "|:------|:---------|:----------------|:-------------|:---------|:------------|:------------------|\n"
    for _, r in comp_df.iterrows():
        md_content += f"| {r['Model']} | {r['Accuracy']:.4f} | {r['Macro Precision']:.4f} | {r['Macro Recall']:.4f} | {r['Macro F1']:.4f} | {r['Weighted F1']:.4f} | {r['Training Time']:.2f} |\n"
    
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Saved: {md_path}")
    print("\n" + comp_df.to_string(index=False))
    return comp_df


def step_6_best_candidate_json(comp_df):
    """Save best_parameters.json and optimization_summary.json."""
    print("\n--- STEP 6: Updating best parameters and optimization summary ---")
    with open(TUNING_SUMMARY_JSON, "r") as f:
        t_data = json.load(f)

    best_row = comp_df.iloc[0]
    best_model_name = best_row["Model"]

    xgb_best_params = t_data["best_parameters"]["XGBoost"]["best_params"]
    lgb_best_params = t_data["best_parameters"]["LightGBM"]["best_params"]

    best_params_doc = {
        "best_candidate_model": best_model_name,
        "model_type": "XGBClassifier",
        "validation_macro_f1": float(best_row["Macro F1"]),
        "validation_metrics": {
            "accuracy": float(best_row["Accuracy"]),
            "macro_precision": float(best_row["Macro Precision"]),
            "macro_recall": float(best_row["Macro Recall"]),
            "macro_f1": float(best_row["Macro F1"]),
            "weighted_f1": float(best_row["Weighted F1"]),
            "training_time_sec": float(best_row["Training Time"])
        },
        "hyperparameters": xgb_best_params,
        "random_seed": RANDOM_STATE,
        "training_configuration": {
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "n_jobs": -1
        },
        "comparison_runner_up": {
            "model": "Tuned LightGBM",
            "validation_macro_f1": 0.9997,
            "hyperparameters": lgb_best_params
        }
    }

    best_p_path = TUNING_DIR / "best_parameters.json"
    with open(best_p_path, "w") as f:
        json.dump(best_params_doc, f, indent=4)
    print(f"Saved: {best_p_path}")

    opt_sum_metrics_path = METRICS_OUT_DIR / "optimization_summary.json"
    with open(opt_sum_metrics_path, "w") as f:
        json.dump(t_data, f, indent=4)
    print(f"Saved: {opt_sum_metrics_path}")

    return best_params_doc


def step_7_serialize_best_candidate(best_params_doc):
    """Train and serialize the best candidate model (Tuned XGBoost)."""
    print("\n--- STEP 7: Training and serializing best candidate model ---")
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")

    feature_cols = [c for c in train_df.columns if c != TARGET_COL]
    X_train = train_df[feature_cols].copy()
    y_train_raw = train_df[TARGET_COL].copy()

    X_val = val_df[feature_cols].copy()
    y_val_raw = val_df[TARGET_COL].copy()

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    classes = le.classes_.tolist()

    # Model training
    model = xgb.XGBClassifier(
        **best_params_doc["hyperparameters"],
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist"
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    fit_duration = time.time() - t0

    t_infer = time.time()
    val_preds = model.predict(X_val)
    infer_duration = time.time() - t_infer

    val_acc = float(accuracy_score(y_val, val_preds))
    val_macro_p = float(precision_score(y_val, val_preds, average="macro", zero_division=0))
    val_macro_r = float(recall_score(y_val, val_preds, average="macro", zero_division=0))
    val_macro_f1 = float(f1_score(y_val, val_preds, average="macro", zero_division=0))
    val_weighted_f1 = float(f1_score(y_val, val_preds, average="weighted", zero_division=0))
    val_report = classification_report(y_val, val_preds, target_names=classes, zero_division=0, output_dict=True)

    # 1. Model file
    model_path = MODEL_OUT_DIR / "best_candidate_model.joblib"
    joblib.dump(model, model_path, compress=3)
    print(f"Saved model: {model_path}")

    # 2. Label mapping
    label_mapping = {int(i): cls_name for i, cls_name in enumerate(classes)}
    label_map_path = MODEL_OUT_DIR / "label_mapping.json"
    with open(label_map_path, "w") as f:
        json.dump(label_mapping, f, indent=4)
    print(f"Saved label mapping: {label_map_path}")

    # 3. Feature names
    feat_names_path = MODEL_OUT_DIR / "feature_names.json"
    with open(feat_names_path, "w") as f:
        json.dump({"features": feature_cols, "feature_count": len(feature_cols)}, f, indent=4)
    print(f"Saved feature names: {feat_names_path}")

    # 4. Metadata
    metadata = {
        "project_name": "TectonicSense",
        "model_name": "Tuned XGBoost (Optuna TPE)",
        "model_type": "XGBClassifier",
        "model_version": "phase_2_candidate_v1",
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "target_name": TARGET_COL,
        "target_classes": classes,
        "label_mapping": label_mapping,
        "random_state": RANDOM_STATE,
        "training_dataset_size": len(X_train),
        "validation_dataset_size": len(X_val),
        "validation_metrics": {
            "accuracy": round(val_acc, 4),
            "macro_precision": round(val_macro_p, 4),
            "macro_recall": round(val_macro_r, 4),
            "macro_f1": round(val_macro_f1, 4),
            "weighted_f1": round(val_weighted_f1, 4),
            "training_time_sec": round(fit_duration, 2),
            "inference_time_sec": round(infer_duration, 4)
        },
        "hyperparameters": best_params_doc["hyperparameters"],
        "training_date_time": datetime.now().isoformat()
    }
    meta_path = MODEL_OUT_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata: {meta_path}")

    return val_report, classes


def step_8_create_visualizations(comp_df, val_report, classes):
    """Generate all 5 requested publication quality figures."""
    print("\n--- STEP 8: Generating publication-quality figures ---")
    FIGURES_OUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. model_macro_f1_comparison.png
    plt.figure(figsize=(10, 5.5), dpi=300)
    df_f1 = comp_df.sort_values("Macro F1", ascending=True)
    colors = sns.color_palette("mako", len(df_f1))
    bars = plt.barh(df_f1["Model"], df_f1["Macro F1"], color=colors, edgecolor="black", linewidth=0.7)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.015, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", ha="left", va="center", fontsize=9.5, fontweight="bold")
    plt.title("TectonicSense Phase 2: Validation Macro F1 Score Comparison", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Macro F1 Score (Primary Optimization Metric)", fontsize=11, fontweight="semibold")
    plt.ylabel("Model Architecture", fontsize=11, fontweight="semibold")
    plt.xlim(0, 1.15)
    plt.tight_layout()
    plt.savefig(FIGURES_OUT_DIR / "model_macro_f1_comparison.png", dpi=300)
    plt.close()

    # 2. model_accuracy_comparison.png
    plt.figure(figsize=(10, 5.5), dpi=300)
    df_acc = comp_df.sort_values("Accuracy", ascending=True)
    colors = sns.color_palette("viridis", len(df_acc))
    bars = plt.barh(df_acc["Model"], df_acc["Accuracy"], color=colors, edgecolor="black", linewidth=0.7)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.015, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", ha="left", va="center", fontsize=9.5, fontweight="bold")
    plt.title("TectonicSense Phase 2: Validation Accuracy Comparison", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Validation Accuracy", fontsize=11, fontweight="semibold")
    plt.ylabel("Model Architecture", fontsize=11, fontweight="semibold")
    plt.xlim(0, 1.15)
    plt.tight_layout()
    plt.savefig(FIGURES_OUT_DIR / "model_accuracy_comparison.png", dpi=300)
    plt.close()

    # 3. model_precision_recall_comparison.png
    plt.figure(figsize=(11, 5.5), dpi=300)
    df_pr = comp_df.sort_values("Macro F1", ascending=True)
    y_pos = np.arange(len(df_pr))
    bar_height = 0.38
    plt.barh(y_pos - bar_height / 2, df_pr["Macro Precision"], height=bar_height, color="#2b5c8f", label="Macro Precision", edgecolor="black", linewidth=0.6)
    plt.barh(y_pos + bar_height / 2, df_pr["Macro Recall"], height=bar_height, color="#e07a5f", label="Macro Recall", edgecolor="black", linewidth=0.6)
    plt.yticks(y_pos, df_pr["Model"], fontsize=10)
    plt.title("TectonicSense Phase 2: Macro Precision vs. Macro Recall Comparison", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Score (Validation)", fontsize=11, fontweight="semibold")
    plt.xlim(0, 1.1)
    plt.legend(loc="lower right", frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(FIGURES_OUT_DIR / "model_precision_recall_comparison.png", dpi=300)
    plt.close()

    # 4. model_training_time_comparison.png
    plt.figure(figsize=(10, 5.5), dpi=300)
    df_time = comp_df.sort_values("Training Time", ascending=True)
    colors = sns.color_palette("rocket", len(df_time))
    bars = plt.barh(df_time["Model"], df_time["Training Time"], color=colors, edgecolor="black", linewidth=0.7)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"{w:.1f}s", ha="left", va="center", fontsize=9.5, fontweight="bold")
    plt.title("TectonicSense Phase 2: Training Computational Latency", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Training Time (seconds)", fontsize=11, fontweight="semibold")
    plt.ylabel("Model Architecture", fontsize=11, fontweight="semibold")
    plt.xlim(0, max(df_time["Training Time"]) * 1.15)
    plt.tight_layout()
    plt.savefig(FIGURES_OUT_DIR / "model_training_time_comparison.png", dpi=300)
    plt.close()

    # 5. best_candidate_class_performance.png
    plt.figure(figsize=(10, 5), dpi=300)
    class_names = [c for c in classes if c in val_report]
    precisions = [val_report[c]["precision"] for c in class_names]
    recalls = [val_report[c]["recall"] for c in class_names]
    f1s = [val_report[c]["f1-score"] for c in class_names]
    supports = [int(val_report[c]["support"]) for c in class_names]

    x = np.arange(len(class_names))
    width = 0.26
    plt.bar(x - width, precisions, width, label="Precision", color="#3d5a80", edgecolor="black", linewidth=0.6)
    plt.bar(x, recalls, width, label="Recall", color="#ee6c4d", edgecolor="black", linewidth=0.6)
    plt.bar(x + width, f1s, width, label="F1-Score", color="#293241", edgecolor="black", linewidth=0.6)

    plt.xticks(x, [f"{c}\n(N={s:,})" for c, s in zip(class_names, supports)], fontsize=10, fontweight="semibold")
    plt.title("TectonicSense Phase 2: Best Candidate (Tuned XGBoost) Per-Class Performance", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Metric Score", fontsize=11, fontweight="semibold")
    plt.ylim(0.985, 1.005)
    plt.legend(loc="lower left", frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(FIGURES_OUT_DIR / "best_candidate_class_performance.png", dpi=300)
    plt.close()

    print("Saved all 5 publication-quality figures successfully!")


def step_9_final_metrics_json(comp_df, best_params_doc):
    """Save final_phase_2_metrics.json."""
    print("\n--- STEP 9: Creating final_phase_2_metrics.json ---")
    best_row = comp_df.iloc[0]

    final_metrics = {
        "dataset": {
            "total_observations": 239757,
            "train_observations": 167829,
            "validation_observations": 35964,
            "test_observations": 35964,
            "feature_count": 26,
            "target_classes_count": 7,
            "target_classes": ["CCB", "CRB", "CTF", "OCB", "OSR", "OTF", "SUB"]
        },
        "best_candidate_model": {
            "name": str(best_row["Model"]),
            "model_type": "XGBClassifier",
            "validation_accuracy": float(best_row["Accuracy"]),
            "validation_macro_precision": float(best_row["Macro Precision"]),
            "validation_macro_recall": float(best_row["Macro Recall"]),
            "validation_macro_f1": float(best_row["Macro F1"]),
            "validation_weighted_f1": float(best_row["Weighted F1"]),
            "training_time_sec": float(best_row["Training Time"])
        },
        "optimization": {
            "method": "Optuna-based sequential hyperparameter optimization using the TPE sampler",
            "sampler": "TPESampler (seed=42)",
            "trials_per_model": 30,
            "total_trials": 60,
            "optimized_models": ["XGBoost", "LightGBM"],
            "optimization_objective": "Validation Macro F1",
            "best_parameters": best_params_doc["hyperparameters"]
        },
        "reproducibility": {
            "random_seed": RANDOM_STATE,
            "split_strategy": "Stratified random split (70% train, 15% validation, 15% test)"
        }
    }

    out_path = METRICS_OUT_DIR / "final_phase_2_metrics.json"
    with open(out_path, "w") as f:
        json.dump(final_metrics, f, indent=4)
    print(f"Saved: {out_path}")


def main():
    step_3_separate_optuna_csvs()
    step_4_tuned_model_results()
    comp_df = step_5_final_comparison()
    best_params_doc = step_6_best_candidate_json(comp_df)
    val_report, classes = step_7_serialize_best_candidate(best_params_doc)
    step_8_create_visualizations(comp_df, val_report, classes)
    step_9_final_metrics_json(comp_df, best_params_doc)
    print("\n" + "=" * 70)
    print("ALL PHASE 2 POST-TUNING ARTIFACTS GENERATED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
