"""TectonicSense - Phase 3 Final Test Set Evaluation Script

Evaluates the serialized Phase 2 best candidate model (Tuned XGBoost)
on the held-out test dataset (data/processed/splits/test.csv).

Outputs:
  - results/phase_3_evaluation/final_test/final_test_metrics.json
  - results/phase_3_evaluation/final_test/classification_report.csv
  - results/phase_3_evaluation/final_test/confusion_matrix.csv
  - results/phase_3_evaluation/final_test/test_predictions.csv
  - Figures in results/phase_3_evaluation/figures/:
      - final_test_confusion_matrix.png
      - final_test_confusion_matrix_normalized.png
      - final_test_class_metrics.png
      - final_test_precision_recall_by_class.png
      - prediction_confidence_distribution.png
"""

from pathlib import Path
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
import joblib

RANDOM_STATE = 42

ROOT_DIR = Path(__file__).resolve().parents[1]
TEST_CSV = ROOT_DIR / "data" / "processed" / "splits" / "test.csv"
MODEL_DIR = ROOT_DIR / "models" / "phase_2_best_candidate"
REFINED_CSV = ROOT_DIR / "data" / "interim" / "earthquakes_pb2002_refined.csv"
ML_DATASET_CSV = ROOT_DIR / "data" / "processed" / "tectonicsense_ml_dataset.csv"

OUT_DIR = ROOT_DIR / "results" / "phase_3_evaluation" / "final_test"
FIG_DIR = ROOT_DIR / "results" / "phase_3_evaluation" / "figures"

TARGET_COL = "boundary_class"


def load_artifacts():
    """Load serialized model, features, label mapping."""
    print("=" * 70)
    print("LOADING PHASE 2 BEST CANDIDATE MODEL & TEST SPLIT")
    print("=" * 70)

    model_path = MODEL_DIR / "best_candidate_model.joblib"
    feat_path = MODEL_DIR / "feature_names.json"
    label_path = MODEL_DIR / "label_mapping.json"

    model = joblib.load(model_path)
    with open(feat_path, "r") as f:
        features = json.load(f)["features"]
    with open(label_path, "r") as f:
        label_mapping = json.load(f)

    test_df = pd.read_csv(TEST_CSV)
    print(f"Loaded model: {type(model).__name__}")
    print(f"Features: {len(features)}")
    print(f"Test samples: {len(test_df):,}")

    return model, features, label_mapping, test_df


def get_test_event_ids(test_df):
    """Reconstruct exact event_ids from refined dataset matching test split."""
    if not REFINED_CSV.exists() or not ML_DATASET_CSV.exists():
        return [f"test_event_{i:06d}" for i in range(len(test_df))]

    ml_df = pd.read_csv(ML_DATASET_CSV, usecols=[TARGET_COL])
    train_idx, temp_idx = train_test_split(
        ml_df.index,
        test_size=0.30,
        stratify=ml_df[TARGET_COL],
        random_state=RANDOM_STATE
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=ml_df.loc[temp_idx, TARGET_COL],
        random_state=RANDOM_STATE
    )

    refined_df = pd.read_csv(REFINED_CSV, usecols=["event_id"])
    event_ids = refined_df.loc[test_idx, "event_id"].values
    return event_ids


def evaluate_test(model, features, label_mapping, test_df):
    """Run test evaluation and compute all classification metrics."""
    X_test = test_df[features].copy()
    y_test_str = test_df[TARGET_COL].copy()

    # Map target strings to integers
    class_to_idx = {v: int(k) for k, v in label_mapping.items()}
    idx_to_class = {int(k): v for k, v in label_mapping.items()}
    classes_ordered = [idx_to_class[i] for i in range(len(label_mapping))]

    y_test_int = y_test_str.map(class_to_idx).values

    print("\nExecuting model prediction on test set...")
    t_start = time.time()
    preds_raw = model.predict(X_test)
    probs_raw = model.predict_proba(X_test)
    infer_time = time.time() - t_start

    if preds_raw.ndim > 1 and preds_raw.shape[1] > 1:
        preds_int = np.argmax(preds_raw, axis=1)
    else:
        preds_int = np.asarray(preds_raw).ravel().astype(int)

    preds_str = [idx_to_class[i] for i in preds_int]
    confidence = np.max(probs_raw, axis=1)

    # Core Metrics
    acc = float(accuracy_score(y_test_int, preds_int))
    macro_p = float(precision_score(y_test_int, preds_int, average="macro", zero_division=0))
    macro_r = float(recall_score(y_test_int, preds_int, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_test_int, preds_int, average="macro", zero_division=0))

    weighted_p = float(precision_score(y_test_int, preds_int, average="weighted", zero_division=0))
    weighted_r = float(recall_score(y_test_int, preds_int, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(y_test_int, preds_int, average="weighted", zero_division=0))

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION RESULTS")
    print("=" * 70)
    print(f"Test Accuracy:       {acc:.6f}")
    print(f"Test Macro Precision:{macro_p:.6f}")
    print(f"Test Macro Recall:   {macro_r:.6f}")
    print(f"Test Macro F1:       {macro_f1:.6f}")
    print(f"Test Weighted F1:    {weighted_f1:.6f}")
    print(f"Inference Duration:  {infer_time:.4f}s ({infer_time/len(test_df)*1e6:.2f} us/event)")

    # Per-class classification report
    report_dict = classification_report(
        y_test_int,
        preds_int,
        target_names=classes_ordered,
        zero_division=0,
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()

    # Confusion matrix
    cm = confusion_matrix(y_test_int, preds_int, labels=list(range(len(classes_ordered))))
    cm_df = pd.DataFrame(cm, index=classes_ordered, columns=classes_ordered)

    # Predictions DataFrame
    event_ids = get_test_event_ids(test_df)
    preds_df = pd.DataFrame({
        "event_id": event_ids,
        "true_label": y_test_str.values,
        "predicted_label": preds_str,
        "confidence": confidence.round(6),
        "is_correct": (y_test_str.values == preds_str)
    })

    metrics_summary = {
        "test_samples": len(test_df),
        "inference_time_sec": round(infer_time, 4),
        "latency_microseconds_per_sample": round(infer_time / len(test_df) * 1e6, 2),
        "accuracy": round(acc, 6),
        "macro_precision": round(macro_p, 6),
        "macro_recall": round(macro_r, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_precision": round(weighted_p, 6),
        "weighted_recall": round(weighted_r, 6),
        "weighted_f1": round(weighted_f1, 6),
        "per_class_metrics": {
            cls: {
                "precision": round(report_dict[cls]["precision"], 6),
                "recall": round(report_dict[cls]["recall"], 6),
                "f1_score": round(report_dict[cls]["f1-score"], 6),
                "support": int(report_dict[cls]["support"])
            }
            for cls in classes_ordered
        }
    }

    return metrics_summary, report_df, cm_df, preds_df, classes_ordered, cm, probs_raw


def generate_visualizations(cm, classes_ordered, report_df, preds_df):
    """Generate publication-quality figures for final test set."""
    print("\nGenerating final test set publication figures...")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Confusion Matrix (Raw counts)
    plt.figure(figsize=(8.5, 7), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes_ordered,
        yticklabels=classes_ordered,
        cbar=True,
        linewidths=0.5
    )
    plt.title("TectonicSense Final Test: Confusion Matrix (Raw Counts)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Tectonic Boundary Class", fontsize=11, fontweight="semibold")
    plt.ylabel("True Tectonic Boundary Class", fontsize=11, fontweight="semibold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "final_test_confusion_matrix.png", dpi=300)
    plt.close()

    # 2. Normalized Confusion Matrix
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8.5, 7), dpi=300)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".4f",
        cmap="Blues",
        xticklabels=classes_ordered,
        yticklabels=classes_ordered,
        cbar=True,
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5
    )
    plt.title("TectonicSense Final Test: Normalized Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Tectonic Boundary Class", fontsize=11, fontweight="semibold")
    plt.ylabel("True Tectonic Boundary Class", fontsize=11, fontweight="semibold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "final_test_confusion_matrix_normalized.png", dpi=300)
    plt.close()

    # 3. Per-Class F1 Score Comparison
    plt.figure(figsize=(9, 5), dpi=300)
    class_f1s = [report_df.loc[c, "f1-score"] for c in classes_ordered]
    bars = plt.bar(classes_ordered, class_f1s, color="#2b5c8f", edgecolor="black", linewidth=0.7, width=0.55)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.0005, f"{h:.4f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    plt.title("TectonicSense Final Test: Per-Class F1 Score", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("PB2002 Tectonic Boundary Class", fontsize=11, fontweight="semibold")
    plt.ylabel("F1 Score", fontsize=11, fontweight="semibold")
    plt.ylim(0.99, 1.002)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "final_test_class_metrics.png", dpi=300)
    plt.close()

    # 4. Per-Class Precision vs Recall
    plt.figure(figsize=(10, 5.2), dpi=300)
    x = np.arange(len(classes_ordered))
    width = 0.35
    prec = [report_df.loc[c, "precision"] for c in classes_ordered]
    rec = [report_df.loc[c, "recall"] for c in classes_ordered]

    plt.bar(x - width / 2, prec, width, label="Precision", color="#3d5a80", edgecolor="black", linewidth=0.6)
    plt.bar(x + width / 2, rec, width, label="Recall", color="#ee6c4d", edgecolor="black", linewidth=0.6)
    plt.xticks(x, classes_ordered, fontsize=10, fontweight="semibold")
    plt.title("TectonicSense Final Test: Per-Class Precision vs. Recall", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("PB2002 Tectonic Boundary Class", fontsize=11, fontweight="semibold")
    plt.ylabel("Score", fontsize=11, fontweight="semibold")
    plt.ylim(0.99, 1.002)
    plt.legend(frameon=True, facecolor="white", loc="lower left")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "final_test_precision_recall_by_class.png", dpi=300)
    plt.close()

    # 5. Prediction Confidence Distribution
    plt.figure(figsize=(9, 5), dpi=300)
    sns.histplot(preds_df["confidence"], bins=40, color="#1d3557", kde=True, edgecolor="black")
    plt.title("TectonicSense Final Test: Prediction Confidence Distribution", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Maximum Softmax Predicted Probability", fontsize=11, fontweight="semibold")
    plt.ylabel("Observation Count", fontsize=11, fontweight="semibold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "prediction_confidence_distribution.png", dpi=300)
    plt.close()

    print("Final test figures saved successfully.")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    model, features, label_mapping, test_df = load_artifacts()
    metrics_summary, report_df, cm_df, preds_df, classes_ordered, cm, probs_raw = evaluate_test(
        model, features, label_mapping, test_df
    )

    # Save metrics JSON
    metrics_path = OUT_DIR / "final_test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"\nSaved metrics: {metrics_path}")

    # Save classification report CSV
    report_path = OUT_DIR / "classification_report.csv"
    report_df.to_csv(report_path)
    print(f"Saved classification report: {report_path}")

    # Save confusion matrix CSV
    cm_path = OUT_DIR / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    print(f"Saved confusion matrix: {cm_path}")

    # Save predictions CSV
    preds_path = OUT_DIR / "test_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    print(f"Saved predictions: {preds_path}")

    # Visualizations
    generate_visualizations(cm, classes_ordered, report_df, preds_df)

    print("\n" + "=" * 70)
    print("STEP 3A & 3B COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
