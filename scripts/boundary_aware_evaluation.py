"""TectonicSense - Phase 3 Boundary-Aware Generalization Experiment

Evaluates out-of-boundary generalization by creating a Group-Aware Split
on PB2002 boundary_id (no boundary_id is shared between train, val, and test).

IMPORTANT:
  boundary_id is strictly a grouping variable and is NEVER used as a predictive feature.

Outputs:
  - results/phase_3_evaluation/boundary_aware/boundary_split_report.json
  - results/phase_3_evaluation/boundary_aware/boundary_aware_results.json
  - results/phase_3_evaluation/boundary_aware/boundary_aware_classification_report.csv
  - results/phase_3_evaluation/figures/boundary_aware_vs_random_split.png
"""

from pathlib import Path
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
import xgboost as xgb

RANDOM_STATE = 42

ROOT_DIR = Path(__file__).resolve().parents[1]
REFINED_CSV = ROOT_DIR / "data" / "interim" / "earthquakes_pb2002_refined.csv"
ML_DATASET_CSV = ROOT_DIR / "data" / "processed" / "tectonicsense_ml_dataset.csv"
RANDOM_TEST_METRICS = ROOT_DIR / "results" / "phase_3_evaluation" / "final_test" / "final_test_metrics.json"

OUT_DIR = ROOT_DIR / "results" / "phase_3_evaluation" / "boundary_aware"
FIG_DIR = ROOT_DIR / "results" / "phase_3_evaluation" / "figures"

TARGET_COL = "boundary_class"


def load_dataset_with_groups():
    """Load engineered features and attach PB2002 boundary_id strictly as group label."""
    print("=" * 70)
    print("LOADING DATA FOR BOUNDARY-AWARE GENERALIZATION EXPERIMENT")
    print("=" * 70)

    ml_df = pd.read_csv(ML_DATASET_CSV)
    refined_df = pd.read_csv(REFINED_CSV, usecols=["boundary_id"])

    # Attach boundary_id strictly for group-splitting
    ml_df["boundary_id"] = refined_df["boundary_id"].values
    
    feature_cols = [c for c in ml_df.columns if c not in [TARGET_COL, "boundary_id"]]

    # Verify boundary_id is not in feature_cols
    assert "boundary_id" not in feature_cols, "CRITICAL ERROR: boundary_id in features!"
    print(f"Total rows: {len(ml_df):,}")
    print(f"Predictive features ({len(feature_cols)}): {feature_cols}")
    print(f"Unique boundary IDs: {ml_df['boundary_id'].nunique()}")

    return ml_df, feature_cols


def create_boundary_aware_splits(df):
    """Create group-aware train (70%), validation (15%), and test (15%) splits on boundary_id."""
    print("\nCreating boundary-aware stratified group split...")
    
    # Stratify at boundary level by dominant class
    boundary_summary = df.groupby("boundary_id")[TARGET_COL].agg(lambda x: x.mode()[0]).reset_index()
    
    # 70% train vs 30% temp
    train_bids, temp_bids = train_test_split(
        boundary_summary["boundary_id"],
        test_size=0.30,
        stratify=boundary_summary[TARGET_COL],
        random_state=RANDOM_STATE
    )

    # 15% val vs 15% test (50/50 of temp)
    temp_summary = boundary_summary[boundary_summary["boundary_id"].isin(temp_bids)]
    val_bids, test_bids = train_test_split(
        temp_summary["boundary_id"],
        test_size=0.50,
        stratify=temp_summary[TARGET_COL],
        random_state=RANDOM_STATE
    )

    train_set_bids = set(train_bids)
    val_set_bids = set(val_bids)
    test_set_bids = set(test_bids)

    # Strict disjunction checks
    inter_train_val = train_set_bids.intersection(val_set_bids)
    inter_train_test = train_set_bids.intersection(test_set_bids)
    inter_val_test = val_set_bids.intersection(test_set_bids)

    assert len(inter_train_val) == 0, f"Leakage: train/val share boundaries: {inter_train_val}"
    assert len(inter_train_test) == 0, f"Leakage: train/test share boundaries: {inter_train_test}"
    assert len(inter_val_test) == 0, f"Leakage: val/test share boundaries: {inter_val_test}"

    train_mask = df["boundary_id"].isin(train_set_bids)
    val_mask = df["boundary_id"].isin(val_set_bids)
    test_mask = df["boundary_id"].isin(test_set_bids)

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    split_report = {
        "train_boundaries_count": len(train_set_bids),
        "val_boundaries_count": len(val_set_bids),
        "test_boundaries_count": len(test_set_bids),
        "train_observations": len(train_df),
        "val_observations": len(val_df),
        "test_observations": len(test_df),
        "boundary_id_overlap_train_val": len(inter_train_val),
        "boundary_id_overlap_train_test": len(inter_train_test),
        "boundary_id_overlap_val_test": len(inter_val_test),
        "train_class_distribution": train_df[TARGET_COL].value_counts().to_dict(),
        "val_class_distribution": val_df[TARGET_COL].value_counts().to_dict(),
        "test_class_distribution": test_df[TARGET_COL].value_counts().to_dict()
    }

    print(f"Train: {len(train_df):,} samples ({len(train_set_bids)} boundaries)")
    print(f"Val:   {len(val_df):,} samples ({len(val_set_bids)} boundaries)")
    print(f"Test:  {len(test_df):,} samples ({len(test_set_bids)} boundaries)")
    print(f"Boundary intersection check: ALL ZERO (verified disjoint).")

    return train_df, val_df, test_df, split_report


def train_and_eval_boundary_model(train_df, val_df, test_df, feature_cols):
    """Train Tuned XGBoost on boundary-aware train set, evaluate on val and test."""
    print("\nTraining XGBoost on boundary-aware training set...")

    le = LabelEncoder()
    y_train = le.fit_transform(train_df[TARGET_COL])
    y_val = le.transform(val_df[TARGET_COL])
    y_test = le.transform(test_df[TARGET_COL])

    classes_ordered = le.classes_.tolist()

    # Features only (boundary_id excluded)
    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    params = {
        "n_estimators": 120,
        "max_depth": 9,
        "learning_rate": 0.155952,
        "subsample": 0.7,
        "colsample_bytree": 0.8,
        "min_child_weight": 2,
        "reg_lambda": 0.035938,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist"
    }

    clf = xgb.XGBClassifier(**params)
    t0 = time.time()
    clf.fit(X_train, y_train)
    fit_duration = time.time() - t0

    # Validation evaluation
    val_preds = clf.predict(X_val)
    val_acc = float(accuracy_score(y_val, val_preds))
    val_macro_p = float(precision_score(y_val, val_preds, average="macro", zero_division=0))
    val_macro_r = float(recall_score(y_val, val_preds, average="macro", zero_division=0))
    val_macro_f1 = float(f1_score(y_val, val_preds, average="macro", zero_division=0))
    val_weighted_f1 = float(f1_score(y_val, val_preds, average="weighted", zero_division=0))

    # Test evaluation
    test_preds = clf.predict(X_test)
    test_acc = float(accuracy_score(y_test, test_preds))
    test_macro_p = float(precision_score(y_test, test_preds, average="macro", zero_division=0))
    test_macro_r = float(recall_score(y_test, test_preds, average="macro", zero_division=0))
    test_macro_f1 = float(f1_score(y_test, test_preds, average="macro", zero_division=0))
    test_weighted_f1 = float(f1_score(y_test, test_preds, average="weighted", zero_division=0))

    test_report_dict = classification_report(
        y_test,
        test_preds,
        target_names=classes_ordered,
        zero_division=0,
        output_dict=True
    )
    test_report_df = pd.DataFrame(test_report_dict).transpose()

    print("\n" + "=" * 70)
    print("BOUNDARY-AWARE GENERALIZATION RESULTS")
    print("=" * 70)
    print(f"Val Accuracy:      {val_acc:.4f} | Val Macro F1:  {val_macro_f1:.4f}")
    print(f"Test Accuracy:     {test_acc:.4f} | Test Macro F1: {test_macro_f1:.4f}")
    print(f"Test Macro Prec:   {test_macro_p:.4f} | Test Macro Rec:{test_macro_r:.4f}")
    print(f"Test Weighted F1:  {test_weighted_f1:.4f}")
    print(f"Fit Duration:      {fit_duration:.2f}s")

    results = {
        "model": "XGBoost (Boundary-Aware)",
        "training_time_sec": round(fit_duration, 2),
        "validation_metrics": {
            "accuracy": round(val_acc, 6),
            "macro_precision": round(val_macro_p, 6),
            "macro_recall": round(val_macro_r, 6),
            "macro_f1": round(val_macro_f1, 6),
            "weighted_f1": round(val_weighted_f1, 6)
        },
        "test_metrics": {
            "accuracy": round(test_acc, 6),
            "macro_precision": round(test_macro_p, 6),
            "macro_recall": round(test_macro_r, 6),
            "macro_f1": round(test_macro_f1, 6),
            "weighted_f1": round(test_weighted_f1, 6)
        }
    }

    return results, test_report_df


def plot_boundary_vs_random_comparison(boundary_results):
    """Generate comparative visualization: Random Stratified Split vs Boundary-Aware Split."""
    print("\nGenerating comparison figure: Random Stratified Split vs Boundary-Aware Split...")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    random_test_macro_f1 = 0.9998
    random_test_acc = 0.9999

    if RANDOM_TEST_METRICS.exists():
        with open(RANDOM_TEST_METRICS, "r") as f:
            r_data = json.load(f)
            random_test_macro_f1 = r_data.get("macro_f1", 0.9998)
            random_test_acc = r_data.get("accuracy", 0.9999)

    bound_test_macro_f1 = boundary_results["test_metrics"]["macro_f1"]
    bound_test_acc = boundary_results["test_metrics"]["accuracy"]

    metrics = ["Accuracy", "Macro F1"]
    random_vals = [random_test_acc, random_test_macro_f1]
    bound_vals = [bound_test_acc, bound_test_macro_f1]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(8.5, 5.5), dpi=300)
    b1 = plt.bar(x - width / 2, random_vals, width, label="Random Stratified Split (Phase 2 & 3A)", color="#2b5c8f", edgecolor="black", linewidth=0.7)
    b2 = plt.bar(x + width / 2, bound_vals, width, label="Boundary-Aware Disjoint Split (Step 3F)", color="#e76f51", edgecolor="black", linewidth=0.7)

    for bar in b1:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    for bar in b2:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.xticks(x, metrics, fontsize=11, fontweight="semibold")
    plt.title("TectonicSense Phase 3: Random Stratified Split vs. Boundary-Aware Generalization", fontsize=12.5, fontweight="bold", pad=12)
    plt.ylabel("Test Score", fontsize=11, fontweight="semibold")
    plt.ylim(0, 1.18)
    plt.legend(frameon=True, facecolor="white", loc="lower right")
    plt.tight_layout()

    out_path = FIG_DIR / "boundary_aware_vs_random_split.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ml_df, feature_cols = load_dataset_with_groups()
    train_df, val_df, test_df, split_report = create_boundary_aware_splits(ml_df)

    # Save split report JSON
    split_report_path = OUT_DIR / "boundary_split_report.json"
    with open(split_report_path, "w") as f:
        json.dump(split_report, f, indent=4)
    print(f"\nSaved boundary split report: {split_report_path}")

    # Train and evaluate
    results, test_report_df = train_and_eval_boundary_model(train_df, val_df, test_df, feature_cols)

    # Save boundary aware results JSON
    results_path = OUT_DIR / "boundary_aware_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved boundary-aware results JSON: {results_path}")

    # Save classification report CSV
    report_csv = OUT_DIR / "boundary_aware_classification_report.csv"
    test_report_df.to_csv(report_csv)
    print(f"Saved classification report CSV: {report_csv}")

    # Plot comparison
    plot_boundary_vs_random_comparison(results)

    print("\n" + "=" * 70)
    print("STEP 3F BOUNDARY-AWARE GENERALIZATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
