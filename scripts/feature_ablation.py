"""TectonicSense - Phase 3 Feature Group Ablation Study

Evaluates predictive dependence across structured feature subsets:
  - Group 1: Seismic Only (4 features)
  - Group 2: Seismic + Geographic (12 features)
  - Group 3: Seismic + Geographic + Limited Tectonic (21 features)
  - Group 4: Full Feature Set (26 features)
  - PB2002 Metadata Ablation: Seismic + Geographic (Direct PB2002 metadata removed)

Model: Same XGBoost architecture (Tuned parameters from Phase 2)
Training: data/processed/splits/train.csv
Validation: data/processed/splits/validation.csv
Test: data/processed/splits/test.csv

Outputs:
  - results/phase_3_evaluation/ablation/ablation_results.csv
  - results/phase_3_evaluation/ablation/pb2002_metadata_ablation.csv
  - results/phase_3_evaluation/figures/ablation_macro_f1.png
  - results/phase_3_evaluation/figures/ablation_accuracy.png
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
    f1_score
)
import xgboost as xgb

RANDOM_STATE = 42

ROOT_DIR = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT_DIR / "data" / "processed" / "splits"
BEST_PARAMS_JSON = ROOT_DIR / "results" / "phase_2_machine_learning" / "hyperparameter_tuning" / "best_parameters.json"

ABLATION_DIR = ROOT_DIR / "results" / "phase_3_evaluation" / "ablation"
FIG_DIR = ROOT_DIR / "results" / "phase_3_evaluation" / "figures"

TARGET_COL = "boundary_class"


# Feature Group Definitions
GROUP_1_SEISMIC = [
    "magnitude",
    "magnitude_squared",
    "depth_km",
    "depth_abs_km"
]

GROUP_2_SEISMIC_GEOGRAPHIC = GROUP_1_SEISMIC + [
    "longitude",
    "latitude",
    "longitude_sin",
    "longitude_cos",
    "latitude_sin",
    "latitude_cos",
    "boundary_distance_km",
    "log_boundary_distance_km"
]

GROUP_3_LIMITED_TECTONIC = GROUP_2_SEISMIC_GEOGRAPHIC + [
    "boundary_azimuth_sin",
    "boundary_azimuth_cos",
    "relative_velocity",
    "velocity_azimuth_sin",
    "velocity_azimuth_cos",
    "divergent_component",
    "right_lateral_component",
    "tectonic_motion_magnitude",
    "orogen_flag"
]

# Group 4 is Full 26 features
FULL_FEATURES = [
    "magnitude",
    "magnitude_squared",
    "depth_km",
    "depth_abs_km",
    "longitude",
    "latitude",
    "longitude_sin",
    "longitude_cos",
    "latitude_sin",
    "latitude_cos",
    "boundary_distance_km",
    "log_boundary_distance_km",
    "boundary_length_km",
    "boundary_azimuth_sin",
    "boundary_azimuth_cos",
    "relative_velocity",
    "velocity_azimuth_sin",
    "velocity_azimuth_cos",
    "divergent_component",
    "right_lateral_component",
    "absolute_divergent_component",
    "absolute_right_lateral_component",
    "tectonic_motion_magnitude",
    "boundary_elevation_m",
    "seafloor_age_ma",
    "orogen_flag"
]


def load_data():
    """Load train, val, and test splits."""
    print("=" * 70)
    print("LOADING DATA SPLITS FOR ABLATION STUDY")
    print("=" * 70)

    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    le = LabelEncoder()
    y_train = le.fit_transform(train_df[TARGET_COL])
    y_val = le.transform(val_df[TARGET_COL])
    y_test = le.transform(test_df[TARGET_COL])

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"Classes ({len(le.classes_)}): {list(le.classes_)}")

    return train_df, val_df, test_df, y_train, y_val, y_test, le


def get_model_params():
    """Load tuned hyperparameters from Phase 2."""
    if BEST_PARAMS_JSON.exists():
        with open(BEST_PARAMS_JSON, "r") as f:
            data = json.load(f)
            params = data.get("hyperparameters", {})
    else:
        params = {
            "n_estimators": 120,
            "max_depth": 9,
            "learning_rate": 0.155952,
            "subsample": 0.7,
            "colsample_bytree": 0.8,
            "min_child_weight": 2,
            "reg_lambda": 0.035938
        }
    return params


def train_and_evaluate_subset(group_name, feature_list, train_df, val_df, test_df, y_train, y_val, y_test, params):
    """Train XGBoost on specified feature group and evaluate on val and test."""
    print(f"\n--- Training Feature Group: {group_name} ({len(feature_list)} features) ---")
    
    X_train = train_df[feature_list].copy()
    X_val = val_df[feature_list].copy()
    X_test = test_df[feature_list].copy()

    clf = xgb.XGBClassifier(
        **params,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist"
    )

    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

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

    print(f"[{group_name}] Val Macro F1: {val_macro_f1:.4f} | Test Macro F1: {test_macro_f1:.4f} | Test Acc: {test_acc:.4f} (Fit: {train_time:.1f}s)")

    return {
        "Feature Group": group_name,
        "Feature Count": len(feature_list),
        "Val Macro F1": round(val_macro_f1, 6),
        "Accuracy": round(test_acc, 6),
        "Macro Precision": round(test_macro_p, 6),
        "Macro Recall": round(test_macro_r, 6),
        "Macro F1": round(test_macro_f1, 6),
        "Weighted F1": round(test_weighted_f1, 6),
        "Training Time": round(train_time, 2)
    }


def plot_ablation_figures(ablation_df):
    """Plot publication-quality figures comparing ablation groups."""
    print("\nGenerating ablation figures...")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. ablation_macro_f1.png
    plt.figure(figsize=(9.5, 5.5), dpi=300)
    colors = ["#e76f51", "#f4a261", "#2a9d8f", "#264653"]
    bars = plt.bar(ablation_df["Feature Group"], ablation_df["Macro F1"], color=colors, edgecolor="black", linewidth=0.7, width=0.55)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.title("TectonicSense Phase 3: Feature Group Ablation (Macro F1)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Ablation Group", fontsize=11, fontweight="semibold")
    plt.ylabel("Test Macro F1 Score", fontsize=11, fontweight="semibold")
    plt.ylim(0, 1.15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ablation_macro_f1.png", dpi=300)
    plt.close()

    # 2. ablation_accuracy.png
    plt.figure(figsize=(9.5, 5.5), dpi=300)
    bars = plt.bar(ablation_df["Feature Group"], ablation_df["Accuracy"], color=colors, edgecolor="black", linewidth=0.7, width=0.55)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.title("TectonicSense Phase 3: Feature Group Ablation (Accuracy)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Ablation Group", fontsize=11, fontweight="semibold")
    plt.ylabel("Test Accuracy", fontsize=11, fontweight="semibold")
    plt.ylim(0, 1.15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ablation_accuracy.png", dpi=300)
    plt.close()

    print("Saved ablation figures successfully.")


def main():
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df, y_train, y_val, y_test, le = load_data()
    params = get_model_params()

    groups = [
        ("Group 1: Seismic Only", GROUP_1_SEISMIC),
        ("Group 2: Seismic + Geographic", GROUP_2_SEISMIC_GEOGRAPHIC),
        ("Group 3: Limited Tectonic", GROUP_3_LIMITED_TECTONIC),
        ("Group 4: Full Feature Set", FULL_FEATURES)
    ]

    ablation_records = []
    for g_name, g_feats in groups:
        res = train_and_evaluate_subset(
            g_name, g_feats, train_df, val_df, test_df, y_train, y_val, y_test, params
        )
        ablation_records.append(res)

    ablation_df = pd.DataFrame(ablation_records)

    # Save standard ablation results CSV
    ablation_csv = ABLATION_DIR / "ablation_results.csv"
    columns_order = [
        "Feature Group", "Feature Count", "Accuracy",
        "Macro Precision", "Macro Recall", "Macro F1",
        "Weighted F1", "Training Time"
    ]
    ablation_df[columns_order].to_csv(ablation_csv, index=False)
    print(f"\nSaved ablation results CSV: {ablation_csv}")
    print(ablation_df[columns_order].to_string(index=False))

    # Save Step 3E: PB2002 metadata ablation comparison
    # Compares Full (Group 4) vs Group 2 (Direct PB2002 metadata removed)
    pb2002_metadata_df = ablation_df[
        ablation_df["Feature Group"].isin(["Group 4: Full Feature Set", "Group 2: Seismic + Geographic"])
    ].copy()
    pb2002_metadata_df["Condition"] = [
        "PB2002 Metadata Removed (Seismic + Geographic Only)",
        "Full Feature Set (With PB2002 Kinematic Metadata)"
    ]
    pb2002_metadata_csv = ABLATION_DIR / "pb2002_metadata_ablation.csv"
    pb2002_metadata_df.to_csv(pb2002_metadata_csv, index=False)
    print(f"Saved PB2002 metadata ablation CSV: {pb2002_metadata_csv}")

    # Generate figures
    plot_ablation_figures(ablation_df)

    print("\n" + "=" * 70)
    print("STEP 3D & 3E ABLATION STUDY COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
