"""TectonicSense - Phase 3 SHAP Explainability Script

Computes SHAP values on a reproducible representative sample of the test set
(2,000 observations, random_state=42) using TreeExplainer on the serialized
best candidate XGBoost model.

Outputs:
  - results/phase_3_evaluation/explainability/shap_feature_importance.csv
  - results/phase_3_evaluation/explainability/shap_summary_bar.png
  - results/phase_3_evaluation/explainability/shap_summary_beeswarm.png
  - results/phase_3_evaluation/explainability/shap_dependence_<feature>.png (top 5 features)
  - results/phase_3_evaluation/explainability/shap_summary.json
"""

from pathlib import Path
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

RANDOM_STATE = 42
SAMPLE_SIZE = 2000

ROOT_DIR = Path(__file__).resolve().parents[1]
TEST_CSV = ROOT_DIR / "data" / "processed" / "splits" / "test.csv"
MODEL_DIR = ROOT_DIR / "models" / "phase_2_best_candidate"
EXPLAIN_DIR = ROOT_DIR / "results" / "phase_3_evaluation" / "explainability"

TARGET_COL = "boundary_class"


def load_model_and_sample():
    """Load candidate model and extract a reproducible representative test sample."""
    print("=" * 70)
    print("PHASE 3: SHAP TREE-EXPLAINER INITIALIZATION")
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
    
    # Stratified representative sample of 2,000 observations
    sample_df = test_df.groupby(TARGET_COL, group_keys=False).apply(
        lambda g: g.sample(n=int(np.round(len(g) / len(test_df) * SAMPLE_SIZE)), random_state=RANDOM_STATE)
    ).reset_index(drop=True)

    # Ensure exactly SAMPLE_SIZE
    if len(sample_df) < SAMPLE_SIZE:
        needed = SAMPLE_SIZE - len(sample_df)
        remaining = test_df.drop(sample_df.index)
        extra = remaining.sample(n=needed, random_state=RANDOM_STATE)
        sample_df = pd.concat([sample_df, extra], ignore_index=True)
    elif len(sample_df) > SAMPLE_SIZE:
        sample_df = sample_df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)

    X_sample = sample_df[features].copy()
    y_sample_str = sample_df[TARGET_COL].copy()

    print(f"Sample size: {len(X_sample)} observations x {len(features)} features")
    print(f"Target distribution in sample:\n{y_sample_str.value_counts()}")

    return model, features, label_mapping, X_sample, y_sample_str


def compute_shap(model, features, label_mapping, X_sample):
    """Compute SHAP values using TreeExplainer."""
    print("\nComputing TreeExplainer SHAP values...")
    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_sample)
    duration = time.time() - t0
    print(f"TreeExplainer completed in {duration:.2f}s")
    print(f"SHAP explanation tensor shape: {explanation.shape}")

    # explanation.values shape: (N, n_features, n_classes)
    shap_vals = explanation.values
    classes_ordered = [label_mapping[str(i)] for i in range(len(label_mapping))]

    # Global importance: mean across observations, then mean across 7 classes
    # Shape: (n_features,)
    mean_abs_per_class = np.mean(np.abs(shap_vals), axis=0) # shape: (n_features, n_classes)
    global_mean_abs = np.mean(mean_abs_per_class, axis=1) # shape: (n_features,)

    importance_df = pd.DataFrame({
        "feature": features,
        "mean_abs_shap": global_mean_abs
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print("\nTOP 10 MOST IMPORTANT FEATURES (GLOBAL MEAN |SHAP|):")
    for i, r in importance_df.head(10).iterrows():
        print(f"  {i+1:2d}. {r['feature']:<32} : {r['mean_abs_shap']:.6f}")

    return explanation, shap_vals, importance_df, classes_ordered, duration


def plot_shap_summary_bar(importance_df):
    """Generate high quality SHAP global importance bar plot."""
    plt.figure(figsize=(10, 6.5), dpi=300)
    sns.set_theme(style="whitegrid")

    df_top = importance_df.sort_values("mean_abs_shap", ascending=True)
    colors = sns.color_palette("viridis", len(df_top))

    bars = plt.barh(df_top["feature"], df_top["mean_abs_shap"], color=colors, edgecolor="black", linewidth=0.7)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", ha="left", va="center", fontsize=8.5, fontweight="bold")

    plt.title("TectonicSense Phase 3: Global Feature Importance (Mean |SHAP|)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Mean Absolute SHAP Value (Impact on Log-Odds Output across All 7 Classes)", fontsize=11, fontweight="semibold")
    plt.ylabel("Predictive Feature", fontsize=11, fontweight="semibold")
    plt.xlim(0, max(importance_df["mean_abs_shap"]) * 1.15)
    plt.tight_layout()

    out_path = EXPLAIN_DIR / "shap_summary_bar.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_shap_summary_beeswarm(explanation, features, importance_df):
    """Generate SHAP beeswarm plot for the dominant majority class (SUB) and top features."""
    # Class 6 corresponds to SUB (majority class representing ~48% of global seismicity)
    sub_class_idx = 6
    sub_explanation = shap.Explanation(
        values=explanation.values[:, :, sub_class_idx],
        base_values=explanation.base_values[:, sub_class_idx] if explanation.base_values.ndim > 1 else explanation.base_values,
        data=explanation.data,
        feature_names=features
    )

    plt.figure(figsize=(11, 7), dpi=300)
    shap.plots.beeswarm(sub_explanation, max_display=15, show=False)
    plt.title("TectonicSense Phase 3: SHAP Beeswarm Plot (Subduction Zone Class)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    out_path = EXPLAIN_DIR / "shap_summary_beeswarm.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_dependence_plots(explanation, features, importance_df, X_sample):
    """Generate dependence scatter plots for top 5 features."""
    top5_features = importance_df["feature"].head(5).tolist()
    print(f"\nGenerating dependence plots for top 5 features: {top5_features}")

    for feat in top5_features:
        feat_idx = features.index(feat)
        plt.figure(figsize=(8.5, 5), dpi=300)
        sns.set_theme(style="whitegrid")

        # Use majority class (SUB) SHAP values for dependence visualization
        x_vals = X_sample[feat].values
        y_shap = explanation.values[:, feat_idx, 6]

        plt.scatter(x_vals, y_shap, alpha=0.6, c="#1d3557", edgecolors="none", s=25)
        plt.axhline(0, color="gray", linestyle="--", linewidth=1.0)
        plt.title(f"TectonicSense SHAP Dependence: {feat} (SUB Class)", fontsize=12, fontweight="bold", pad=10)
        plt.xlabel(f"{feat} Feature Value", fontsize=11, fontweight="semibold")
        plt.ylabel(f"SHAP Value for Subduction Zone (SUB)", fontsize=11, fontweight="semibold")
        plt.tight_layout()

        out_path = EXPLAIN_DIR / f"shap_dependence_{feat}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved dependence plot: {out_path}")


def main():
    EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)

    model, features, label_mapping, X_sample, y_sample_str = load_model_and_sample()
    explanation, shap_vals, importance_df, classes_ordered, duration = compute_shap(
        model, features, label_mapping, X_sample
    )

    # 1. Save CSV
    csv_path = EXPLAIN_DIR / "shap_feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    print(f"\nSaved feature importance CSV: {csv_path}")

    # 2. Plots
    plot_shap_summary_bar(importance_df)
    plot_shap_summary_beeswarm(explanation, features, importance_df)
    plot_dependence_plots(explanation, features, importance_df, X_sample)

    # 3. Save JSON
    top10_list = importance_df.head(10).to_dict(orient="records")
    summary_json = {
        "sample_size": SAMPLE_SIZE,
        "random_seed": RANDOM_STATE,
        "calculation_duration_sec": round(duration, 2),
        "target_classes": classes_ordered,
        "top_10_features": top10_list,
        "all_feature_importance": importance_df.to_dict(orient="records")
    }

    json_path = EXPLAIN_DIR / "shap_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=4)
    print(f"Saved SHAP summary JSON: {json_path}")

    print("\n" + "=" * 70)
    print("STEP 3C SHAP EXPLAINABILITY COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
