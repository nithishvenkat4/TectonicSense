"""TectonicSense - Phase 3 Error Analysis Script

Performs rigorous empirical error analysis on incorrect classifications
from the final test set evaluation (data/processed/splits/test.csv).

Outputs:
  - results/phase_3_evaluation/error_analysis/error_analysis.csv
  - results/phase_3_evaluation/error_analysis/error_summary.json
  - Figures in results/phase_3_evaluation/figures/:
      - error_distribution_by_class.png
      - error_distribution_by_magnitude.png
      - error_distribution_by_depth.png
      - top_confusion_pairs.png
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parents[1]
TEST_PREDS_CSV = ROOT_DIR / "results" / "phase_3_evaluation" / "final_test" / "test_predictions.csv"
TEST_DATA_CSV = ROOT_DIR / "data" / "processed" / "splits" / "test.csv"

OUT_DIR = ROOT_DIR / "results" / "phase_3_evaluation" / "error_analysis"
FIG_DIR = ROOT_DIR / "results" / "phase_3_evaluation" / "figures"


def load_and_combine():
    """Merge predictions with physical feature columns for error analysis."""
    print("=" * 70)
    print("PHASE 3 ERROR ANALYSIS ON FINAL TEST PREDICTIONS")
    print("=" * 70)

    preds_df = pd.read_csv(TEST_PREDS_CSV)
    test_df = pd.read_csv(TEST_DATA_CSV)

    combined_df = preds_df.copy()
    combined_df["magnitude"] = test_df["magnitude"].values
    combined_df["depth_km"] = test_df["depth_km"].values
    combined_df["latitude"] = test_df["latitude"].values
    combined_df["longitude"] = test_df["longitude"].values
    combined_df["boundary_distance_km"] = test_df["boundary_distance_km"].values

    errors_df = combined_df[~combined_df["is_correct"]].copy()
    print(f"Total test observations: {len(combined_df):,}")
    print(f"Total misclassifications: {len(errors_df)} ({(len(errors_df)/len(combined_df))*100:.4f}% error rate)")

    return combined_df, errors_df


def analyze_errors(errors_df, total_count):
    """Compute empirical error metrics across confusion pairs and feature strata."""
    errors_df["confusion_pair"] = errors_df["true_label"] + " -> " + errors_df["predicted_label"]
    confusion_pair_counts = errors_df["confusion_pair"].value_counts().to_dict()

    errors_by_class = errors_df["true_label"].value_counts().to_dict()

    # Binned analysis
    errors_df["mag_bin"] = pd.cut(
        errors_df["magnitude"],
        bins=[4.0, 5.0, 6.0, 7.0, 10.0],
        labels=["4.0-5.0", "5.0-6.0", "6.0-7.0", "7.0+"],
        include_lowest=True
    )
    errors_by_mag = errors_df["mag_bin"].value_counts().to_dict()

    errors_df["depth_bin"] = pd.cut(
        errors_df["depth_km"],
        bins=[-5, 35, 70, 300, 1000],
        labels=["Shallow (0-35km)", "Intermediate-Shallow (35-70km)", "Intermediate (70-300km)", "Deep (300km+)"],
        include_lowest=True
    )
    errors_by_depth = errors_df["depth_bin"].value_counts().to_dict()

    errors_df["dist_bin"] = pd.cut(
        errors_df["boundary_distance_km"],
        bins=[-1, 20, 50, 100, 500],
        labels=["Proximal (<20km)", "Near (20-50km)", "Intermediate (50-100km)", "Distal (>100km)"],
        include_lowest=True
    )
    errors_by_dist = errors_df["dist_bin"].value_counts().to_dict()

    print("\nMisclassified Observations Breakdown:")
    for i, r in errors_df.iterrows():
        print(f"  Event: {r['event_id']:<14} | True: {r['true_label']:<4} | Pred: {r['predicted_label']:<4} | Conf: {r['confidence']:.4f} | Mag: {r['magnitude']:.1f} | Depth: {r['depth_km']:.1f}km | Dist: {r['boundary_distance_km']:.1f}km")

    summary = {
        "total_test_observations": total_count,
        "total_errors": len(errors_df),
        "overall_error_rate_percent": round(len(errors_df) / total_count * 100, 4),
        "confusion_pairs": confusion_pair_counts,
        "errors_by_true_class": errors_by_class,
        "errors_by_magnitude_range": {str(k): int(v) for k, v in errors_by_mag.items()},
        "errors_by_depth_range": {str(k): int(v) for k, v in errors_by_depth.items()},
        "errors_by_boundary_distance": {str(k): int(v) for k, v in errors_by_dist.items()}
    }

    return summary


def plot_error_figures(errors_df):
    """Generate publication-quality error analysis visualizations."""
    print("\nGenerating error analysis figures...")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Top Confusion Pairs
    pair_counts = errors_df["confusion_pair"].value_counts()
    plt.figure(figsize=(8, 4.5), dpi=300)
    bars = plt.barh(pair_counts.index, pair_counts.values, color="#e76f51", edgecolor="black", linewidth=0.7)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.05, bar.get_y() + bar.get_height() / 2, f"{int(w)} event", ha="left", va="center", fontsize=10, fontweight="bold")
    plt.title("TectonicSense Final Test: Specific Misclassification Pairs", fontsize=12.5, fontweight="bold", pad=12)
    plt.xlabel("Misclassified Observation Count", fontsize=11, fontweight="semibold")
    plt.ylabel("Confusion Transition (True -> Predicted)", fontsize=11, fontweight="semibold")
    plt.xlim(0, max(pair_counts.values) + 1)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "top_confusion_pairs.png", dpi=300)
    plt.close()

    # 2. Errors by Class
    class_errs = errors_df["true_label"].value_counts()
    plt.figure(figsize=(7, 4.5), dpi=300)
    bars = plt.bar(class_errs.index, class_errs.values, color="#264653", edgecolor="black", linewidth=0.7, width=0.5)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f"{int(h)}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.title("TectonicSense Final Test: Misclassifications by Ground Truth Class", fontsize=12.5, fontweight="bold", pad=12)
    plt.xlabel("PB2002 Ground Truth Class", fontsize=11, fontweight="semibold")
    plt.ylabel("Error Count", fontsize=11, fontweight="semibold")
    plt.ylim(0, max(class_errs.values) + 1)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "error_distribution_by_class.png", dpi=300)
    plt.close()

    # 3. Errors by Magnitude Range
    plt.figure(figsize=(8, 4.5), dpi=300)
    sns.histplot(errors_df["magnitude"], bins=10, color="#2a9d8f", kde=False, edgecolor="black")
    plt.title("TectonicSense Final Test: Error Distribution by Earthquake Magnitude", fontsize=12.5, fontweight="bold", pad=12)
    plt.xlabel("Earthquake Magnitude (M)", fontsize=11, fontweight="semibold")
    plt.ylabel("Error Count", fontsize=11, fontweight="semibold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "error_distribution_by_magnitude.png", dpi=300)
    plt.close()

    # 4. Errors by Depth Range
    plt.figure(figsize=(8, 4.5), dpi=300)
    sns.histplot(errors_df["depth_km"], bins=10, color="#f4a261", kde=False, edgecolor="black")
    plt.title("TectonicSense Final Test: Error Distribution by Focal Depth", fontsize=12.5, fontweight="bold", pad=12)
    plt.xlabel("Hypocenter Depth (km)", fontsize=11, fontweight="semibold")
    plt.ylabel("Error Count", fontsize=11, fontweight="semibold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "error_distribution_by_depth.png", dpi=300)
    plt.close()

    print("Saved error analysis figures successfully.")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    combined_df, errors_df = load_and_combine()

    # Save error analysis CSV
    err_csv_path = OUT_DIR / "error_analysis.csv"
    errors_df.to_csv(err_csv_path, index=False)
    print(f"\nSaved error analysis CSV: {err_csv_path}")

    # Summary JSON
    summary = analyze_errors(errors_df, len(combined_df))
    summary_path = OUT_DIR / "error_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Saved error summary JSON: {summary_path}")

    # Visualizations
    plot_error_figures(errors_df)

    print("\n" + "=" * 70)
    print("STEP 3G ERROR ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
