"""TectonicSense - Phase 2 Model Comparison Script

Combines baseline and advanced model benchmark results.
Generates:
  - results/phase_2_machine_learning/model_comparison/model_comparison.csv
  - results/phase_2_machine_learning/model_comparison/model_comparison.json
  - Visualizations in results/phase_2_machine_learning/figures/:
      1. macro_f1_comparison.png
      2. accuracy_comparison.png
      3. weighted_f1_comparison.png
      4. training_time_comparison.png
"""

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parents[1]
BASELINE_JSON = ROOT_DIR / "results" / "phase_2_machine_learning" / "baseline" / "baseline_results.json"
ADVANCED_JSON = ROOT_DIR / "results" / "phase_2_machine_learning" / "advanced_models" / "advanced_model_results.json"

COMPARISON_DIR = ROOT_DIR / "results" / "phase_2_machine_learning" / "model_comparison"
FIGURES_DIR = ROOT_DIR / "results" / "phase_2_machine_learning" / "figures"


def load_all_results():
    """Load both baseline and advanced model JSON results."""
    all_models = {}

    if BASELINE_JSON.exists():
        with open(BASELINE_JSON, "r") as f:
            baseline_data = json.load(f)
            all_models.update(baseline_data)
    else:
        print(f"Warning: {BASELINE_JSON} not found.")

    if ADVANCED_JSON.exists():
        with open(ADVANCED_JSON, "r") as f:
            advanced_data = json.load(f)
            all_models.update(advanced_data)
    else:
        print(f"Warning: {ADVANCED_JSON} not found.")

    return all_models


def build_comparison_dataframe(all_results):
    """Format into standard comparison dataframe sorted by Macro F1 descending."""
    rows = []
    for model_name, data in all_results.items():
        val = data.get("validation", {})
        rows.append({
            "Model": model_name,
            "Accuracy": val.get("accuracy", 0.0),
            "Macro Precision": val.get("macro_precision", 0.0),
            "Macro Recall": val.get("macro_recall", 0.0),
            "Macro F1": val.get("macro_f1", 0.0),
            "Weighted F1": val.get("weighted_f1", 0.0),
            "Training Time": val.get("training_time_sec", 0.0),
            "Inference Time": val.get("inference_time_sec", 0.0),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(by="Macro F1", ascending=False).reset_index(drop=True)
    return df


def plot_metric_bar(df, metric_col, title, filename, ylabel, palette="viridis"):
    """Generate high quality single metric bar plot."""
    plt.figure(figsize=(10, 5.5), dpi=300)
    sns.set_theme(style="whitegrid")

    df_sorted = df.sort_values(by=metric_col, ascending=True)
    colors = sns.color_palette(palette, len(df_sorted))

    bars = plt.barh(df_sorted["Model"], df_sorted[metric_col], color=colors, edgecolor="black", linewidth=0.8)

    for bar in bars:
        width = bar.get_width()
        if metric_col == "Training Time":
            label_text = f"{width:.1f}s"
        else:
            label_text = f"{width:.4f}"
        plt.text(
            width + (0.01 * (max(df_sorted[metric_col]) if max(df_sorted[metric_col]) > 0 else 1)),
            bar.get_y() + bar.get_height() / 2,
            label_text,
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold"
        )

    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel(ylabel, fontsize=12, fontweight="semibold")
    plt.ylabel("Model Architecture", fontsize=12, fontweight="semibold")
    if metric_col != "Training Time":
        plt.xlim(0, 1.12)
    plt.tight_layout()

    out_path = FIGURES_DIR / filename
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved figure: {out_path}")


def main():
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 2 MODEL COMPARISON & BENCHMARKING")
    print("=" * 70)

    all_results = load_all_results()
    if not all_results:
        raise RuntimeError("No model results found to compare!")

    comp_df = build_comparison_dataframe(all_results)

    # Save CSV
    csv_path = COMPARISON_DIR / "model_comparison.csv"
    comp_df.to_csv(csv_path, index=False)
    print(f"Saved comparison CSV: {csv_path}")

    # Save JSON
    json_path = COMPARISON_DIR / "model_comparison.json"
    comp_json = {
        "models": all_results,
        "ranking_by_macro_f1": comp_df.to_dict(orient="records")
    }
    with open(json_path, "w") as f:
        json.dump(comp_json, f, indent=4)
    print(f"Saved comparison JSON: {json_path}")

    print("\n" + "=" * 70)
    print("OVERALL MODEL BENCHMARK (SORTED BY MACRO F1)")
    print("=" * 70)
    print(comp_df.to_string(index=False))

    # Generate the 4 required figures
    print("\nGenerating Phase 2 benchmarking figures...")
    plot_metric_bar(
        comp_df,
        metric_col="Macro F1",
        title="TectonicSense Phase 2: Macro F1 Comparison (Validation)",
        filename="macro_f1_comparison.png",
        ylabel="Validation Macro F1 Score",
        palette="crest"
    )

    plot_metric_bar(
        comp_df,
        metric_col="Accuracy",
        title="TectonicSense Phase 2: Accuracy Comparison (Validation)",
        filename="accuracy_comparison.png",
        ylabel="Validation Accuracy",
        palette="viridis"
    )

    plot_metric_bar(
        comp_df,
        metric_col="Weighted F1",
        title="TectonicSense Phase 2: Weighted F1 Comparison (Validation)",
        filename="weighted_f1_comparison.png",
        ylabel="Validation Weighted F1 Score",
        palette="mako"
    )

    plot_metric_bar(
        comp_df,
        metric_col="Training Time",
        title="TectonicSense Phase 2: Training Computational Latency",
        filename="training_time_comparison.png",
        ylabel="Training Time (seconds)",
        palette="rocket"
    )

    print("\nComparison and visualization generation complete!")


if __name__ == "__main__":
    main()
