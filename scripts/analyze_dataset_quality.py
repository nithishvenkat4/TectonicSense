from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "interim" / "earthquakes_pb2002_nearest.csv"

METRICS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"

METRICS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("TECTONICSENSE - STEP 3A")
print("DATASET QUALITY & BOUNDARY DISTANCE ANALYSIS")
print("=" * 70)

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(INPUT, low_memory=False)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")

print("\nColumns:")
for col in df.columns:
    print(f"  - {col}")


# ---------------------------------------------------------
# Identify important columns
# ---------------------------------------------------------

distance_col = "boundary_distance_km"
target_col = "boundary_class"

if distance_col not in df.columns:
    raise ValueError(
        f"{distance_col} not found.\n"
        f"Available columns: {list(df.columns)}"
    )

if target_col not in df.columns:
    raise ValueError(
        f"{target_col} not found.\n"
        f"Available columns: {list(df.columns)}"
    )


# ---------------------------------------------------------
# Basic quality checks
# ---------------------------------------------------------

duplicate_rows = df.duplicated().sum()

if "event_id" in df.columns:
    duplicate_events = df["event_id"].duplicated().sum()
else:
    duplicate_events = None

missing = df.isna().sum()

print("\n" + "=" * 70)
print("DATA QUALITY")
print("=" * 70)

print(f"Duplicate rows      : {duplicate_rows:,}")

if duplicate_events is not None:
    print(f"Duplicate event IDs : {duplicate_events:,}")

print(f"Missing values      : {missing.sum():,}")

if missing.sum() > 0:
    print("\nColumns containing missing values:")
    print(missing[missing > 0])


# ---------------------------------------------------------
# Target distribution
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TECTONIC CLASS DISTRIBUTION")
print("=" * 70)

class_counts = df[target_col].value_counts().sort_index()

for cls, count in class_counts.items():
    percentage = count / len(df) * 100
    print(f"{cls:>4} : {count:>8,} ({percentage:6.2f}%)")


# ---------------------------------------------------------
# Boundary distance statistics
# ---------------------------------------------------------

distance = pd.to_numeric(
    df[distance_col],
    errors="coerce"
)

if distance.isna().any():
    raise ValueError("Invalid distance values detected.")


print("\n" + "=" * 70)
print("BOUNDARY DISTANCE STATISTICS")
print("=" * 70)

print(f"Minimum        : {distance.min():.3f} km")
print(f"25th percentile: {distance.quantile(.25):.3f} km")
print(f"Median         : {distance.median():.3f} km")
print(f"Mean           : {distance.mean():.3f} km")
print(f"75th percentile: {distance.quantile(.75):.3f} km")
print(f"90th percentile: {distance.quantile(.90):.3f} km")
print(f"95th percentile: {distance.quantile(.95):.3f} km")
print(f"99th percentile: {distance.quantile(.99):.3f} km")
print(f"Maximum        : {distance.max():.3f} km")


# ---------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DISTANCE THRESHOLD ANALYSIS")
print("=" * 70)

thresholds = [
    10,
    25,
    50,
    100,
    150,
    200,
    300,
    500,
    1000,
    2000,
]

threshold_results = []

print(
    f"{'Threshold':>12}"
    f"{'Retained':>14}"
    f"{'Retention':>14}"
    f"{'Removed':>14}"
)

for threshold in thresholds:

    retained = (distance <= threshold).sum()
    removed = len(df) - retained

    retention = retained / len(df) * 100

    print(
        f"{threshold:>9} km"
        f"{retained:>14,}"
        f"{retention:>13.2f}%"
        f"{removed:>14,}"
    )

    threshold_results.append({
        "threshold_km": threshold,
        "retained_events": int(retained),
        "retention_percent": round(retention, 3),
        "removed_events": int(removed),
    })


threshold_df = pd.DataFrame(threshold_results)

threshold_df.to_csv(
    METRICS / "boundary_distance_threshold_analysis.csv",
    index=False
)


# ---------------------------------------------------------
# Class distribution under thresholds
# ---------------------------------------------------------

class_threshold_results = []

for threshold in [50, 100, 200, 500, 1000]:

    subset = df[distance <= threshold]

    counts = subset[target_col].value_counts()

    row = {
        "threshold_km": threshold,
        "retained_events": len(subset),
    }

    for cls in sorted(df[target_col].unique()):
        row[f"class_{cls}"] = int(counts.get(cls, 0))

    class_threshold_results.append(row)


class_threshold_df = pd.DataFrame(
    class_threshold_results
)

class_threshold_df.to_csv(
    METRICS / "class_distribution_by_distance_threshold.csv",
    index=False
)


# ---------------------------------------------------------
# Save quality report
# ---------------------------------------------------------

report = {
    "rows": int(len(df)),
    "columns": int(len(df.columns)),
    "duplicate_rows": int(duplicate_rows),
    "duplicate_event_ids": (
        int(duplicate_events)
        if duplicate_events is not None
        else None
    ),
    "missing_values": int(missing.sum()),

    "class_distribution": {
        str(k): int(v)
        for k, v in class_counts.items()
    },

    "distance_statistics_km": {
        "minimum": float(distance.min()),
        "q25": float(distance.quantile(.25)),
        "median": float(distance.median()),
        "mean": float(distance.mean()),
        "q75": float(distance.quantile(.75)),
        "q90": float(distance.quantile(.90)),
        "q95": float(distance.quantile(.95)),
        "q99": float(distance.quantile(.99)),
        "maximum": float(distance.max()),
    },

    "threshold_analysis": threshold_results
}


with open(
    METRICS / "dataset_quality_report.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=4
    )


# ---------------------------------------------------------
# Visualization 1
# ---------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.hist(
    distance,
    bins=100
)

plt.xlabel(
    "Earthquake-to-PB2002 Boundary Distance (km)"
)

plt.ylabel(
    "Number of Earthquakes"
)

plt.title(
    "TectonicSense Boundary Association Distance Distribution"
)

plt.tight_layout()

plt.savefig(
    FIGURES / "boundary_distance_distribution.png",
    dpi=200
)

plt.close()


# ---------------------------------------------------------
# Visualization 2 - log scale
# ---------------------------------------------------------

positive_distance = distance[distance > 0]

plt.figure(figsize=(9, 5))

plt.hist(
    positive_distance,
    bins=100
)

plt.xscale("log")

plt.xlabel(
    "Boundary Distance (km, logarithmic scale)"
)

plt.ylabel(
    "Number of Earthquakes"
)

plt.title(
    "TectonicSense Boundary Distance Distribution"
)

plt.tight_layout()

plt.savefig(
    FIGURES / "boundary_distance_distribution_log.png",
    dpi=200
)

plt.close()


print("\n" + "=" * 70)
print("STEP 3A COMPLETE")
print("=" * 70)

print("\nGenerated files:")

print(
    "results/metrics/dataset_quality_report.json"
)

print(
    "results/metrics/boundary_distance_threshold_analysis.csv"
)

print(
    "results/metrics/class_distribution_by_distance_threshold.csv"
)

print(
    "results/figures/boundary_distance_distribution.png"
)

print(
    "results/figures/boundary_distance_distribution_log.png"
)

print(
    "\nIMPORTANT: No events were removed."
)

print(
    "We will choose the final distance threshold only "
    "after inspecting these results."
)