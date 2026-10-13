from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "interim"
    / "earthquakes_pb2002_refined.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "tectonicsense_ml_dataset.csv"
)

METRICS_FILE = (
    ROOT
    / "results"
    / "metrics"
    / "feature_engineering_report.json"
)

DOCS_FILE = (
    ROOT
    / "docs"
    / "feature_engineering.md"
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
DOCS_FILE.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "boundary_class"

REQUIRED_COLUMNS = [
    "event_id",
    "time",
    "longitude",
    "latitude",
    "depth_km",
    "magnitude",
    "magnitude_type",
    "boundary_distance_km",
    "boundary_step_id",
    "boundary_id",
    "boundary_class",
    "orogen_flag",
    "boundary_length_km",
    "boundary_azimuth_deg",
    "relative_velocity",
    "velocity_azimuth_deg",
    "divergent_component",
    "right_lateral_component",
    "boundary_elevation_m",
    "seafloor_age_ma",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 60)
    print("TectonicSense Feature Engineering")
    print("=" * 60)

    print("\nLoading refined PB2002 earthquake dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns)}")

    missing_required = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_required:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_required)
        )

    return df


# ============================================================
# BASIC VALIDATION
# ============================================================

def validate_input(df):

    print("\nValidating input dataset...")

    duplicate_events = df["event_id"].duplicated().sum()

    missing_values = int(df.isna().sum().sum())

    print(f"Duplicate event IDs: {duplicate_events:,}")
    print(f"Missing values: {missing_values:,}")

    if duplicate_events > 0:
        raise ValueError(
            "Duplicate event IDs detected."
        )

    if missing_values > 0:
        raise ValueError(
            "Missing values detected."
        )

    valid_classes = sorted(
        df[TARGET].dropna().unique().tolist()
    )

    print(f"Target classes: {valid_classes}")

    print("\nTarget distribution:")
    print(df[TARGET].value_counts())


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):

    print("\nCreating engineered features...")

    data = df.copy()

    # --------------------------------------------------------
    # 1. Earthquake features
    # --------------------------------------------------------

    data["depth_abs_km"] = data["depth_km"].abs()

    data["magnitude_squared"] = (
        data["magnitude"] ** 2
    )

    # --------------------------------------------------------
    # 2. Global spatial encoding
    # --------------------------------------------------------

    longitude_rad = np.radians(
        data["longitude"]
    )

    latitude_rad = np.radians(
        data["latitude"]
    )

    data["longitude_sin"] = np.sin(
        longitude_rad
    )

    data["longitude_cos"] = np.cos(
        longitude_rad
    )

    data["latitude_sin"] = np.sin(
        latitude_rad
    )

    data["latitude_cos"] = np.cos(
        latitude_rad
    )

    # --------------------------------------------------------
    # 3. Boundary-distance features
    # --------------------------------------------------------

    data["log_boundary_distance_km"] = np.log1p(
        data["boundary_distance_km"]
    )

    # --------------------------------------------------------
    # 4. Boundary orientation
    # --------------------------------------------------------

    boundary_azimuth_rad = np.radians(
        data["boundary_azimuth_deg"]
    )

    data["boundary_azimuth_sin"] = np.sin(
        boundary_azimuth_rad
    )

    data["boundary_azimuth_cos"] = np.cos(
        boundary_azimuth_rad
    )

    # --------------------------------------------------------
    # 5. Plate-motion orientation
    # --------------------------------------------------------

    velocity_azimuth_rad = np.radians(
        data["velocity_azimuth_deg"]
    )

    data["velocity_azimuth_sin"] = np.sin(
        velocity_azimuth_rad
    )

    data["velocity_azimuth_cos"] = np.cos(
        velocity_azimuth_rad
    )

    # --------------------------------------------------------
    # 6. Tectonic-motion magnitude
    # --------------------------------------------------------

    data["absolute_divergent_component"] = (
        data["divergent_component"].abs()
    )

    data["absolute_right_lateral_component"] = (
        data["right_lateral_component"].abs()
    )

    data["tectonic_motion_magnitude"] = np.sqrt(
        data["divergent_component"] ** 2
        +
        data["right_lateral_component"] ** 2
    )

    return data


# ============================================================
# SELECT FINAL ML FEATURES
# ============================================================

def select_ml_dataset(data):

    feature_columns = [
        # Earthquake
        "magnitude",
        "magnitude_squared",
        "depth_km",
        "depth_abs_km",

        # Spatial
        "longitude",
        "latitude",
        "longitude_sin",
        "longitude_cos",
        "latitude_sin",
        "latitude_cos",

        # Boundary relationship
        "boundary_distance_km",
        "log_boundary_distance_km",

        # Boundary geometry
        "boundary_length_km",
        "boundary_azimuth_sin",
        "boundary_azimuth_cos",

        # Plate motion
        "relative_velocity",
        "velocity_azimuth_sin",
        "velocity_azimuth_cos",
        "divergent_component",
        "right_lateral_component",
        "absolute_divergent_component",
        "absolute_right_lateral_component",
        "tectonic_motion_magnitude",

        # Geological
        "boundary_elevation_m",
        "seafloor_age_ma",
        "orogen_flag",
    ]

    final_columns = feature_columns + [TARGET]

    ml_df = data[final_columns].copy()

    return ml_df, feature_columns


# ============================================================
# FINAL VALIDATION
# ============================================================

def validate_final_dataset(
    ml_df,
    feature_columns
):

    print("\n" + "=" * 60)
    print("FINAL ML DATASET VALIDATION")
    print("=" * 60)

    print(
        f"Rows: {len(ml_df):,}"
    )

    print(
        f"Features: {len(feature_columns)}"
    )

    print(
        f"Target: {TARGET}"
    )

    print(
        f"Total columns: {len(ml_df.columns)}"
    )

    missing = int(
        ml_df.isna().sum().sum()
    )

    duplicates = int(
        ml_df.duplicated().sum()
    )

    print(
        f"Missing values: {missing:,}"
    )

    print(
        f"Duplicate rows: {duplicates:,}"
    )

    if missing > 0:
        raise ValueError(
            "Final dataset contains missing values."
        )

    if duplicates > 0:
        print(
            "Warning: duplicate feature rows exist."
        )

    print("\nFeature list:")

    for i, feature in enumerate(
        feature_columns,
        start=1
    ):
        print(
            f"{i:02d}. {feature}"
        )

    print("\nFeature dtypes:")

    print(
        ml_df[feature_columns].dtypes
    )

    print("\nTarget distribution:")

    print(
        ml_df[TARGET].value_counts()
    )


# ============================================================
# GENERATE REPORT
# ============================================================

def generate_report(
    original_df,
    ml_df,
    feature_columns
):

    target_counts = (
        ml_df[TARGET]
        .value_counts()
        .sort_index()
    )

    report = {

        "project": "TectonicSense",

        "stage": "Step 3C - Feature Engineering",

        "input_dataset": str(
            INPUT_FILE.relative_to(ROOT)
        ),

        "output_dataset": str(
            OUTPUT_FILE.relative_to(ROOT)
        ),

        "rows": int(
            len(ml_df)
        ),

        "original_columns": int(
            len(original_df.columns)
        ),

        "final_feature_count": int(
            len(feature_columns)
        ),

        "target": TARGET,

        "features": feature_columns,

        "excluded_identifier_columns": [
            "event_id",
            "boundary_id",
            "boundary_step_id",
        ],

        "excluded_raw_text_columns": [
            "place",
            "magnitude_type",
        ],

        "excluded_raw_metadata_columns": [
            "time",
            "event_type",
            "status",
        ],

        "target_distribution": {
            str(k): int(v)
            for k, v in target_counts.items()
        },

        "missing_values": int(
            ml_df.isna().sum().sum()
        ),

        "feature_engineering": {
            "depth_abs_km": (
                "Absolute earthquake depth"
            ),

            "magnitude_squared": (
                "Squared earthquake magnitude"
            ),

            "longitude_sin": (
                "Sine encoding of longitude"
            ),

            "longitude_cos": (
                "Cosine encoding of longitude"
            ),

            "latitude_sin": (
                "Sine encoding of latitude"
            ),

            "latitude_cos": (
                "Cosine encoding of latitude"
            ),

            "log_boundary_distance_km": (
                "log1p transformed boundary distance"
            ),

            "boundary_azimuth_sin": (
                "Sine encoding of boundary azimuth"
            ),

            "boundary_azimuth_cos": (
                "Cosine encoding of boundary azimuth"
            ),

            "velocity_azimuth_sin": (
                "Sine encoding of velocity azimuth"
            ),

            "velocity_azimuth_cos": (
                "Cosine encoding of velocity azimuth"
            ),

            "absolute_divergent_component": (
                "Absolute divergent motion"
            ),

            "absolute_right_lateral_component": (
                "Absolute right-lateral motion"
            ),

            "tectonic_motion_magnitude": (
                "Magnitude of combined divergent "
                "and right-lateral components"
            ),
        },
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    return report


# ============================================================
# DOCUMENTATION
# ============================================================

def generate_documentation(
    feature_columns
):

    content = f"""# TectonicSense — Feature Engineering

## Step

Step 3C — Final ML Dataset and Feature Engineering

## Input

`data/interim/earthquakes_pb2002_refined.csv`

## Output

`data/processed/tectonicsense_ml_dataset.csv`

## Target

The machine-learning target is:

`boundary_class`

The seven PB2002 classes are:

- CCB
- CRB
- CTF
- OCB
- OSR
- OTF
- SUB

## Final Features

The final ML dataset contains {len(feature_columns)} predictive features.

"""

    for i, feature in enumerate(
        feature_columns,
        start=1
    ):
        content += f"{i}. `{feature}`\\n"

    content += """

## Identifier Leakage Prevention

The following identifier columns are excluded:

- `event_id`
- `boundary_id`
- `boundary_step_id`

These columns identify individual earthquake events or PB2002 boundary segments and are therefore not used as predictive features.

## Raw Text Exclusion

The following text fields are excluded:

- `place`
- `magnitude_type`

The current model focuses on numerical seismic, spatial, tectonic and geological information.

## Circular Feature Encoding

Angular variables are represented using sine and cosine transformations.

This avoids discontinuities around 0°/360°.

Encoded variables include:

- Boundary azimuth
- Velocity azimuth
- Longitude
- Latitude

## Derived Features

Additional numerical features include:

- Absolute earthquake depth
- Squared magnitude
- Log-transformed boundary distance
- Absolute divergent component
- Absolute right-lateral component
- Combined tectonic-motion magnitude

## Data Retention

No earthquake observations are removed based on boundary distance during this stage.

The boundary distance is retained as a predictive feature.

Distance-based filtering may be evaluated later as a controlled experiment.

## Important Modeling Consideration

`boundary_id` will not be used as a model feature.

During later evaluation, both random stratified splitting and group-aware splitting by `boundary_id` can be compared to assess potential geographic generalization and leakage effects.
"""

    with open(
        DOCS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(content)


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    validate_input(df)

    engineered = create_features(df)

    ml_df, feature_columns = select_ml_dataset(
        engineered
    )

    validate_final_dataset(
        ml_df,
        feature_columns
    )

    ml_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    generate_report(
        df,
        ml_df,
        feature_columns
    )

    generate_documentation(
        feature_columns
    )

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)

    print(
        f"ML dataset saved to:\n{OUTPUT_FILE}"
    )

    print(
        f"\nFeature report saved to:\n{METRICS_FILE}"
    )

    print(
        f"\nDocumentation saved to:\n{DOCS_FILE}"
    )


if __name__ == "__main__":
    main()