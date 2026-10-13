from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

USGS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "usgs_earthquakes_clean.csv"
)

PB2002_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "pb2002"
    / "PB2002_steps.dat"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "earthquakes_pb2002_nearest.csv"
)

EARTH_RADIUS_KM = 6371.0088


# ============================================================
# PB2002 LOADER
# ============================================================

def load_pb2002(path):
    """
    Load PB2002_steps.dat.

    Expected columns:
    step_id
    boundary_id
    start_lon
    start_lat
    end_lon
    end_lat
    length_km
    azimuth_deg
    relative_velocity
    velocity_azimuth_deg
    divergent_component
    right_lateral_component
    elevation_m
    seafloor_age_ma
    boundary_class
    """

    columns = [
        "step_id",
        "boundary_id",
        "start_lon",
        "start_lat",
        "end_lon",
        "end_lat",
        "length_km",
        "azimuth_deg",
        "relative_velocity",
        "velocity_azimuth_deg",
        "divergent_component",
        "right_lateral_component",
        "elevation_m",
        "seafloor_age_ma",
        "boundary_class",
    ]

    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 15:
                continue

            try:
                rows.append([
                    parts[0].strip().strip(":"),
                    parts[1].strip().strip(":"),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                    float(parts[5]),
                    float(parts[6]),
                    float(parts[7]),
                    float(parts[8]),
                    float(parts[9]),
                    float(parts[10]),
                    float(parts[11]),
                    float(parts[12]),
                    float(parts[13]),
                    parts[14].strip().strip(":"),
                ])

            except ValueError:
                continue

    df = pd.DataFrame(rows, columns=columns)

    # Orogen marker:
    # PB2002 uses * to indicate an orogen marker.
    df["orogen_flag"] = (
        df["boundary_class"]
        .astype(str)
        .str.endswith("*")
        .astype(int)
    )

    df["boundary_class"] = (
        df["boundary_class"]
        .astype(str)
        .str.rstrip("*")
    )

    return df


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("TECTONICSENSE — EARTHQUAKE / PB2002 ASSOCIATION")
print("=" * 70)

print("\nLoading USGS earthquake dataset...")

earthquakes = pd.read_csv(
    USGS_FILE,
    parse_dates=["time"]
)

print(f"Earthquake records: {len(earthquakes):,}")

print("\nLoading PB2002 boundary steps...")

pb = load_pb2002(PB2002_FILE)

print(f"PB2002 boundary steps: {len(pb):,}")

print(
    f"PB2002 classes: "
    f"{sorted(pb['boundary_class'].unique().tolist())}"
)


# ============================================================
# PREPARE PB2002 MIDPOINTS
# ============================================================

print("\nCalculating PB2002 segment midpoints...")

# Calculate midpoint carefully with longitude wrapping.
lon1 = pb["start_lon"].to_numpy()
lon2 = pb["end_lon"].to_numpy()

lat1 = pb["start_lat"].to_numpy()
lat2 = pb["end_lat"].to_numpy()

dlon = ((lon2 - lon1 + 180) % 360) - 180

mid_lon = lon1 + dlon / 2
mid_lon = ((mid_lon + 180) % 360) - 180

mid_lat = (lat1 + lat2) / 2

pb["mid_lon"] = mid_lon
pb["mid_lat"] = mid_lat


# ============================================================
# BUILD BALLTREE
# ============================================================

print("\nBuilding global spherical spatial index...")

boundary_coords = np.radians(
    pb[["mid_lat", "mid_lon"]].to_numpy()
)

tree = BallTree(
    boundary_coords,
    metric="haversine"
)


# ============================================================
# FIND NEAREST BOUNDARY
# ============================================================

print("\nFinding nearest PB2002 boundary for each earthquake...")

earthquake_coords = np.radians(
    earthquakes[["latitude", "longitude"]].to_numpy()
)

distances_rad, indices = tree.query(
    earthquake_coords,
    k=1
)

distances_km = (
    distances_rad[:, 0]
    * EARTH_RADIUS_KM
)

nearest_indices = indices[:, 0]

nearest_pb = (
    pb.iloc[nearest_indices]
    .reset_index(drop=True)
)


# ============================================================
# CREATE ASSOCIATION DATASET
# ============================================================

result = earthquakes.copy()

result["boundary_distance_km"] = distances_km

result["boundary_step_id"] = (
    nearest_pb["step_id"].values
)

result["boundary_id"] = (
    nearest_pb["boundary_id"].values
)

result["boundary_class"] = (
    nearest_pb["boundary_class"].values
)

result["orogen_flag"] = (
    nearest_pb["orogen_flag"].values
)

result["boundary_length_km"] = (
    nearest_pb["length_km"].values
)

result["boundary_azimuth_deg"] = (
    nearest_pb["azimuth_deg"].values
)

result["relative_velocity"] = (
    nearest_pb["relative_velocity"].values
)

result["velocity_azimuth_deg"] = (
    nearest_pb["velocity_azimuth_deg"].values
)

result["divergent_component"] = (
    nearest_pb["divergent_component"].values
)

result["right_lateral_component"] = (
    nearest_pb["right_lateral_component"].values
)

result["boundary_elevation_m"] = (
    nearest_pb["elevation_m"].values
)

result["seafloor_age_ma"] = (
    nearest_pb["seafloor_age_ma"].values
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("ASSOCIATION RESULTS")
print("=" * 70)

print(
    f"\nTotal associated earthquakes: "
    f"{len(result):,}"
)

print("\nBoundary distance statistics:")

print(
    result["boundary_distance_km"]
    .describe()
    .round(3)
)

print("\nBoundary class distribution:")

print(
    result["boundary_class"]
    .value_counts()
    .sort_index()
)

print("\nOrogen flag distribution:")

print(
    result["orogen_flag"]
    .value_counts()
    .sort_index()
)


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved to:")

print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("PB2002 ASSOCIATION COMPLETE")
print("=" * 70)