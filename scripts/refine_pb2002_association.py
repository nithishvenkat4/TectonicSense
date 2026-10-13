from pathlib import Path
import re
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

USGS_FILE = ROOT / "data" / "processed" / "usgs_earthquakes_clean.csv"
PB2002_FILE = ROOT / "data" / "raw" / "pb2002" / "pb2002_steps.dat"

OUTPUT_FILE = ROOT / "data" / "interim" / "earthquakes_pb2002_refined.csv"

METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PB2002 PARSER
# ============================================================

def load_pb2002():
    """
    Parse PB2002_steps.dat.

    Expected fields:

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

    records = []

    with open(PB2002_FILE, "r", encoding="utf-8", errors="ignore") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            # Skip obvious comments/header lines
            if line.startswith("#"):
                continue

            parts = line.split()

            # We need at least 15 fields
            if len(parts) < 15:
                continue

            try:
                step_id = int(parts[0])

                # Boundary ID may appear as AF-AN or :AF-AN
                boundary_id = parts[1].strip().lstrip(":")

                start_lon = float(parts[2])
                start_lat = float(parts[3])

                end_lon = float(parts[4])
                end_lat = float(parts[5])

                length_km = float(parts[6])
                azimuth_deg = float(parts[7])

                relative_velocity = float(parts[8])
                velocity_azimuth_deg = float(parts[9])

                divergent_component = float(parts[10])
                right_lateral_component = float(parts[11])

                elevation_m = float(parts[12])

                seafloor_age_ma = float(parts[13])

                # Boundary class may appear as OTF or :OTF
                raw_class = parts[14].strip().lstrip(":")

                # PB2002 uses * to indicate orogen markers
                orogen_flag = 1 if "*" in raw_class else 0

                boundary_class = raw_class.replace("*", "")

                records.append({
                    "step_id": step_id,
                    "boundary_id": boundary_id,
                    "start_lon": start_lon,
                    "start_lat": start_lat,
                    "end_lon": end_lon,
                    "end_lat": end_lat,
                    "length_km": length_km,
                    "azimuth_deg": azimuth_deg,
                    "relative_velocity": relative_velocity,
                    "velocity_azimuth_deg": velocity_azimuth_deg,
                    "divergent_component": divergent_component,
                    "right_lateral_component": right_lateral_component,
                    "elevation_m": elevation_m,
                    "seafloor_age_ma": seafloor_age_ma,
                    "boundary_class": boundary_class,
                    "orogen_flag": orogen_flag,
                })

            except (ValueError, IndexError):
                continue

    if not records:
        raise ValueError("No PB2002 records were parsed.")

    df = pd.DataFrame(records)

    print(f"PB2002 segments parsed: {len(df):,}")
    print(f"Unique boundary IDs: {df['boundary_id'].nunique():,}")
    print(
        "Boundary classes:",
        sorted(df["boundary_class"].dropna().unique().tolist())
    )

    print("\nClass distribution:")
    print(df["boundary_class"].value_counts())

    print("\nOrogen flag:")
    print(df["orogen_flag"].value_counts())

    return df


# ============================================================
# GREAT-CIRCLE UTILITIES
# ============================================================

def lonlat_to_xyz(lon, lat):
    """
    Convert longitude/latitude in degrees
    to 3D unit sphere coordinates.
    """

    lon = np.radians(lon)
    lat = np.radians(lat)

    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    return np.column_stack((x, y, z))


def angular_distance(a, b):
    """
    Great-circle angular distance between two unit vectors.
    """

    dot = np.sum(a * b, axis=1)

    dot = np.clip(dot, -1.0, 1.0)

    return np.arccos(dot)


def point_to_arc_distance(point, start, end):
    """
    Approximate shortest great-circle distance from a point
    to the great-circle arc defined by start/end.

    Returns angular distance in radians.
    """

    # Normal vector of great-circle plane
    normal = np.cross(start, end)

    normal_norm = np.linalg.norm(normal)

    if normal_norm < 1e-12:
        # Degenerate segment
        return min(
            np.arccos(np.clip(np.dot(point, start), -1, 1)),
            np.arccos(np.clip(np.dot(point, end), -1, 1))
        )

    normal = normal / normal_norm

    # Project point onto great-circle plane
    projection = point - np.dot(point, normal) * normal

    projection_norm = np.linalg.norm(projection)

    if projection_norm < 1e-12:
        return min(
            np.arccos(np.clip(np.dot(point, start), -1, 1)),
            np.arccos(np.clip(np.dot(point, end), -1, 1))
        )

    projection = projection / projection_norm

    # Check both antipodal projections
    candidates = [
        projection,
        -projection
    ]

    best = float("inf")

    for candidate in candidates:

        # Check whether candidate lies on the minor arc
        d1 = np.arccos(
            np.clip(np.dot(start, candidate), -1, 1)
        )

        d2 = np.arccos(
            np.clip(np.dot(candidate, end), -1, 1)
        )

        d12 = np.arccos(
            np.clip(np.dot(start, end), -1, 1)
        )

        if abs((d1 + d2) - d12) < 1e-5:

            distance = np.arccos(
                np.clip(np.dot(point, candidate), -1, 1)
            )

            best = min(best, distance)

    # If projection is not on the segment,
    # nearest point is one of the endpoints.
    if not np.isfinite(best):

        best = min(
            np.arccos(np.clip(np.dot(point, start), -1, 1)),
            np.arccos(np.clip(np.dot(point, end), -1, 1))
        )

    return best


# ============================================================
# REFINED ASSOCIATION
# ============================================================

def associate_earthquakes():

    print("\nLoading earthquake data...")

    earthquakes = pd.read_csv(USGS_FILE)

    print(f"Earthquakes loaded: {len(earthquakes):,}")

    print("\nLoading PB2002 boundary data...")

    boundaries = load_pb2002()

    # --------------------------------------------------------
    # Boundary midpoint coordinates
    # --------------------------------------------------------

    midpoint_lon = (
        boundaries["start_lon"] +
        boundaries["end_lon"]
    ) / 2.0

    midpoint_lat = (
        boundaries["start_lat"] +
        boundaries["end_lat"]
    ) / 2.0

    # Handle longitude wraparound approximately
    lon_diff = (
        boundaries["end_lon"] -
        boundaries["start_lon"]
    )

    midpoint_lon = np.where(
        np.abs(lon_diff) > 180,
        midpoint_lon + np.where(midpoint_lon < 0, 180, -180),
        midpoint_lon
    )

    midpoint_lon = ((midpoint_lon + 180) % 360) - 180

    midpoint_lat = midpoint_lat.clip(-90, 90)

    midpoint_coords = np.radians(
        np.column_stack([
            midpoint_lat,
            midpoint_lon
        ])
    )

    # --------------------------------------------------------
    # BallTree candidate search
    # --------------------------------------------------------

    print("\nBuilding BallTree...")

    tree = BallTree(
        midpoint_coords,
        metric="haversine"
    )

    earthquake_coords = np.radians(
        earthquakes[["latitude", "longitude"]].values
    )

    # Search multiple candidate segments
    K = 20

    print(
        f"Searching {K} candidate PB2002 segments "
        "for every earthquake..."
    )

    distances, indices = tree.query(
        earthquake_coords,
        k=K
    )

    # --------------------------------------------------------
    # Refined point-to-segment search
    # --------------------------------------------------------

    print("\nComputing refined segment distances...")

    earthquake_xyz = lonlat_to_xyz(
        earthquakes["longitude"].values,
        earthquakes["latitude"].values
    )

    start_xyz = lonlat_to_xyz(
        boundaries["start_lon"].values,
        boundaries["start_lat"].values
    )

    end_xyz = lonlat_to_xyz(
        boundaries["end_lon"].values,
        boundaries["end_lat"].values
    )

    best_distance_km = np.full(
        len(earthquakes),
        np.inf
    )

    best_indices = np.full(
        len(earthquakes),
        -1,
        dtype=int
    )

    EARTH_RADIUS_KM = 6371.0088

    for i in range(len(earthquakes)):

        point = earthquake_xyz[i]

        best_angle = float("inf")
        best_idx = -1

        for candidate_idx in indices[i]:

            angle = point_to_arc_distance(
                point,
                start_xyz[candidate_idx],
                end_xyz[candidate_idx]
            )

            if angle < best_angle:
                best_angle = angle
                best_idx = candidate_idx

        best_distance_km[i] = (
            best_angle * EARTH_RADIUS_KM
        )

        best_indices[i] = best_idx

        if (i + 1) % 10000 == 0:
            print(
                f"Processed {i + 1:,} / "
                f"{len(earthquakes):,}"
            )

    # --------------------------------------------------------
    # Attach boundary information
    # --------------------------------------------------------

    selected = boundaries.iloc[best_indices].reset_index(drop=True)

    result = earthquakes.copy()

    result["boundary_distance_km"] = best_distance_km

    result["boundary_step_id"] = selected["step_id"].values
    result["boundary_id"] = selected["boundary_id"].values

    result["boundary_class"] = selected["boundary_class"].values
    result["orogen_flag"] = selected["orogen_flag"].values

    result["boundary_length_km"] = selected["length_km"].values
    result["boundary_azimuth_deg"] = selected["azimuth_deg"].values

    result["relative_velocity"] = selected[
        "relative_velocity"
    ].values

    result["velocity_azimuth_deg"] = selected[
        "velocity_azimuth_deg"
    ].values

    result["divergent_component"] = selected[
        "divergent_component"
    ].values

    result["right_lateral_component"] = selected[
        "right_lateral_component"
    ].values

    result["boundary_elevation_m"] = selected[
        "elevation_m"
    ].values

    result["seafloor_age_ma"] = selected[
        "seafloor_age_ma"
    ].values

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n============================================")
    print("REFINED ASSOCIATION COMPLETE")
    print("============================================")

    print(
        f"Total earthquakes: "
        f"{len(result):,}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print("\nDistance statistics:")

    print(
        result["boundary_distance_km"].describe(
            percentiles=[
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99
            ]
        )
    )

    print("\nBoundary classes:")

    print(
        result["boundary_class"].value_counts()
    )

    print("\nOrogen flag:")

    print(
        result["orogen_flag"].value_counts()
    )

    # --------------------------------------------------------
    # Metrics JSON
    # --------------------------------------------------------

    metrics = {
        "total_earthquakes": int(len(result)),
        "pb2002_segments": int(len(boundaries)),
        "unique_boundary_ids": int(
            boundaries["boundary_id"].nunique()
        ),
        "candidate_k": K,
        "distance_km": {
            "min": float(result["boundary_distance_km"].min()),
            "q25": float(result["boundary_distance_km"].quantile(0.25)),
            "median": float(result["boundary_distance_km"].median()),
            "mean": float(result["boundary_distance_km"].mean()),
            "q75": float(result["boundary_distance_km"].quantile(0.75)),
            "q90": float(result["boundary_distance_km"].quantile(0.90)),
            "q95": float(result["boundary_distance_km"].quantile(0.95)),
            "q99": float(result["boundary_distance_km"].quantile(0.99)),
            "max": float(result["boundary_distance_km"].max()),
        },
        "class_distribution": {
            str(k): int(v)
            for k, v in result["boundary_class"]
            .value_counts()
            .items()
        }
    }

    with open(
        METRICS_DIR /
        "refined_pb2002_association_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Distance histogram
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.hist(
        result["boundary_distance_km"],
        bins=100
    )

    plt.xlabel(
        "Distance to PB2002 Boundary (km)"
    )

    plt.ylabel(
        "Number of Earthquakes"
    )

    plt.title(
        "Refined Earthquake-to-Tectonic-Boundary Distance"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        "refined_boundary_distance_distribution.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Log-scale histogram
    # --------------------------------------------------------

    positive_distances = result[
        result["boundary_distance_km"] > 0
    ]["boundary_distance_km"]

    plt.figure(figsize=(10, 6))

    plt.hist(
        positive_distances,
        bins=np.logspace(
            np.log10(0.01),
            np.log10(max(positive_distances)),
            100
        )
    )

    plt.xscale("log")

    plt.xlabel(
        "Distance to PB2002 Boundary (km)"
    )

    plt.ylabel(
        "Number of Earthquakes"
    )

    plt.title(
        "Refined Boundary Distance Distribution — Log Scale"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        "refined_boundary_distance_distribution_log.png",
        dpi=300
    )

    plt.close()

    print("\nFigures saved.")

    return result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    associate_earthquakes()