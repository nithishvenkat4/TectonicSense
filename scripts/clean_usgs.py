from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "usgs_earthquakes_2010_2025_m4.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "usgs_earthquakes_clean.csv"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("TECTONICSENSE - USGS DATA CLEANING")
print("=" * 60)

print()
print("Loading:")
print(INPUT_FILE)

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["time"]
)

print()
print("Original records:", len(df))


# ============================================================
# CHECK EVENT TYPES
# ============================================================

print()
print("Original event types:")
print(df["event_type"].value_counts())


# ============================================================
# KEEP ONLY EARTHQUAKES
# ============================================================

df = df[
    df["event_type"] == "earthquake"
].copy()

print()
print("After earthquake filtering:", len(df))


# ============================================================
# REMOVE DUPLICATE EVENT IDs
# ============================================================

duplicates = df["event_id"].duplicated().sum()

print()
print("Duplicate event IDs:", duplicates)

df = df.drop_duplicates(
    subset="event_id"
)


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "longitude",
    "latitude",
    "depth_km",
    "magnitude"
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# CHECK MISSING VALUES
# ============================================================

print()
print("Missing values:")
print(df.isna().sum())


# ============================================================
# REMOVE RECORDS WITH MISSING CORE VALUES
# ============================================================

core_columns = [
    "event_id",
    "time",
    "longitude",
    "latitude",
    "depth_km",
    "magnitude"
]

before = len(df)

df = df.dropna(
    subset=core_columns
)

after = len(df)

print()
print(
    "Records removed because of missing core values:",
    before - after
)


# ============================================================
# VALIDATE COORDINATES
# ============================================================

invalid_latitude = (
    (df["latitude"] < -90) |
    (df["latitude"] > 90)
)

invalid_longitude = (
    (df["longitude"] < -180) |
    (df["longitude"] > 180)
)

print()
print(
    "Invalid latitude records:",
    invalid_latitude.sum()
)

print(
    "Invalid longitude records:",
    invalid_longitude.sum()
)


# ============================================================
# REMOVE INVALID COORDINATES
# ============================================================

df = df[
    ~invalid_latitude &
    ~invalid_longitude
].copy()


# ============================================================
# SORT BY TIME
# ============================================================

df = df.sort_values(
    "time"
).reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 60)
print("CLEAN USGS DATASET")
print("=" * 60)

print()

print(
    "Total earthquake events:",
    f"{len(df):,}"
)

print(
    "Unique event IDs:",
    f"{df['event_id'].nunique():,}"
)

print()

print(
    "Date range:"
)

print(
    df["time"].min(),
    "to",
    df["time"].max()
)

print()

print(
    "Magnitude range:"
)

print(
    df["magnitude"].min(),
    "to",
    df["magnitude"].max()
)

print()

print(
    "Depth range:"
)

print(
    df["depth_km"].min(),
    "to",
    df["depth_km"].max()
)

print()

print(
    "Missing values:"
)

print(
    df.isna().sum()
)

print()

print(
    "Event types:"
)

print(
    df["event_type"].value_counts()
)

print()

print(
    "Saved to:"
)

print(
    OUTPUT_FILE
)

print()
print("=" * 60)
print("USGS CLEANING COMPLETE")
print("=" * 60)