from pathlib import Path
import pandas as pd
import requests
import time


# ============================================================
# CONFIGURATION
# ============================================================

START_YEAR = 2010
END_YEAR = 2025
MIN_MAGNITUDE = 4.0

USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Output directories
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "usgs"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

RAW_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DOWNLOAD ONE YEAR
# ============================================================

def download_year(year):

    start_date = f"{year}-01-01"
    end_date = f"{year + 1}-01-01"

    output_file = RAW_DIR / f"usgs_{year}.csv"

    print()
    print("=" * 60)
    print(f"Downloading USGS earthquake data: {year}")
    print("=" * 60)

    params = {
        "format": "geojson",
        "starttime": start_date,
        "endtime": end_date,
        "minmagnitude": MIN_MAGNITUDE,
        "orderby": "time-asc",
        "limit": 20000
    }

    try:

        response = requests.get(
            USGS_URL,
            params=params,
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

    except requests.RequestException as e:

        print(f"ERROR downloading {year}: {e}")
        return None


    records = []


    # ========================================================
    # EXTRACT EVENTS
    # ========================================================

    for event in data.get("features", []):

        properties = event.get("properties", {})
        geometry = event.get("geometry")

        if geometry is None:
            continue

        coordinates = geometry.get("coordinates")

        if coordinates is None or len(coordinates) < 3:
            continue

        records.append({

            "event_id": event.get("id"),

            "time": pd.to_datetime(
                properties.get("time"),
                unit="ms",
                utc=True
            ),

            "longitude": coordinates[0],

            "latitude": coordinates[1],

            "depth_km": coordinates[2],

            "magnitude": properties.get("mag"),

            "magnitude_type": properties.get("magType"),

            "place": properties.get("place"),

            "event_type": properties.get("type"),

            "status": properties.get("status")
        })


    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    df = pd.DataFrame(records)


    # ========================================================
    # SAVE YEARLY FILE
    # ========================================================

    df.to_csv(
        output_file,
        index=False
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    print(f"Events downloaded : {len(df):,}")
    print(f"Saved to          : {output_file}")

    print()
    print("Missing values:")

    print(df.isna().sum())

    print()

    print(
        "Duplicate event IDs:",
        df["event_id"].duplicated().sum()
    )

    print()

    if len(df) > 0:

        print(
            "Magnitude range:",
            df["magnitude"].min(),
            "to",
            df["magnitude"].max()
        )

        print(
            "Depth range:",
            df["depth_km"].min(),
            "to",
            df["depth_km"].max()
        )

        print(
            "Event types:"
        )

        print(
            df["event_type"].value_counts()
        )

    return df


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print()
    print("=" * 60)
    print("TECTONICSENSE - USGS EARTHQUAKE DATA ACQUISITION")
    print("=" * 60)

    print()
    print(f"Year range     : {START_YEAR} - {END_YEAR}")
    print(f"Minimum magnitude: {MIN_MAGNITUDE}")
    print(f"Output folder  : {RAW_DIR}")

    all_data = []

    # ========================================================
    # DOWNLOAD EACH YEAR
    # ========================================================

    for year in range(START_YEAR, END_YEAR + 1):

        df = download_year(year)

        if df is not None and not df.empty:

            all_data.append(df)

        # Small delay between API requests
        time.sleep(1)


    # ========================================================
    # COMBINE ALL YEARS
    # ========================================================

    if not all_data:

        print()
        print("ERROR: No earthquake data downloaded.")
        return


    print()
    print("=" * 60)
    print("COMBINING DATA")
    print("=" * 60)

    usgs = pd.concat(
        all_data,
        ignore_index=True
    )


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    before = len(usgs)

    usgs = usgs.drop_duplicates(
        subset="event_id"
    )

    after = len(usgs)

    print()
    print(
        f"Duplicates removed: {before - after}"
    )


    # ========================================================
    # SORT BY TIME
    # ========================================================

    usgs = usgs.sort_values(
        "time"
    ).reset_index(
        drop=True
    )


    # ========================================================
    # SAVE COMBINED DATASET
    # ========================================================

    output_file = (
        INTERIM_DIR /
        "usgs_earthquakes_2010_2025_m4.csv"
    )

    usgs.to_csv(
        output_file,
        index=False
    )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 60)
    print("FINAL USGS DATASET")
    print("=" * 60)

    print()

    print(
        "Total earthquakes:",
        f"{len(usgs):,}"
    )

    print(
        "Unique earthquakes:",
        f"{usgs['event_id'].nunique():,}"
    )

    print()

    print(
        "Date range:"
    )

    print(
        usgs["time"].min(),
        "to",
        usgs["time"].max()
    )

    print()

    print(
        "Magnitude range:"
    )

    print(
        usgs["magnitude"].min(),
        "to",
        usgs["magnitude"].max()
    )

    print()

    print(
        "Depth range:"
    )

    print(
        usgs["depth_km"].min(),
        "to",
        usgs["depth_km"].max()
    )

    print()

    print(
        "Missing values:"
    )

    print(
        usgs.isna().sum()
    )

    print()

    print(
        "Event types:"
    )

    print(
        usgs["event_type"].value_counts()
    )

    print()

    print(
        "Magnitude types:"
    )

    print(
        usgs["magnitude_type"].value_counts()
    )

    print()
    print("=" * 60)
    print("USGS ACQUISITION COMPLETE")
    print("=" * 60)

    print()
    print(
        f"Combined dataset saved to:\n{output_file}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()