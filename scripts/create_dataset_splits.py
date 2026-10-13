from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# TectonicSense - Dataset Splitting
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "processed" / "tectonicsense_ml_dataset.csv"

OUTPUT_DIR = ROOT / "data" / "processed" / "splits"
REPORT_FILE = ROOT / "results" / "metrics" / "dataset_split_report.json"
DOC_FILE = ROOT / "docs" / "dataset_splitting.md"

TARGET = "boundary_class"

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


def load_dataset():
    print("=" * 60)
    print("TectonicSense Dataset Splitting")
    print("=" * 60)

    print("\nLoading ML dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns)}")

    return df


def validate_dataset(df):

    print("\nValidating dataset...")

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")

    missing = df.isnull().sum().sum()

    if missing > 0:
        raise ValueError(
            f"Dataset contains {missing} missing values."
        )

    duplicate_rows = df.duplicated().sum()

    print(f"Missing values: {missing}")
    print(f"Duplicate rows: {duplicate_rows}")

    print("\nOriginal class distribution:")
    print(df[TARGET].value_counts())


def create_splits(df):

    print("\nCreating 70/15/15 stratified split...")

    train_df, temp_df = train_test_split(
        df,
        test_size=(VALIDATION_SIZE + TEST_SIZE),
        stratify=df[TARGET],
        random_state=RANDOM_STATE
    )

    validation_relative_size = (
        VALIDATION_SIZE /
        (VALIDATION_SIZE + TEST_SIZE)
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - validation_relative_size),
        stratify=temp_df[TARGET],
        random_state=RANDOM_STATE
    )

    return train_df, validation_df, test_df


def print_split_summary(name, df):

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    print(f"Rows: {len(df):,}")
    print(f"Percentage: {len(df) / TOTAL_ROWS * 100:.2f}%")

    print("\nClass distribution:")

    counts = df[TARGET].value_counts()
    percentages = df[TARGET].value_counts(normalize=True) * 100

    summary = pd.DataFrame({
        "count": counts,
        "percentage": percentages.round(2)
    })

    print(summary)


def save_splits(train_df, validation_df, test_df):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_file = OUTPUT_DIR / "train.csv"
    validation_file = OUTPUT_DIR / "validation.csv"
    test_file = OUTPUT_DIR / "test.csv"

    train_df.to_csv(train_file, index=False)
    validation_df.to_csv(validation_file, index=False)
    test_df.to_csv(test_file, index=False)

    print("\nSplit datasets saved:")
    print(train_file)
    print(validation_file)
    print(test_file)

    return train_file, validation_file, test_file


def generate_report(
    train_df,
    validation_df,
    test_df,
    train_file,
    validation_file,
    test_file
):

    def class_distribution(df):

        counts = df[TARGET].value_counts()

        return {
            str(k): int(v)
            for k, v in counts.items()
        }

    report = {
        "project": "TectonicSense",
        "stage": "Dataset Splitting",

        "input_file": str(INPUT_FILE),

        "total_rows": int(TOTAL_ROWS),

        "split_strategy": "Stratified random split",

        "random_state": RANDOM_STATE,

        "ratios": {
            "train": TRAIN_SIZE,
            "validation": VALIDATION_SIZE,
            "test": TEST_SIZE
        },

        "sizes": {
            "train": int(len(train_df)),
            "validation": int(len(validation_df)),
            "test": int(len(test_df))
        },

        "files": {
            "train": str(train_file),
            "validation": str(validation_file),
            "test": str(test_file)
        },

        "class_distribution": {
            "train": class_distribution(train_df),
            "validation": class_distribution(validation_df),
            "test": class_distribution(test_df)
        },

        "target": TARGET
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\nReport saved:")
    print(REPORT_FILE)


def generate_documentation():

    documentation = """# TectonicSense Dataset Splitting

## Objective

The engineered TectonicSense dataset is divided into training,
validation, and test subsets for reproducible machine learning
experiments.

## Split Strategy

A stratified random split is used:

- Training: 70%
- Validation: 15%
- Test: 15%

Stratification is performed using `boundary_class` so that the
seven tectonic boundary classes remain proportionally represented
across the three subsets.

## Reproducibility

Random state:

`42`

Using a fixed random state ensures that the same dataset produces
the same split during repeated experiments.

## Evaluation Design

The stratified random split provides the primary baseline
evaluation.

A separate boundary-aware evaluation will be performed later
using `boundary_id` as a grouping variable. This experiment will
test whether the trained models generalize to previously unseen
tectonic boundaries.

`boundary_id` is not used as a predictive ML feature.

## Target

The prediction target is:

`boundary_class`

The seven classes are:

- CCB
- CRB
- CTF
- OCB
- OSR
- OTF
- SUB
"""

    with open(DOC_FILE, "w", encoding="utf-8") as f:
        f.write(documentation)

    print("\nDocumentation saved:")
    print(DOC_FILE)


def main():

    global TOTAL_ROWS

    df = load_dataset()

    TOTAL_ROWS = len(df)

    validate_dataset(df)

    train_df, validation_df, test_df = create_splits(df)

    print_split_summary("TRAINING SET", train_df)
    print_split_summary("VALIDATION SET", validation_df)
    print_split_summary("TEST SET", test_df)

    train_file, validation_file, test_file = save_splits(
        train_df,
        validation_df,
        test_df
    )

    generate_report(
        train_df,
        validation_df,
        test_df,
        train_file,
        validation_file,
        test_file
    )

    generate_documentation()

    print("\n" + "=" * 60)
    print("DATASET SPLITTING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()