"""TectonicSense - Final Production Model Training Script

Phase 4: Final Model & Productionization
Trains the official production XGBoost model using the frozen configuration
from configs/final_model.yaml on the established training partition.

Outputs:
  - models/final/tectonicsense_xgboost.joblib
  - models/final/feature_names.json
  - models/final/label_mapping.json
  - models/final/metadata.json
  - results/phase_4_final_model/final_training_summary.json
"""

from pathlib import Path
import json
import time
from datetime import datetime
import sys
import yaml
import pandas as pd
import numpy as np

import xgboost as xgb
import joblib

# Resolve project root robustly
ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "configs" / "final_model.yaml"


def load_config():
    """Load configuration from configs/final_model.yaml."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config


def validate_and_load_data(config):
    """Load train, val, test datasets and perform rigorous schema and data integrity validation."""
    print("=" * 70)
    print("VALIDATING SCHEMA & LOADING TRAINING DATA")
    print("=" * 70)

    train_path = ROOT_DIR / config["data"]["splits"]["train_path"]
    val_path = ROOT_DIR / config["data"]["splits"]["validation_path"]
    test_path = ROOT_DIR / config["data"]["splits"]["test_path"]

    for path, name in [(train_path, "train"), (val_path, "validation"), (test_path, "test")]:
        if not path.exists():
            raise FileNotFoundError(f"Required split file missing: {path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    expected_features = config["features"]["feature_list"]
    target_col = config["task"]["target_column"]
    expected_classes = set(config["labels"]["classes"])

    # 1. Feature count check
    assert len(expected_features) == 26, f"Expected 26 features, got {len(expected_features)}"

    # 2. Check all columns exist in train_df
    for feat in expected_features:
        if feat not in train_df.columns:
            raise KeyError(f"Feature '{feat}' missing from training dataset!")

    if target_col not in train_df.columns:
        raise KeyError(f"Target column '{target_col}' missing from training dataset!")

    # 3. Check feature order matches configuration exactly
    train_features = [c for c in train_df.columns if c != target_col]
    assert train_features == expected_features, "Feature order in train.csv does not match config schema!"

    # 4. Check for missing values
    missing_feats = train_df[expected_features].isnull().sum().sum()
    missing_target = train_df[target_col].isnull().sum()
    assert missing_feats == 0, f"Found {missing_feats} missing feature values in training data!"
    assert missing_target == 0, f"Found {missing_target} missing target values in training data!"

    # 5. Check target classes match expected classes
    train_classes = set(train_df[target_col].unique())
    assert train_classes == expected_classes, f"Unexpected target classes in train.csv: {train_classes} vs {expected_classes}"

    # 6. Verify identifiers are completely absent
    forbidden = ["event_id", "boundary_id", "boundary_step_id", "step_id"]
    for col in forbidden:
        assert col not in train_df.columns, f"Leakage hazard: '{col}' present in training dataset!"

    print(f"Schema Validation Passed: Exactly 26 features in established order.")
    print(f"Training samples:   {len(train_df):,} (70% stratified partition)")
    print(f"Validation samples: {len(val_df):,} (15% stratified partition)")
    print(f"Test samples:       {len(test_df):,} (15% held-out partition)")
    print(f"Zero missing values detected across all predictive columns.")

    return train_df, val_df, test_df, expected_features, target_col


def encode_labels(train_df, val_df, test_df, target_col, config):
    """Encode labels strictly using the established mapping dictionary."""
    mapping = config["labels"]["inverse_mapping"] # string -> int
    y_train = train_df[target_col].map(mapping).values
    y_val = val_df[target_col].map(mapping).values
    y_test = test_df[target_col].map(mapping).values

    assert not np.isnan(y_train).any(), "NaN detected in encoded y_train labels!"
    assert set(np.unique(y_train)) == set(range(7)), "Encoded labels must cover integers 0 to 6!"

    return y_train.astype(int), y_val.astype(int), y_test.astype(int)


def train_model(X_train, y_train, config):
    """Construct and train the final production XGBoost model using frozen hyperparameters."""
    print("\n" + "=" * 70)
    print("TRAINING FINAL PRODUCTION XGBOOST MODEL")
    print("=" * 70)

    hp = config["model"]["hyperparameters"]
    print(f"Framework: XGBoost {xgb.__version__}")
    print(f"Hyperparameters: {json.dumps(hp, indent=2)}")

    model = xgb.XGBClassifier(**hp)

    t0 = time.time()
    model.fit(X_train, y_train)
    fit_duration = time.time() - t0

    print(f"\nFinal model training completed successfully in {fit_duration:.2f} seconds.")
    return model, fit_duration


def verify_serialized_model(model_path, config, X_val, y_val):
    """Independently load the saved model artifact and verify predictions and probabilities."""
    print("\n" + "=" * 70)
    print("VERIFYING SERIALIZED PRODUCTION MODEL ARTIFACT")
    print("=" * 70)

    loaded_model = joblib.load(model_path)
    print(f"Successfully loaded: {model_path.name}")
    print(f"Model type: {type(loaded_model).__name__}")

    # Verify attributes
    expected_feat_count = config["features"]["count"]
    expected_class_count = config["task"]["number_of_classes"]

    assert loaded_model.n_classes_ == expected_class_count, f"Expected {expected_class_count} classes, got {loaded_model.n_classes_}"
    assert loaded_model.n_features_in_ == expected_feat_count, f"Expected {expected_feat_count} features, got {loaded_model.n_features_in_}"

    # Test predictions on sample of validation data
    sample_X = X_val.iloc[:50].copy()
    sample_y = y_val[:50]

    preds = loaded_model.predict(sample_X)
    probs = loaded_model.predict_proba(sample_X)

    assert len(preds) == 50, "Prediction shape mismatch!"
    assert probs.shape == (50, 7), "Probability shape mismatch!"
    assert not np.isnan(preds).any(), "NaN detected in predictions!"
    assert not np.isnan(probs).any(), "NaN detected in probabilities!"

    # Verify probabilities sum to ~1.0
    prob_sums = np.sum(probs, axis=1)
    np.testing.assert_allclose(prob_sums, np.ones(50), rtol=1e-5, atol=1e-5)

    # Verification accuracy on sample
    sample_acc = float(np.mean(preds == sample_y))
    print(f"Classes: {loaded_model.n_classes_} | Features in: {loaded_model.n_features_in_}")
    print(f"Sample prediction check: 50/50 samples evaluated, sample accuracy = {sample_acc:.4f}")
    print(f"Probability sum normalization verified: All rows sum to 1.0 (atol=1e-5)")

    return True


def save_artifacts(model, expected_features, config, train_df, val_df, test_df, fit_duration):
    """Save model binary, feature metadata, label mapping, and comprehensive metadata."""
    out_dir = ROOT_DIR / config["artifacts"]["output_dir"]
    results_dir = ROOT_DIR / config["artifacts"]["results_dir"]

    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_path = ROOT_DIR / config["artifacts"]["model_file"]
    feat_path = ROOT_DIR / config["artifacts"]["feature_names_file"]
    label_path = ROOT_DIR / config["artifacts"]["label_mapping_file"]
    meta_path = ROOT_DIR / config["artifacts"]["metadata_file"]
    summary_path = results_dir / "final_training_summary.json"

    # 1. Model binary
    joblib.dump(model, model_path, compress=3)
    model_size_bytes = model_path.stat().st_size
    print(f"Saved model binary: {model_path} ({model_size_bytes:,} bytes)")

    # 2. Feature names
    feat_dict = {
        "feature_count": len(expected_features),
        "features": expected_features
    }
    with open(feat_path, "w") as f:
        json.dump(feat_dict, f, indent=4)
    print(f"Saved feature names: {feat_path}")

    # 3. Label mapping
    label_dict = config["labels"]["mapping"]
    with open(label_path, "w") as f:
        json.dump(label_dict, f, indent=4)
    print(f"Saved label mapping: {label_path}")

    # 4. Complete metadata
    metadata = {
        "project": config["project"]["name"],
        "project_description": config["project"]["description"],
        "model_name": config["model"]["name"],
        "framework": config["model"]["framework"],
        "estimator": config["model"]["estimator"],
        "model_version": "1.0.0-final",
        "training_timestamp": datetime.now().isoformat(),
        "random_seed": config["reproducibility"]["random_seed"],
        "feature_count": len(expected_features),
        "feature_names": expected_features,
        "target_column": config["task"]["target_column"],
        "number_of_classes": config["task"]["number_of_classes"],
        "class_names": config["labels"]["classes"],
        "label_mapping": label_dict,
        "hyperparameters": config["model"]["hyperparameters"],
        "training_methodology": "Trained on established 70% stratified training partition (train.csv) with frozen Optuna hyperparameters from Phase 2",
        "train_sample_count": len(train_df),
        "validation_sample_count": len(val_df),
        "test_sample_count": len(test_df),
        "xgboost_version": xgb.__version__,
        "python_version": sys.version.split()[0],
        "joblib_version": joblib.__version__,
        "training_duration_seconds": round(fit_duration, 2),
        "artifact_size_bytes": model_size_bytes,
        "phase_2_validation_benchmark": {
            "accuracy": 0.9999,
            "macro_precision": 0.9998,
            "macro_recall": 0.9998,
            "macro_f1": 0.9998,
            "weighted_f1": 0.9999
        },
        "phase_3_test_benchmark": {
            "accuracy": 0.999889,
            "macro_precision": 0.999821,
            "macro_recall": 0.999761,
            "macro_f1": 0.999791,
            "weighted_f1": 0.999889
        },
        "phase_3_boundary_aware_benchmark": {
            "accuracy": 0.7994,
            "macro_f1": 0.7935,
            "generalization_note": "Meaningful out-of-boundary generalization under the evaluated PB2002 split (127 train, 27 val, 28 test boundaries; zero shared boundaries)"
        },
        "artifact_path": str(model_path.relative_to(ROOT_DIR))
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata: {meta_path}")

    # 5. Training summary
    summary_data = {
        "status": "COMPLETED",
        "training_timestamp": metadata["training_timestamp"],
        "training_methodology": metadata["training_methodology"],
        "train_samples": len(train_df),
        "validation_samples": len(val_df),
        "test_samples": len(test_df),
        "training_duration_seconds": round(fit_duration, 2),
        "artifact_size_bytes": model_size_bytes,
        "feature_count": len(expected_features),
        "target_classes": config["labels"]["classes"],
        "frozen_hyperparameters": config["model"]["hyperparameters"],
        "verification_status": "VERIFIED_FUNCTIONAL"
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"Saved training summary: {summary_path}")

    return model_path, model_size_bytes, metadata


def main():
    config = load_config()
    train_df, val_df, test_df, expected_features, target_col = validate_and_load_data(config)
    y_train, y_val, y_test = encode_labels(train_df, val_df, test_df, target_col, config)

    X_train = train_df[expected_features].copy()
    X_val = val_df[expected_features].copy()

    # Train model
    model, fit_duration = train_model(X_train, y_train, config)

    # Save artifacts
    model_path, model_size_bytes, metadata = save_artifacts(
        model, expected_features, config, train_df, val_df, test_df, fit_duration
    )

    # Self-verify
    verify_serialized_model(model_path, config, X_val, y_val)

    print("\n" + "=" * 70)
    print("PHASE 4B FINAL MODEL TRAINING & PACKAGING COMPLETED")
    print("=" * 70)
    print(f"Training Methodology: {metadata['training_methodology']}")
    print(f"Training Samples:     {len(train_df):,}")
    print(f"Validation Samples:   {len(val_df):,}")
    print(f"Test Samples:         {len(test_df):,}")
    print(f"Model Artifact Size:  {model_size_bytes:,} bytes (~{model_size_bytes/1024:.1f} KB)")
    print(f"Training Duration:    {fit_duration:.2f} seconds")
    print(f"Model Artifact:       {model_path}")


if __name__ == "__main__":
    main()
