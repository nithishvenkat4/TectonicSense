"""TectonicSense - Best Candidate Model Verification Script

Loads the serialized candidate model, feature list, and label mapping.
Validates model prediction on a batch of validation samples.
"""

from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models" / "phase_2_best_candidate"
VAL_CSV = ROOT_DIR / "data" / "processed" / "splits" / "validation.csv"


def verify_model():
    print("=" * 70)
    print("VERIFYING SERIALIZED PHASE 2 CANDIDATE MODEL")
    print("=" * 70)

    model_path = MODEL_DIR / "best_candidate_model.joblib"
    meta_path = MODEL_DIR / "metadata.json"
    feat_path = MODEL_DIR / "feature_names.json"
    label_path = MODEL_DIR / "label_mapping.json"

    # 1. Check artifact existence
    for p in [model_path, meta_path, feat_path, label_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {p}")
        print(f"Artifact found: {p.name}")

    # 2. Load model
    model = joblib.load(model_path)
    print(f"Model loaded successfully: {type(model).__name__}")

    # 3. Load feature names & label mapping
    with open(feat_path, "r") as f:
        features_info = json.load(f)
        features = features_info["features"]
    print(f"Feature count verified: {len(features)}")

    with open(label_path, "r") as f:
        label_mapping = json.load(f)
    print(f"Label classes: {list(label_mapping.values())}")

    # 4. Load sample validation data
    val_df = pd.read_csv(VAL_CSV, nrows=25)
    X_sample = val_df[features].copy()
    y_sample_true = val_df["boundary_class"].copy()

    # 5. Generate predictions
    preds_raw = model.predict(X_sample)
    if preds_raw.ndim > 1 and preds_raw.shape[1] > 1:
        preds_indices = np.argmax(preds_raw, axis=1)
    else:
        preds_indices = np.asarray(preds_raw).ravel().astype(int)

    # 6. Verify predictions
    int_str_map = {int(k): v for k, v in label_mapping.items()}
    pred_class_names = [int_str_map[idx] for idx in preds_indices]

    print("\nSample Predictions vs Ground Truth (first 5):")
    for i in range(5):
        print(f"  Sample {i+1:02d}: Predicted={pred_class_names[i]:<4} | True={y_sample_true.iloc[i]:<4} | Match={pred_class_names[i] == y_sample_true.iloc[i]}")

    # Shape and validity checks
    assert len(pred_class_names) == len(X_sample), "Prediction length mismatch!"
    valid_classes = set(label_mapping.values())
    for pred in pred_class_names:
        assert pred in valid_classes, f"Invalid class predicted: {pred}"

    print("\n" + "=" * 70)
    print("VERIFICATION SUCCESS: Candidate model is intact, functional, and ready for Phase 3 evaluation!")
    print("=" * 70)


if __name__ == "__main__":
    verify_model()
