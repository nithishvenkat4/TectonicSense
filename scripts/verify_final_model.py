"""TectonicSense - Final Production Model Independent Verification

Phase 4: Final Model & Productionization
Validates that the final model artifact, feature schema, label mapping,
and production inference engine are functional, deterministic, structurally correct,
and robust to invalid inputs.

Outputs:
  - Console verification report
  - results/phase_4_final_model/final_model_verification.json
"""

from pathlib import Path
import json
import datetime
from typing import Dict, Any, List
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# Deterministic project paths
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "final" / "tectonicsense_xgboost.joblib"
FEATURES_PATH = ROOT_DIR / "models" / "final" / "feature_names.json"
LABELS_PATH = ROOT_DIR / "models" / "final" / "label_mapping.json"
METADATA_PATH = ROOT_DIR / "models" / "final" / "metadata.json"
TEST_DATA_PATH = ROOT_DIR / "data" / "processed" / "splits" / "test.csv"
OUTPUT_REPORT_PATH = ROOT_DIR / "results" / "phase_4_final_model" / "final_model_verification.json"

import sys
sys.path.insert(0, str(ROOT_DIR))

# Import production inference engine
from scripts.predict import TectonicPredictor, get_predictor


def run_verification() -> Dict[str, Any]:
    checks_passed = []
    checks_failed = []

    def record_check(name: str, passed: bool, detail: str = ""):
        if passed:
            checks_passed.append({"check": name, "detail": detail})
            print(f"[PASS] {name}")
        else:
            checks_failed.append({"check": name, "detail": detail})
            print(f"[FAIL] {name} - {detail}")

    print("=" * 60)
    print("TECTONICSENSE FINAL MODEL VERIFICATION")
    print("=" * 60)

    # -------------------------------------------------------------
    # 1. Artifact Existence & Loading Checks
    # -------------------------------------------------------------
    artifact_details = {}

    # Model artifact
    if MODEL_PATH.exists() and MODEL_PATH.is_file():
        try:
            model = joblib.load(MODEL_PATH)
            model_size = MODEL_PATH.stat().st_size
            artifact_details["model_artifact"] = {
                "path": str(MODEL_PATH.relative_to(ROOT_DIR)),
                "size_bytes": model_size,
                "loaded": True,
            }
            record_check("Model artifact loaded", True, f"Size: {model_size} bytes")
        except Exception as e:
            record_check("Model artifact loaded", False, str(e))
            model = None
    else:
        record_check("Model artifact loaded", False, f"File not found: {MODEL_PATH}")
        model = None

    # Feature metadata
    feature_names = []
    if FEATURES_PATH.exists():
        try:
            with open(FEATURES_PATH, "r", encoding="utf-8") as f:
                feat_data = json.load(f)
            feature_names = feat_data.get("features", [])
            artifact_details["feature_names"] = {
                "path": str(FEATURES_PATH.relative_to(ROOT_DIR)),
                "count": len(feature_names),
                "loaded": True,
            }
            record_check("Feature metadata loaded", True, f"{len(feature_names)} features")
        except Exception as e:
            record_check("Feature metadata loaded", False, str(e))
    else:
        record_check("Feature metadata loaded", False, f"File not found: {FEATURES_PATH}")

    # Label mapping
    label_mapping = {}
    if LABELS_PATH.exists():
        try:
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                raw_labels = json.load(f)
            label_mapping = {int(k): str(v) for k, v in raw_labels.items()}
            artifact_details["label_mapping"] = {
                "path": str(LABELS_PATH.relative_to(ROOT_DIR)),
                "classes": list(label_mapping.values()),
                "loaded": True,
            }
            record_check("Label mapping loaded", True, f"{len(label_mapping)} classes")
        except Exception as e:
            record_check("Label mapping loaded", False, str(e))
    else:
        record_check("Label mapping loaded", False, f"File not found: {LABELS_PATH}")

    # Metadata
    meta_info = {}
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                meta_info = json.load(f)
            artifact_details["metadata"] = {
                "path": str(METADATA_PATH.relative_to(ROOT_DIR)),
                "model_name": meta_info.get("model_name"),
                "framework_version": meta_info.get("xgboost_version"),
                "loaded": True,
            }
            record_check("Metadata loaded", True, f"Model: {meta_info.get('model_name')}")
        except Exception as e:
            record_check("Metadata loaded", False, str(e))
    else:
        record_check("Metadata loaded", False, f"File not found: {METADATA_PATH}")

    # -------------------------------------------------------------
    # 2. Model Structure Checks
    # -------------------------------------------------------------
    structure_details = {}

    is_xgb = isinstance(model, XGBClassifier)
    structure_details["is_xgb_classifier"] = is_xgb
    record_check("Model type verified", is_xgb, f"Type: {type(model).__name__}")

    n_features = getattr(model, "n_features_in_", None)
    features_match = (n_features == 26) and (len(feature_names) == 26)
    structure_details["n_features_in"] = int(n_features) if n_features is not None else None
    structure_details["feature_list_length"] = len(feature_names)
    record_check("26 features verified", features_match, f"Model expects {n_features}, schema defines {len(feature_names)}")

    n_classes = getattr(model, "n_classes_", None)
    classes_attr = list(getattr(model, "classes_", []))
    classes_match = (n_classes == 7) and (len(label_mapping) == 7) and (list(classes_attr) == list(range(7)))
    structure_details["n_classes"] = int(n_classes) if n_classes is not None else None
    structure_details["classes"] = [int(c) for c in classes_attr]
    structure_details["label_mapping_classes"] = list(label_mapping.values())
    record_check("7 classes verified", classes_match, f"Classes: {classes_attr}, Labels: {list(label_mapping.values())}")

    # -------------------------------------------------------------
    # 3. Production Inference Checks
    # -------------------------------------------------------------
    inference_details = {}
    predictor = get_predictor()

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Test split dataset not found: {TEST_DATA_PATH}")

    test_df = pd.read_csv(TEST_DATA_PATH, nrows=10)
    sample_obs = test_df.iloc[0].drop("boundary_class").to_dict()
    valid_classes = set(label_mapping.values())

    # Single inference
    try:
        single_res = predictor.predict_single(sample_obs)
        pred_class = single_res["predicted_class"]
        pred_id = single_res["predicted_class_id"]
        conf = single_res["confidence"]
        probs = single_res["class_probabilities"]

        single_valid = (
            pred_class in valid_classes
            and pred_id in label_mapping
            and label_mapping[pred_id] == pred_class
            and 0.0 <= conf <= 1.0
            and len(probs) == 7
            and set(probs.keys()) == valid_classes
            and not any(np.isnan(v) or np.isinf(v) for v in probs.values())
        )
        inference_details["single_inference"] = {
            "predicted_class": pred_class,
            "predicted_class_id": pred_id,
            "confidence": conf,
            "valid": single_valid,
        }
        record_check("Single inference verified", single_valid, f"Class: {pred_class}, Conf: {conf:.6f}")
    except Exception as e:
        record_check("Single inference verified", False, str(e))
        single_res = None

    # Batch inference (10 rows)
    try:
        batch_input = test_df.drop(columns=["boundary_class"])
        batch_results = predictor.predict_batch(batch_input)
        batch_valid = (
            len(batch_results) == 10
            and all(r["predicted_class"] in valid_classes for r in batch_results)
            and all(0.0 <= r["confidence"] <= 1.0 for r in batch_results)
            and all(len(r["class_probabilities"]) == 7 for r in batch_results)
        )
        inference_details["batch_inference"] = {
            "batch_size": len(batch_results),
            "all_valid": batch_valid,
        }
        record_check("Batch inference verified", batch_valid, f"10/10 test samples evaluated successfully")
    except Exception as e:
        record_check("Batch inference verified", False, str(e))

    # Probability Normalization Check
    if single_res:
        prob_sum = float(sum(single_res["class_probabilities"].values()))
        is_normalized = bool(np.isclose(prob_sum, 1.0, atol=1e-5))
        inference_details["probability_normalization"] = {
            "probability_sum": prob_sum,
            "is_normalized": is_normalized,
        }
        record_check("Probability normalization verified", is_normalized, f"Sum: {prob_sum:.8f}")
    else:
        record_check("Probability normalization verified", False, "Single prediction failed")

    # -------------------------------------------------------------
    # 4. Determinism Checks
    # -------------------------------------------------------------
    determinism_details = {}
    try:
        res_a = predictor.predict_single(sample_obs)
        res_b = predictor.predict_single(sample_obs)

        same_class = res_a["predicted_class"] == res_b["predicted_class"]
        same_id = res_a["predicted_class_id"] == res_b["predicted_class_id"]
        same_conf = np.isclose(res_a["confidence"], res_b["confidence"], atol=1e-7)

        max_prob_diff = max(
            abs(res_a["class_probabilities"][c] - res_b["class_probabilities"][c])
            for c in valid_classes
        )
        prob_identical = max_prob_diff < 1e-7

        deterministic = bool(same_class and same_id and same_conf and prob_identical)
        determinism_details = {
            "same_class": same_class,
            "same_class_id": same_id,
            "same_confidence": bool(same_conf),
            "max_probability_divergence": float(max_prob_diff),
            "deterministic": deterministic,
        }
        record_check(
            "Deterministic prediction verified",
            deterministic,
            f"Class identical: {same_class}, Max prob diff: {max_prob_diff:.2e}",
        )
    except Exception as e:
        record_check("Deterministic prediction verified", False, str(e))

    # -------------------------------------------------------------
    # 5. Invalid Input Handling Checks
    # -------------------------------------------------------------
    invalid_input_details = {}

    # Test 1: Missing feature
    try:
        bad_missing = {k: v for k, v in sample_obs.items() if k != "magnitude"}
        predictor.predict_single(bad_missing)
        record_check("Missing feature rejection verified", False, "Failed to reject missing feature")
        invalid_input_details["missing_feature_rejected"] = False
    except ValueError as e:
        record_check("Missing feature rejection verified", True, f"Rejected cleanly: {type(e).__name__}")
        invalid_input_details["missing_feature_rejected"] = True
    except Exception as e:
        record_check("Missing feature rejection verified", False, f"Unexpected error type: {type(e).__name__}")
        invalid_input_details["missing_feature_rejected"] = False

    # Test 2: Non-numeric input
    try:
        bad_non_numeric = sample_obs.copy()
        bad_non_numeric["magnitude"] = "extremely_high"
        predictor.predict_single(bad_non_numeric)
        record_check("Non-numeric input rejection verified", False, "Failed to reject non-numeric input")
        invalid_input_details["non_numeric_rejected"] = False
    except ValueError as e:
        record_check("Non-numeric input rejection verified", True, f"Rejected cleanly: {type(e).__name__}")
        invalid_input_details["non_numeric_rejected"] = True
    except Exception as e:
        record_check("Non-numeric input rejection verified", False, f"Unexpected error type: {type(e).__name__}")
        invalid_input_details["non_numeric_rejected"] = False

    # Test 3: NaN feature
    try:
        bad_nan = sample_obs.copy()
        bad_nan["depth_km"] = np.nan
        predictor.predict_single(bad_nan)
        record_check("NaN rejection verified", False, "Failed to reject NaN value")
        invalid_input_details["nan_rejected"] = False
    except ValueError as e:
        record_check("NaN rejection verified", True, f"Rejected cleanly: {type(e).__name__}")
        invalid_input_details["nan_rejected"] = True
    except Exception as e:
        record_check("NaN rejection verified", False, f"Unexpected error type: {type(e).__name__}")
        invalid_input_details["nan_rejected"] = False

    # Test 4: Infinite feature
    try:
        bad_inf = sample_obs.copy()
        bad_inf["relative_velocity"] = np.inf
        predictor.predict_single(bad_inf)
        record_check("Infinite-value rejection verified", False, "Failed to reject infinite value")
        invalid_input_details["infinite_rejected"] = False
    except ValueError as e:
        record_check("Infinite-value rejection verified", True, f"Rejected cleanly: {type(e).__name__}")
        invalid_input_details["infinite_rejected"] = True
    except Exception as e:
        record_check("Infinite-value rejection verified", False, f"Unexpected error type: {type(e).__name__}")
        invalid_input_details["infinite_rejected"] = False

    # Test 5: Incorrect feature count (subset / truncated schema)
    try:
        bad_count = {k: sample_obs[k] for k in list(sample_obs.keys())[:20]}
        predictor.predict_single(bad_count)
        record_check("Incorrect feature-count rejection verified", False, "Failed to reject incorrect feature count")
        invalid_input_details["incorrect_feature_count_rejected"] = False
    except ValueError as e:
        record_check("Incorrect feature-count rejection verified", True, f"Rejected cleanly: {type(e).__name__}")
        invalid_input_details["incorrect_feature_count_rejected"] = True
    except Exception as e:
        record_check("Incorrect feature-count rejection verified", False, f"Unexpected error type: {type(e).__name__}")
        invalid_input_details["incorrect_feature_count_rejected"] = False

    # -------------------------------------------------------------
    # Overall Assessment & Report Generation
    # -------------------------------------------------------------
    all_passed = (len(checks_failed) == 0) and (len(checks_passed) >= 16)
    overall_status = "VERIFIED" if all_passed else "FAILED"

    print("\n" + "=" * 60)
    print(f"FINAL STATUS: {overall_status}")
    print("=" * 60)

    report = {
        "status": overall_status,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": {
            "type": type(model).__name__ if model else "Unknown",
            "framework": meta_info.get("framework", "xgboost"),
            "framework_version": meta_info.get("xgboost_version", "3.0.5"),
            "feature_count": len(feature_names),
            "class_count": len(label_mapping),
        },
        "artifact_checks": artifact_details,
        "structure_checks": structure_details,
        "inference_checks": inference_details,
        "determinism_checks": determinism_details,
        "invalid_input_checks": invalid_input_details,
        "checks_passed_count": len(checks_passed),
        "checks_failed_count": len(checks_failed),
        "failed_checks": checks_failed,
        "all_checks_passed": all_passed,
    }

    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"\nVerification report saved to: {OUTPUT_REPORT_PATH.relative_to(ROOT_DIR)}")
    return report


if __name__ == "__main__":
    run_verification()
