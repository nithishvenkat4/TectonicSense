"""TectonicSense - Comprehensive FastAPI Test Suite

Phase 5B: Automated API Verification
Tests all endpoints, schema validation, batch limits, determinism,
OpenAPI/Swagger docs, and error handling.
"""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api.main import app

# Path to test dataset
ROOT_DIR = Path(__file__).resolve().parents[2]
TEST_DATA_PATH = ROOT_DIR / "data" / "processed" / "splits" / "test.csv"


@pytest.fixture(scope="module")
def client():
    """Create reusable FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def valid_sample_records():
    """Load real, validated earthquake records from the test split."""
    if not TEST_DATA_PATH.exists():
        pytest.fail(f"Test split dataset not found at: {TEST_DATA_PATH}")

    df = pd.read_csv(TEST_DATA_PATH, nrows=10)
    features_df = df.drop(columns=["boundary_class"])
    records = features_df.to_dict(orient="records")
    true_labels = df["boundary_class"].tolist()
    return records, true_labels


# ==============================================================================
# 1. Health Endpoint Tests
# ==============================================================================

def test_health_check(client):
    """Test GET /health returns 200 and reports healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "TectonicSense API"
    assert data["model_loaded"] is True


# ==============================================================================
# 2. Model Info Endpoint Tests
# ==============================================================================

def test_model_info(client):
    """Test GET /model-info returns 200, 26 features, 7 classes, and model name."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "Tuned XGBoost"
    assert data["estimator"] == "XGBClassifier"
    assert data["feature_count"] == 26
    assert data["class_count"] == 7
    expected_classes = ["CCB", "CRB", "CTF", "OCB", "OSR", "OTF", "SUB"]
    assert sorted(data["class_names"]) == sorted(expected_classes)
    assert "model_version" in data


# ==============================================================================
# 3. Single Prediction Tests
# ==============================================================================

def test_predict_single_valid(client, valid_sample_records):
    """Test POST /predict with a valid observation."""
    records, true_labels = valid_sample_records
    sample = records[0]

    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    data = response.json()

    valid_classes = {"CCB", "CRB", "CTF", "OCB", "OSR", "OTF", "SUB"}
    assert data["predicted_class"] in valid_classes
    assert isinstance(data["class_id"], int)
    assert 0 <= data["class_id"] < 7
    assert 0.0 <= data["confidence"] <= 1.0

    probs = data["class_probabilities"]
    assert len(probs) == 7
    assert set(probs.keys()) == valid_classes
    prob_sum = sum(probs.values())
    assert np.isclose(prob_sum, 1.0, atol=1e-5)
    assert np.isclose(data["confidence"], probs[data["predicted_class"]], atol=1e-6)


def test_predict_single_missing_feature(client, valid_sample_records):
    """Test POST /predict rejects payloads missing a required feature (HTTP 422)."""
    records, _ = valid_sample_records
    bad_sample = records[0].copy()
    del bad_sample["magnitude"]

    response = client.post("/predict", json=bad_sample)
    assert response.status_code == 422


def test_predict_single_non_numeric(client, valid_sample_records):
    """Test POST /predict rejects non-numeric feature values (HTTP 422)."""
    records, _ = valid_sample_records
    bad_sample = records[0].copy()
    bad_sample["magnitude"] = "extremely_high_magnitude"

    response = client.post("/predict", json=bad_sample)
    assert response.status_code == 422


def test_predict_single_nan(client, valid_sample_records):
    """Test POST /predict rejects NaN values (HTTP 422)."""
    records, _ = valid_sample_records
    bad_sample = records[0].copy()
    bad_sample["depth_km"] = float("nan")

    raw_json = json.dumps(bad_sample, allow_nan=True)
    response = client.post(
        "/predict",
        content=raw_json,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_predict_single_infinity(client, valid_sample_records):
    """Test POST /predict rejects infinite values (HTTP 422)."""
    records, _ = valid_sample_records
    bad_sample = records[0].copy()
    bad_sample["relative_velocity"] = float("inf")

    raw_json = json.dumps(bad_sample, allow_nan=True)
    response = client.post(
        "/predict",
        content=raw_json,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_predict_single_extra_forbidden_fields(client, valid_sample_records):
    """Test POST /predict rejects payloads containing forbidden/leakage fields (HTTP 422)."""
    records, _ = valid_sample_records
    bad_sample = records[0].copy()
    bad_sample["boundary_id"] = 101  # forbidden
    bad_sample["boundary_class"] = "SUB"  # target label leakage forbidden

    response = client.post("/predict", json=bad_sample)
    assert response.status_code == 422


# ==============================================================================
# 4. Batch Prediction Tests
# ==============================================================================

def test_predict_batch_valid(client, valid_sample_records):
    """Test POST /predict-batch with 5 valid observations."""
    records, _ = valid_sample_records
    five_records = records[:5]

    response = client.post(
        "/predict-batch",
        json={"observations": five_records},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["count"] == 5
    assert len(data["predictions"]) == 5

    valid_classes = {"CCB", "CRB", "CTF", "OCB", "OSR", "OTF", "SUB"}
    for pred in data["predictions"]:
        assert pred["predicted_class"] in valid_classes
        assert 0 <= pred["class_id"] < 7
        assert 0.0 <= pred["confidence"] <= 1.0
        assert len(pred["class_probabilities"]) == 7


def test_predict_batch_order_preservation(client, valid_sample_records):
    """Verify that batch prediction strictly preserves input row ordering."""
    records, _ = valid_sample_records
    subset = records[:3]

    # Predict individually
    single_preds = [client.post("/predict", json=r).json()["predicted_class"] for r in subset]

    # Predict in batch
    batch_resp = client.post("/predict-batch", json={"observations": subset})
    assert batch_resp.status_code == 200
    batch_preds = [p["predicted_class"] for p in batch_resp.json()["predictions"]]

    assert single_preds == batch_preds


def test_predict_batch_malformed(client, valid_sample_records):
    """Test POST /predict-batch rejects batches containing malformed observations (HTTP 422)."""
    records, _ = valid_sample_records
    malformed_batch = records[:2]
    # Invalidate 2nd item
    bad_item = malformed_batch[1].copy()
    del bad_item["latitude"]
    payload = {"observations": [malformed_batch[0], bad_item]}

    response = client.post("/predict-batch", json=payload)
    assert response.status_code == 422


def test_predict_batch_exceeds_max_limit(client, valid_sample_records):
    """Test POST /predict-batch rejects batches exceeding 1,000 items (HTTP 422)."""
    records, _ = valid_sample_records
    sample = records[0]
    oversized_list = [sample] * 1001

    response = client.post(
        "/predict-batch",
        json={"observations": oversized_list},
    )
    assert response.status_code == 422


def test_predict_batch_empty_list(client):
    """Test POST /predict-batch rejects empty observation lists (HTTP 422)."""
    response = client.post("/predict-batch", json={"observations": []})
    assert response.status_code == 422


# ==============================================================================
# 5. Documentation & Schema Tests
# ==============================================================================

def test_openapi_spec_availability(client):
    """Test GET /openapi.json returns valid 200 OpenAPI schema."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    assert spec["openapi"].startswith("3.")
    assert spec["info"]["title"] == "TectonicSense API"
    assert "/health" in spec["paths"]
    assert "/model-info" in spec["paths"]
    assert "/predict" in spec["paths"]
    assert "/predict-batch" in spec["paths"]


def test_swagger_ui_availability(client):
    """Test GET /docs returns 200 Swagger HTML page."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower() or "html" in response.text.lower()


def test_redoc_ui_availability(client):
    """Test GET /redoc returns 200 ReDoc HTML page."""
    response = client.get("/redoc")
    assert response.status_code == 200
    assert "redoc" in response.text.lower() or "html" in response.text.lower()


# ==============================================================================
# 6. Prediction Determinism Tests
# ==============================================================================

def test_prediction_determinism(client, valid_sample_records):
    """Verify repeated predictions on identical input produce identical outputs."""
    records, _ = valid_sample_records
    sample = records[0]

    resp1 = client.post("/predict", json=sample).json()
    resp2 = client.post("/predict", json=sample).json()

    assert resp1["predicted_class"] == resp2["predicted_class"]
    assert resp1["class_id"] == resp2["class_id"]
    assert np.isclose(resp1["confidence"], resp2["confidence"], atol=1e-7)

    probs1 = resp1["class_probabilities"]
    probs2 = resp2["class_probabilities"]
    for c in probs1:
        assert np.isclose(probs1[c], probs2[c], atol=1e-7)
