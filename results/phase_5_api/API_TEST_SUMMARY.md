# TectonicSense Phase 5A/5B — API Test Summary

**Component**: Production REST API Backend  
**Framework**: FastAPI 0.116.1 / Uvicorn 0.35.0 / Pydantic 2.11.7  
**Test Suite**: `api/tests/test_api.py`  
**Test Status**: **PASSED (17 / 17 Tests Passed)**  
**Verification Record**: `results/phase_5_api/api_test_report.json`  

---

## 1. API Purpose

The TectonicSense FastAPI service exposes the frozen Phase 4 production machine learning model (`models/final/tectonicsense_xgboost.joblib`) as a high-performance, strictly-validated REST microservice. It enables client applications (interactive user interfaces, automated data pipelines, and third-party consumers) to perform synchronous single-event and batch plate boundary regime classifications.

---

## 2. Endpoints & Capabilities

| HTTP Method | Route | Description | Request Payload | Response Model |
|:---|:---|:---|:---|:---|
| `GET` | `/health` | Service health & model readiness check | None | `HealthResponse` |
| `GET` | `/model-info` | Exposes safe architecture metadata & class list | None | `ModelInfoResponse` |
| `POST` | `/predict` | Single observation classification | Exactly 26 features (`SeismicFeatureInput`) | `PredictionResponse` |
| `POST` | `/predict-batch` | Batch classification (up to 1,000 observations) | List of observations (`BatchPredictionRequest`) | `BatchPredictionResponse` |
| `GET` | `/docs` | Interactive Swagger UI documentation | None | HTML |
| `GET` | `/redoc` | ReDoc API reference documentation | None | HTML |
| `GET` | `/openapi.json` | Machine-readable OpenAPI 3.1 specification | None | JSON |

---

## 3. Strict Validation & Error Handling

The API layer enforces strict data integrity rules using Pydantic v2:
- **Exact Schema Enforcement**: All 26 canonical features must be supplied; extra attributes and leakage fields (`boundary_id`, `boundary_step_id`, `boundary_class`) are strictly forbidden (`extra="forbid"`) and yield `HTTP 422`.
- **Finite Numeric Guard**: Custom post-field validators ensure that non-numeric, `NaN`, `null`, `inf`, or `-inf` values are rejected immediately with `HTTP 422`.
- **Batch Size Limit**: Batches are strictly limited to $[1, 1000]$ records. Empty payloads or payloads exceeding 1,000 rows yield `HTTP 422`.
- **No Stack Trace Leakage**: Internal unexpected exceptions return a safe `HTTP 500` error message (`{"detail": "Inference failed due to an internal server error."}`) while logging full diagnostics server-side.

---

## 4. Test Results Summary

The automated pytest suite (`pytest api/tests/test_api.py -v`) executed 17 test cases against the live FastAPI application:

```text
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\TectonicSense
collected 17 items

api/tests/test_api.py::test_health_check PASSED                          [  5%]
api/tests/test_api.py::test_model_info PASSED                            [ 11%]
api/tests/test_api.py::test_predict_single_valid PASSED                  [ 17%]
api/tests/test_api.py::test_predict_single_missing_feature PASSED        [ 23%]
api/tests/test_api.py::test_predict_single_non_numeric PASSED            [ 29%]
api/tests/test_api.py::test_predict_single_nan PASSED                    [ 35%]
api/tests/test_api.py::test_predict_single_infinity PASSED               [ 41%]
api/tests/test_api.py::test_predict_single_extra_forbidden_fields PASSED [ 47%]
api/tests/test_api.py::test_predict_batch_valid PASSED                   [ 52%]
api/tests/test_api.py::test_predict_batch_order_preservation PASSED      [ 58%]
api/tests/test_api.py::test_predict_batch_malformed PASSED               [ 64%]
api/tests/test_api.py::test_predict_batch_exceeds_max_limit PASSED       [ 70%]
api/tests/test_api.py::test_predict_batch_empty_list PASSED              [ 76%]
api/tests/test_api.py::test_openapi_spec_availability PASSED             [ 82%]
api/tests/test_api.py::test_swagger_ui_availability PASSED               [ 88%]
api/tests/test_api.py::test_redoc_ui_availability PASSED                 [ 94%]
api/tests/test_api.py::test_prediction_determinism PASSED                [100%]

============================= 17 passed in 10.10s =============================
```

- **Total Tests**: 17
- **Passed**: 17 (100%)
- **Failed**: 0
- **Duration**: 10.10 seconds

---

## 5. Example Successful Inference

Executing `POST /predict` with a valid test split earthquake observation:

**Request Payload Snippet**:
```json
{
  "magnitude": 4.8,
  "magnitude_squared": 23.04,
  "depth_km": 10.0,
  "depth_abs_km": 10.0,
  "longitude": -175.12,
  "latitude": -21.34,
  "longitude_sin": -0.085,
  "longitude_cos": -0.996,
  "latitude_sin": -0.364,
  "latitude_cos": 0.931,
  "boundary_distance_km": 14.2,
  "log_boundary_distance_km": 2.721,
  "boundary_length_km": 182.4,
  "boundary_azimuth_sin": 0.342,
  "boundary_azimuth_cos": 0.940,
  "relative_velocity": 78.5,
  "velocity_azimuth_sin": -0.985,
  "velocity_azimuth_cos": 0.174,
  "divergent_component": -74.2,
  "right_lateral_component": 25.6,
  "absolute_divergent_component": 74.2,
  "absolute_right_lateral_component": 25.6,
  "tectonic_motion_magnitude": 78.5,
  "boundary_elevation_m": -6200.0,
  "seafloor_age_ma": 115.0,
  "orogen_flag": 0.0
}
```

**Response Payload (`HTTP 200 OK`)**:
```json
{
  "predicted_class": "CTF",
  "class_id": 2,
  "confidence": 0.9999691247940063,
  "class_probabilities": {
    "CCB": 0.000003,
    "CRB": 0.000010,
    "CTF": 0.999969,
    "OCB": 0.000002,
    "OSR": 0.000002,
    "OTF": 0.000009,
    "SUB": 0.000004
  }
}
```

---

## 6. OpenAPI Documentation

FastAPI automatically generates comprehensive API documentation conforming to the OpenAPI 3.1 standard:
- **Swagger UI**: Accessible at `http://127.0.0.1:8000/docs` with interactive request testing and live schema exploration.
- **ReDoc**: Accessible at `http://127.0.0.1:8000/redoc` for clean, reference-style documentation.
- **OpenAPI Schema**: Accessible at `http://127.0.0.1:8000/openapi.json`.

---

## 7. Relationship to Phase 4 Inference Engine

The API backend adheres strictly to the architectural design principles of TectonicSense:
- **Zero Model Duplication**: The API does not instantiate or train ML models directly; it imports `get_predictor()` from [`scripts/predict.py`](file:///d:/TectonicSense/scripts/predict.py).
- **Single Process In-Memory Model**: The model binary is loaded once during application startup via FastAPI lifespan events and reused across all incoming requests.
- **Intact Phase 4 Core**: Independent regression checks confirmed that `scripts/predict.py` and `scripts/verify_final_model.py` (16/16 checks) remain completely green and unaffected.
