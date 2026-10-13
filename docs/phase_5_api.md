# TectonicSense Phase 5 — FastAPI Production API

**Project**: TectonicSense  
**Document**: `docs/phase_5_api.md`  
**Phase**: Phase 5A + 5B — FastAPI Production API + Testing & Documentation  
**Status**: COMPLETE  

---

## 1. Overview

Phase 5A and 5B introduce a production-ready, asynchronous REST API for TectonicSense, built with **FastAPI**, **Pydantic v2**, and **Uvicorn**.

The primary purpose of the API is to expose the frozen, verified Phase 4 machine learning model ([`models/final/tectonicsense_xgboost.joblib`](file:///d:/TectonicSense/models/final/tectonicsense_xgboost.joblib)) to downstream clients, including:
- Interactive web dashboards (Streamlit / Gradio)
- Automated data processing pipelines
- Microservice clusters and containerized environments

The service provides endpoints for health checks, model metadata inspection, single-event classification, and high-throughput batch classification with strict schema enforcement.

---

## 2. Architecture

The API layer is strictly decoupled from the underlying machine learning logic. The service acts purely as an interface, request validator, and serialization layer:

```text
       Client Application (Browser, UI, Script, cURL)
                             │
                             ▼
                    FastAPI REST Routing
                             │
                             ▼
                Pydantic v2 Input Validation
           (26 numeric features, finite check, no NaNs)
                             │
                             ▼
              TectonicPredictor Singleton Engine
                    (scripts/predict.py)
                             │
                             ▼
                 Serialized XGBoost Model
         (models/final/tectonicsense_xgboost.joblib)
                             │
                             ▼
                 Prediction Response Payload
          (predicted_class, class_id, confidence, probs)
```

> [!NOTE]
> **Separation of Concerns**: FastAPI does **not** contain ML training or preprocessing code. It imports `get_predictor()` from [`scripts/predict.py`](file:///d:/TectonicSense/scripts/predict.py), which manages model binary deserialization, schema reordering, and inference.

---

## 3. API Endpoints

The API exposes the following core endpoints:

| Endpoint | Method | Summary | Description | Status Code |
|:---|:---:|:---|:---|:---:|
| `/health` | `GET` | Service Health Check | Confirms server availability and model initialization | `200` / `503` |
| `/model-info` | `GET` | Model Metadata | Returns safe architecture details, feature count, and class names | `200` |
| `/predict` | `POST` | Single Prediction | Classifies one seismic observation using the 26 features | `200` / `422` |
| `/predict-batch` | `POST` | Batch Prediction | Classifies a list of observations (up to 1,000 items) | `200` / `422` |
| `/docs` | `GET` | Swagger UI | Interactive API documentation and test interface | `200` |
| `/redoc` | `GET` | ReDoc UI | Structured reference API documentation | `200` |
| `/openapi.json` | `GET` | OpenAPI 3.1 Spec | Machine-readable API schema | `200` |

---

## 4. Request Schema

Predictions require exactly **26 numeric features**, defined in [`api/schemas.py`](file:///d:/TectonicSense/api/schemas.py) under the `SeismicFeatureInput` model:

```json
{
  "magnitude": 5.4,
  "magnitude_squared": 29.16,
  "depth_km": 15.0,
  "depth_abs_km": 15.0,
  "longitude": -175.25,
  "latitude": -21.45,
  "longitude_sin": -0.0828,
  "longitude_cos": -0.9966,
  "latitude_sin": -0.3657,
  "latitude_cos": 0.9307,
  "boundary_distance_km": 12.5,
  "log_boundary_distance_km": 2.6027,
  "boundary_length_km": 182.4,
  "boundary_azimuth_sin": 0.3420,
  "boundary_azimuth_cos": 0.9397,
  "relative_velocity": 78.5,
  "velocity_azimuth_sin": -0.9848,
  "velocity_azimuth_cos": 0.1736,
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

> [!CAUTION]
> **Forbidden Input Fields**: Target labels (`boundary_class`) and internal PB2002 identifiers (`boundary_id`, `boundary_step_id`) are forbidden (`extra="forbid"`). Attempting to supply them will result in an immediate `HTTP 422 Unprocessable Entity` response.

---

## 5. Response Schema

### Single Prediction Response (`PredictionResponse`)
```json
{
  "predicted_class": "SUB",
  "class_id": 6,
  "confidence": 0.999942,
  "class_probabilities": {
    "CCB": 0.000001,
    "CRB": 0.000001,
    "CTF": 0.000002,
    "OCB": 0.000005,
    "OSR": 0.000003,
    "OTF": 0.000046,
    "SUB": 0.999942
  }
}
```

- `predicted_class`: One of the seven PB2002 boundary classes (`CCB`, `CRB`, `CTF`, `OCB`, `OSR`, `OTF`, `SUB`).
- `class_id`: Integer index corresponding to the predicted class (0 to 6).
- `confidence`: Estimated softmax probability of the winning class.
- `class_probabilities`: Normalized probability distribution across all 7 classes (summing to 1.0).

---

## 6. Validation

The API employs rigorous validation guards:
1. **Schema Completeness**: All 26 features are mandatory; missing any feature produces `HTTP 422`.
2. **Strict Type Conversion**: Non-numeric fields (e.g. strings, booleans, objects) fail validation with `HTTP 422`.
3. **Finite Value Check**: A custom validator audits every incoming float. Payloads containing `NaN`, `null`, `inf`, or `-inf` are immediately rejected with descriptive validation messages.
4. **No Unexpected Columns**: Pydantic's `extra = "forbid"` setting prevents users from injecting arbitrary keys or model-leakage fields.

---

## 7. Error Handling

Errors are caught cleanly without leaking stack traces or internal paths:

- **Validation Errors (`HTTP 422`)**:
  Returned when input fields fail type checks, boundary constraints, or finiteness validation.
  ```json
  {
    "detail": [
      {
        "type": "value_error",
        "loc": ["body", "depth_km"],
        "msg": "Value error, Numeric value must be finite; NaN, inf, and -inf are strictly forbidden.",
        "input": "nan"
      }
    ]
  }
  ```
- **Service Degraded (`HTTP 503`)**:
  Returned by `/health` if the underlying ML model fails to load into memory.
- **Internal Server Errors (`HTTP 500`)**:
  Unexpected runtime exceptions return a clean generic message while writing the full exception to the server log:
  ```json
  {
    "detail": "Inference failed due to an internal server error."
  }
  ```

---

## 8. Batch Inference

The `/predict-batch` endpoint allows classifying multiple seismic events in a single HTTP request:

```json
{
  "observations": [
    { ...26 features for earthquake 1... },
    { ...26 features for earthquake 2... }
  ]
}
```

- **Order Preservation**: Output predictions match the exact index order of the input observations array.
- **Batch Size Limit**: Strict limit of **1 to 1,000 observations per request**. Empty arrays or arrays with $>1,000$ elements are rejected with `HTTP 422`.
- **Vectorized Inference**: Dispatches the batch to `TectonicPredictor.predict_batch()`, leveraging XGBoost's multithreaded matrix evaluation.

---

## 9. OpenAPI Documentation

FastAPI auto-generates live interactive documentation conforming to the OpenAPI 3.1 specification:
- **Swagger UI (`/docs`)**: Interactive sandbox for testing endpoints directly in the browser.
- **ReDoc (`/redoc`)**: Responsive, human-readable documentation with organized model schemas.
- **OpenAPI Schema (`/openapi.json`)**: Raw JSON specification for API client generation (e.g. `openapi-generator`).

---

## 10. Running Locally

The API is executed using Uvicorn from the project root directory:

### Development Mode (with hot-reloading)
```bash
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

### Production Mode
```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 4
```

Access the service at `http://127.0.0.1:8000`.

---

## 11. Testing

The API includes an automated test suite ([`api/tests/test_api.py`](file:///d:/TectonicSense/api/tests/test_api.py)) using `pytest` and FastAPI's `TestClient`:

```bash
pytest api/tests/test_api.py -v
```

### Test Coverage (17 Tests)
- `test_health_check`: Validates `/health` returns 200 OK.
- `test_model_info`: Validates `/model-info` returns 200 with 26 features and 7 classes.
- `test_predict_single_valid`: Tests single inference with ground truth test data.
- `test_predict_single_missing_feature`: Confirms missing fields return 422.
- `test_predict_single_non_numeric`: Confirms non-numeric fields return 422.
- `test_predict_single_nan`: Confirms NaN floats return 422.
- `test_predict_single_infinity`: Confirms infinite floats return 422.
- `test_predict_single_extra_forbidden_fields`: Confirms `boundary_id` / `boundary_class` return 422.
- `test_predict_batch_valid`: Tests batch inference on 5 observations.
- `test_predict_batch_order_preservation`: Asserts batch order matches individual order.
- `test_predict_batch_malformed`: Confirms batches with invalid rows fail with 422.
- `test_predict_batch_exceeds_max_limit`: Confirms batches $>1000$ fail with 422.
- `test_predict_batch_empty_list`: Confirms empty batches fail with 422.
- `test_openapi_spec_availability`: Confirms `/openapi.json` is accessible.
- `test_swagger_ui_availability`: Confirms `/docs` is accessible.
- `test_redoc_ui_availability`: Confirms `/redoc` is accessible.
- `test_prediction_determinism`: Asserts identical repeated inputs return bit-identical outputs.

---

## 12. Security Considerations

The API implements several defensive measures:
1. **No Sensitive Path Disclosure**: The `/model-info` and error endpoints never reveal server directories, filenames, or internal environment paths.
2. **Resource Exhaustion Guard**: Strict batch limits ($N \le 1,000$) prevent memory exhaustion attacks.
3. **No Target Leakage**: Input validation blocks synthetic labels or segment IDs from skewing predictions.
4. **Deterministic Singleton**: Prevents arbitrary model reloads or re-instantiations over HTTP.
5. **Read-Only Architecture**: The API has no write, update, or retraining capabilities.

---

## 13. Phase 4 Integration

The API maintains complete backward compatibility and structural integrity with the Phase 4 ML core:
- **Zero Model Modification**: The production binary [`models/final/tectonicsense_xgboost.joblib`](file:///d:/TectonicSense/models/final/tectonicsense_xgboost.joblib) is untouched.
- **Regression Assurance**: Running `scripts/predict.py` and `scripts/verify_final_model.py` confirms that all 16 independent model verification checks continue to pass with zero failures.

---

## 14. Phase 5 Next Steps

With the FastAPI backend verified, Phase 5 can proceed to:
- **Phase 5C**: Streamlit or Gradio interactive web interface for geospatial coordinate selection and probability visualization.
- **Phase 5D**: Docker containerization with multi-stage builds and isolated production dependencies.
- **Phase 5E**: Deployment to local containers, cloud services, or Hugging Face Spaces.
