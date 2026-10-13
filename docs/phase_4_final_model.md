# TectonicSense — Phase 4 Final Model

**Project**: TectonicSense  
**Task**: Multiclass Classification of PB2002 Tectonic Plate Boundary Regimes Using Seismic and Geospatial Features  
**Document**: `docs/phase_4_final_model.md`  
**Phase**: Phase 4 — Final Model & Productionization  
**Status**: COMPLETE  

---

## 1. Phase Overview

Phase 4 of the TectonicSense project converts the experimentally selected machine learning model from Phase 2 and Phase 3 into a frozen, reproducible, and verified production artifact. 

During previous research phases:
- **Phase 1** constructed a curated dataset of 239,757 global earthquake observations linked to PB2002 plate boundary segments.
- **Phase 2** benchmarked nine model architectures and optimized top candidates using Optuna, identifying Tuned XGBoost as the primary model.
- **Phase 3** conducted rigorous generalization testing, feature ablation, SHAP explainability, and error diagnosis on held-out test data.

**Phase 4 operationalizes these findings by delivering**:
1. **Final Model Configuration**: A centralized, frozen YAML specification ([`configs/final_model.yaml`](file:///d:/TectonicSense/configs/final_model.yaml)).
2. **Final Model Training**: Deterministic training of the production model on the 70% stratified training split with frozen hyperparameters ([`scripts/train_final_model.py`](file:///d:/TectonicSense/scripts/train_final_model.py)).
3. **Artifact Serialization**: Packaging the trained model, feature schema, label mappings, and operational metadata under [`models/final/`](file:///d:/TectonicSense/models/final/).
4. **Production Inference Pipeline**: A robust, standalone prediction engine ([`scripts/predict.py`](file:///d:/TectonicSense/scripts/predict.py)) supporting single and batch inference with schema validation and probability outputs.
5. **Independent Verification**: A dedicated validation script ([`scripts/verify_final_model.py`](file:///d:/TectonicSense/scripts/verify_final_model.py)) auditing 16 structural, behavioral, deterministic, and security checks.
6. **Final Evaluation Artifacts**: Machine-readable metrics ([`results/phase_4_final_model/final_model_evaluation.json`](file:///d:/TectonicSense/results/phase_4_final_model/final_model_evaluation.json)) and a consolidated narrative summary ([`results/phase_4_final_model/FINAL_MODEL_SUMMARY.md`](file:///d:/TectonicSense/results/phase_4_final_model/FINAL_MODEL_SUMMARY.md)).
7. **Comprehensive Reproducibility**: Exact environment setup and execution steps to reproduce inference and verification without retraining.

### Production Estimator Profile
- **Model**: Tuned XGBoost (`xgboost.sklearn.XGBClassifier`)
- **Framework Version**: XGBoost `3.0.5`
- **Input Features**: Exactly **26 features** (canonical ordering enforced)
- **Target Classes**: Exactly **7 PB2002 classes** (`CCB`, `CRB`, `CTF`, `OCB`, `OSR`, `OTF`, `SUB`)
- **Random Seed**: `42`

> [!IMPORTANT]
> **Scientific Framing**: TectonicSense performs multiclass classification of tectonic plate boundary regimes. It is **not** an earthquake prediction system and does not predict earthquake occurrence, timing, or magnitude.

---

## 2. Final Model Configuration

The final production model parameters and schema are defined in [`configs/final_model.yaml`](file:///d:/TectonicSense/configs/final_model.yaml). This configuration is frozen and must not be altered:

```yaml
project:
  name: "TectonicSense"
  version: "1.0.0"

task:
  type: "multiclass_classification"
  target_column: "boundary_class"
  number_of_classes: 7

reproducibility:
  random_seed: 42

model:
  name: "Tuned XGBoost"
  framework: "xgboost"
  estimator: "XGBClassifier"
  hyperparameters:
    n_estimators: 120
    max_depth: 9
    learning_rate: 0.15595246363789106
    subsample: 0.7
    colsample_bytree: 0.8
    min_child_weight: 2
    reg_lambda: 0.0359375612249994
    objective: "multi:softprob"
    eval_metric: "mlogloss"
    tree_method: "hist"
    random_state: 42
    n_jobs: -1

features:
  count: 26

classes:
  count: 7
  mapping:
    0: "CCB"
    1: "CRB"
    2: "CTF"
    3: "OCB"
    4: "OSR"
    5: "OTF"
    6: "SUB"
```

---

## 3. Final Feature Schema

Exactly **26 features** are ingested by the model in the strict order recorded in [`models/final/feature_names.json`](file:///d:/TectonicSense/models/final/feature_names.json). The features fall into five distinct groups:

| Index | Feature Name | Group | Description |
|:---:|:---|:---:|:---|
| 1 | `magnitude` | A. Seismic | Earthquake magnitude ($M_w$ / $M_b$) |
| 2 | `magnitude_squared` | E. Engineered | Non-linear energy scaling ($M^2$) |
| 3 | `depth_km` | A. Seismic | Hypocentral depth in kilometers |
| 4 | `depth_abs_km` | E. Engineered | Absolute depth distance ($\|depth\|$) |
| 5 | `longitude` | B. Geographic | Epicenter longitude ($[-180^\circ, +180^\circ]$) |
| 6 | `latitude` | B. Geographic | Epicenter latitude ($[-90^\circ, +90^\circ]$) |
| 7 | `longitude_sin` | B. Geographic | Cyclical sine encoding of longitude |
| 8 | `longitude_cos` | B. Geographic | Cyclical cosine encoding of longitude |
| 9 | `latitude_sin` | B. Geographic | Cyclical sine encoding of latitude |
| 10 | `latitude_cos` | B. Geographic | Cyclical cosine encoding of latitude |
| 11 | `boundary_distance_km` | C. Boundary Geometry | Geodesic distance to nearest PB2002 boundary segment |
| 12 | `log_boundary_distance_km` | E. Engineered | Natural logarithm $\ln(1 + \text{distance})$ |
| 13 | `boundary_length_km` | C. Boundary Geometry | Total length of the nearest PB2002 fault/trench segment |
| 14 | `boundary_azimuth_sin` | C. Boundary Geometry | Sine of the local boundary strike azimuth |
| 15 | `boundary_azimuth_cos` | C. Boundary Geometry | Cosine of the local boundary strike azimuth |
| 16 | `relative_velocity` | D. Kinematic/Geophysical | Relative plate motion rate ($\text{mm/yr}$) |
| 17 | `velocity_azimuth_sin` | D. Kinematic/Geophysical | Sine of the relative velocity azimuth vector |
| 18 | `velocity_azimuth_cos` | D. Kinematic/Geophysical | Cosine of the relative velocity azimuth vector |
| 19 | `divergent_component` | D. Kinematic/Geophysical | Normal velocity component (positive = spreading, negative = convergence) |
| 20 | `right_lateral_component` | D. Kinematic/Geophysical | Strike-slip component (positive = right-lateral, negative = left-lateral) |
| 21 | `absolute_divergent_component` | D. Kinematic/Geophysical | Absolute normal component magnitude |
| 22 | `absolute_right_lateral_component` | D. Kinematic/Geophysical | Absolute strike-slip component magnitude |
| 23 | `tectonic_motion_magnitude` | D. Kinematic/Geophysical | Euclidean magnitude of boundary kinematic velocity |
| 24 | `boundary_elevation_m` | D. Kinematic/Geophysical | Surface elevation / ocean trench bathymetry in meters |
| 25 | `seafloor_age_ma` | D. Kinematic/Geophysical | Crustal age in millions of years (Ma) |
| 26 | `orogen_flag` | D. Kinematic/Geophysical | Binary indicator ($1 = \text{within PB2002 orogen}$, $0 = \text{oceanic/cratonic}$) |

> [!NOTE]
> **Schema Guard**: Discrete segment identifiers (`boundary_id`, `boundary_step_id`) are **explicitly excluded** from the feature set to prevent artificial label leakage. The inference pipeline enforces this exact 26-column schema and reorders DataFrame columns automatically.

---

## 4. Final Model Training

The production model was trained using [`scripts/train_final_model.py`](file:///d:/TectonicSense/scripts/train_final_model.py) following a strict protocol:

- **Source Dataset**: `data/processed/tectonicsense_ml_dataset.csv` (239,757 observations).
- **Training Partition**: `data/processed/splits/train.csv` (167,829 samples, 70.0%).
- **Validation Partition**: `data/processed/splits/validation.csv` (35,964 samples, 15.0%).
- **Untouched Test Partition**: `data/processed/splits/test.csv` (35,964 samples, 15.0%).
- **Splitting Strategy**: Stratified random splitting preserving PB2002 class proportions across all three partitions.
- **Random Seed**: `42`.
- **Training Duration**: **22.89 seconds** on the 167,829 training rows.
- **Serialization Format**: Python `joblib` binary format (`.joblib`).
- **Artifact Size**: **842,753 bytes** (~823 KB).

> [!IMPORTANT]
> The final production model was trained solely on the established 70% training split (`train.csv`). The validation and test sets remained strictly isolated for independent evaluation and was never exposed during model fitting.

---

## 5. Model Artifacts

All production model files are organized in the [`models/final/`](file:///d:/TectonicSense/models/final/) directory:

| File Path | File Size | Description |
|:---|:---:|:---|
| [`models/final/tectonicsense_xgboost.joblib`](file:///d:/TectonicSense/models/final/tectonicsense_xgboost.joblib) | 842,753 B | Serialized `XGBClassifier` estimator binary containing trained decision trees |
| [`models/final/feature_names.json`](file:///d:/TectonicSense/models/final/feature_names.json) | 684 B | Canonical list of 26 features with strict index ordering |
| [`models/final/label_mapping.json`](file:///d:/TectonicSense/models/final/label_mapping.json) | 125 B | Class index mapping dictionary (`0: CCB` to `6: SUB`) |
| [`models/final/metadata.json`](file:///d:/TectonicSense/models/final/metadata.json) | 3,250 B | Complete provenance record (versions, hyperparameters, training time, benchmark metrics) |

---

## 6. Production Inference Pipeline

The inference pipeline in [`scripts/predict.py`](file:///d:/TectonicSense/scripts/predict.py) provides a unified, production-grade interface for scoring new observations:

### Core Capabilities
1. **Dynamic Path Resolution**: Uses Python `pathlib` relative to the repository root, ensuring portability across local, containerized, and cloud environments without hardcoded system paths.
2. **Schema & Input Validation (`validate_and_format_input`)**:
   - Rejects payloads with missing features.
   - Automatically aligns and reorders DataFrame columns to match `feature_names.json`.
   - Validates that all features are strictly numeric.
   - Detects and rejects `NaN` and `None` values with descriptive error messages.
   - Detects and rejects infinite values (`inf` / `-inf`).
   - Asserts exact feature count equals 26.
3. **Dual Prediction Interfaces**:
   - `predict_single(features)`: Accepts a dictionary, Pandas Series, or 1-row DataFrame; returns predicted class name, class ID, confidence, and full 7-class probability map.
   - `predict_batch(dataframe)`: High-throughput batch inference preserving input row order.
4. **Calibrated Probability Outputs**: Uses `predict_proba()` to compute softmax probabilities summing to 1.0. Confidence is defined as the exact probability of the predicted class.
5. **Singleton Access (`get_predictor()`)**: Reuses an in-memory `TectonicPredictor` instance across web requests in API frameworks (FastAPI) and interactive dashboards (Streamlit).

---

## 7. Independent Verification

The finalized model and inference engine were subjected to an independent audit via [`scripts/verify_final_model.py`](file:///d:/TectonicSense/scripts/verify_final_model.py). 

The audit evaluated 16 separate assertions across five categories:
1. **Artifact Existence & Integrity (4 checks)**: Model binary, feature names, label mapping, and metadata loaded cleanly.
2. **Model Graph Structure (3 checks)**: Verified estimator is `XGBClassifier`, expects 26 features (`n_features_in_ = 26`), and outputs 7 classes (`classes_ = [0..6]`).
3. **Inference Functionality (3 checks)**: Single inference, batch inference (10 rows), and probability normalization verified on held-out test data.
4. **Strict Determinism (1 check)**: Repeated evaluations on identical inputs produced bit-identical predictions and class probabilities ($\Delta = 0.00\text{e}+00$).
5. **Security & Input Validation (5 checks)**: Confirmed that missing features, non-numeric values, `NaN` values, infinite values, and schema mismatches are cleanly caught and rejected with `ValueError`.

### Verification Outcome
```text
============================================================
TECTONICSENSE FINAL MODEL VERIFICATION
============================================================
[PASS] Model artifact loaded
[PASS] Feature metadata loaded
[PASS] Label mapping loaded
[PASS] Metadata loaded
[PASS] Model type verified
[PASS] 26 features verified
[PASS] 7 classes verified
[PASS] Single inference verified
[PASS] Batch inference verified
[PASS] Probability normalization verified
[PASS] Deterministic prediction verified
[PASS] Missing feature rejection verified
[PASS] Non-numeric input rejection verified
[PASS] NaN rejection verified
[PASS] Infinite-value rejection verified
[PASS] Incorrect feature-count rejection verified

============================================================
FINAL STATUS: VERIFIED
============================================================
```

All **16 / 16 checks passed**. The complete verification report is recorded at [`results/phase_4_final_model/final_model_verification.json`](file:///d:/TectonicSense/results/phase_4_final_model/final_model_verification.json).

---

## 8. Final Model Evaluation

The finalized model performance is documented under two distinct evaluation paradigms:

### 8.1 Random Stratified Test Evaluation
Evaluated on 35,964 untouched test observations ([`data/processed/splits/test.csv`](file:///d:/TectonicSense/data/processed/splits/test.csv)):

| Metric | Value |
|:---|:---:|
| **Test Observations** | 35,964 |
| **Accuracy** | **0.999889** (99.99%) |
| **Macro Precision** | **0.999821** |
| **Macro Recall** | **0.999761** |
| **Macro F1 Score** | **0.999791** (~0.9998) |
| **Weighted F1 Score** | **0.999889** |
| **Total Errors** | 4 / 35,964 (0.0111% error rate) |
| **Inference Latency** | 13.80 microseconds per sample |

### 8.2 Boundary-Aware Disjoint Evaluation
To evaluate out-of-boundary transfer, a group-disjoint experiment was conducted where continuous fault segments (`boundary_id`) were completely partitioned:
- **Train Boundaries**: 127 boundaries (145,287 observations)
- **Validation Boundaries**: 27 boundaries (53,380 observations)
- **Test Boundaries**: 28 boundaries (41,090 observations)
- **Boundary Overlap**: **0 (Zero overlap between partitions)**

| Metric | Random Stratified Test | Boundary-Aware Test |
|:---|:---:|:---:|
| **Accuracy** | 99.99% | **79.94%** |
| **Macro Precision** | 99.98% | **79.89%** |
| **Macro Recall** | 99.98% | **80.78%** |
| **Macro F1 Score** | **99.98%** | **79.35%** |
| **Weighted F1 Score** | 99.99% | **80.05%** |

### Evaluation Insight
Boundary-aware evaluation represents a substantially harder generalization challenge. In random stratified splits, seismic events occurring along the same continuous geological fault are present in both training and test sets, allowing the model to leverage spatial interpolation. When entire boundaries are held out, the model achieves **79.35% Macro F1**, demonstrating **meaningful out-of-boundary generalization under the evaluated PB2002 split**.

---

## 9. Feature Ablation

Progressive feature ablation was conducted on the held-out test split to quantify the contribution of each feature family:

| Feature Configuration | Features | Accuracy | Macro F1 | Weighted F1 | Training Time |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Group 1: Seismic Only** | 4 | 50.89% | **20.28%** | 40.62% | 19.07s |
| **Group 2: Seismic + Geographic** | 12 | 95.34% | **92.73%** | 95.31% | 26.78s |
| **Group 3: Limited Tectonic** | 21 | 99.98% | **99.96%** | 99.98% | 22.15s |
| **Group 4: Full Feature Set** | 26 | 99.99% | **99.98%** | 99.99% | 20.26s |

### Scientific Finding
**Seismic features alone provided substantially weaker classification performance under the evaluated feature configuration (Macro F1 = 20.28%).** Hypocentral depth and magnitude alone cannot discriminate complex boundary regimes. Incorporating geographic coordinates boosts Macro F1 to 92.73%, while PB2002 kinematic and geophysical variables achieve 99.98%.

---

## 10. Model Benchmarking

In Phase 2, nine model architectures were evaluated on the 35,964-observation validation set:

| Architecture | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Training Time | Selection |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Tuned XGBoost** | **0.9999** | **0.9998** | **0.9998** | **0.9998** | **0.9999** | 19.00s | **SELECTED FINAL** |
| **Tuned LightGBM** | 0.9998 | 0.9997 | 0.9997 | 0.9997 | 0.9998 | 19.35s | Benchmark |
| **HistGradientBoosting** | 0.9998 | 0.9997 | 0.9997 | 0.9997 | 0.9998 | 30.19s | Benchmark |
| **LightGBM (Default)** | 0.9998 | 0.9996 | 0.9996 | 0.9996 | 0.9998 | 19.54s | Benchmark |
| **XGBoost (Default)** | 0.9998 | 0.9997 | 0.9996 | 0.9996 | 0.9998 | 21.25s | Benchmark |
| **Random Forest** | 0.9998 | 0.9997 | 0.9996 | 0.9996 | 0.9998 | 22.70s | Benchmark |
| **CatBoost** | 0.9983 | 0.9974 | 0.9969 | 0.9972 | 0.9983 | 43.23s | Benchmark |
| **Logistic Regression** | 0.9410 | 0.9060 | 0.9020 | 0.9036 | 0.9407 | 21.92s | Linear Baseline |
| **Dummy Classifier** | 0.4811 | 0.0687 | 0.1429 | 0.0928 | 0.3126 | 0.10s | Baseline |

Tuned XGBoost was established as the best-performing model among the evaluated candidates under the experimental setup, exhibiting the highest Macro F1, fast training, compact artifact size, and calibrated probability outputs. Hyperparameter optimization was conducted using Optuna and subsequently frozen in Phase 4.

---

## 11. SHAP / Interpretability

SHAP (`shap.TreeExplainer`) analysis on 2,000 stratified test observations revealed the primary physical drivers of model decisions:

1. **`divergent_component`** (Mean \|SHAP\| = 2.0318): Dominant feature separating divergent boundaries (`OSR`, `CRB`) from convergent boundaries (`SUB`, `CCB`).
2. **`boundary_elevation_m`** (Mean \|SHAP\| = 1.4451): Separates deep ocean trenches (subduction) from ocean ridges and continental landmasses.
3. **`absolute_divergent_component`** (Mean \|SHAP\| = 0.8775): Magnitude of boundary-normal displacement.
4. **`seafloor_age_ma`** (Mean \|SHAP\| = 0.4082): Distinguishes young oceanic spreading centers from older continental margins.
5. **`absolute_right_lateral_component`** (Mean \|SHAP\| = 0.3891): Isolates transform strike-slip boundaries (`OTF`, `CTF`).

### Central Methodological Implication
The target class is derived from PB2002 boundary information and several predictors are also derived from PB2002 metadata. Therefore, the high random-split performance should not be interpreted as independent discovery of tectonic boundary classes from seismic observations alone. The model learns the joint physical and metadata mapping established in PB2002.

---

## 12. Scientific Interpretation

TectonicSense classifies the **tectonic boundary environment** associated with an earthquake observation into one of seven PB2002-derived boundary classes using seismic, geographic, geometric, and tectonic metadata.

### Explicit Scope Boundaries
- **Classification System**: The model categorizes seismic events into tectonic regimes.
- **NOT Earthquake Prediction**: The model does **NOT** forecast earthquake occurrence, timing, or location.
- **NOT Magnitude Forecasting**: The model does **NOT** predict the magnitude of future earthquakes.
- **NOT Causal Geology**: Feature attributions reflect empirical statistical correlations within the PB2002 model, not causal physical mechanisms.

---

## 13. Limitations

A scientifically sound evaluation must explicitly state project limitations:

1. **PB2002-Derived Ground Truth**: Target boundary classes reflect the PB2002 geological model rather than direct in-situ physical measurements.
2. **PB2002-Derived Predictor Variables**: Features such as relative velocity and boundary azimuth are derived from PB2002 metadata.
3. **Potential Metadata Circularity**: The model partly learns the relationship between PB2002 metadata and PB2002 target labels.
4. **Spatial Autocorrelation in Random Splits**: Random stratified splitting permits multiple earthquakes along the same continuous fault to appear in both train and test splits.
5. **Necessity of Boundary-Aware Evaluation**: Testing generalization across isolated boundaries is critical for realistic assessment.
6. **Performance Drop Across Unseen Boundaries**: Macro F1 decreases from 99.98% to 79.35% under boundary-disjoint evaluation.
7. **No Causal Physical Claims**: Tree splits represent statistical patterns, not physical proof.
8. **No Earthquake Forecasting Capability**: The model cannot predict future seismic activity.
9. **No Universal Geographic Generalization**: The model is trained on mapped PB2002 plate boundaries and may not generalize to diffuse intraplate seismic zones.
10. **Model Representation Specificity**: PB2002 is one specific realization of global plate kinematics; other tectonic models (e.g., MORVEL, Bird 2003 updates) may produce slight variations.

---

## 14. Reproducibility

The finalized model and verification suite can be fully reproduced using the following steps without retraining:

### Step 1: Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install exact pinned dependencies
pip install -r requirements.txt
```

### Step 2: Run Production Inference Self-Test
```bash
python scripts/predict.py
```
*Executes single-observation and 5-row batch inference using [`data/processed/splits/test.csv`](file:///d:/TectonicSense/data/processed/splits/test.csv).*

### Step 3: Run Independent Model Verification
```bash
python scripts/verify_final_model.py
```
*Validates artifacts, model structure, inference outputs, determinism, and invalid input rejection (16/16 checks).*

### Step 4: Validate Machine-Readable JSON Reports
```bash
python -m json.tool results/phase_4_final_model/final_model_verification.json
python -m json.tool results/phase_4_final_model/final_model_evaluation.json
```

---

## 15. Project Reproducibility Map

| Component | File Path | Purpose |
|:---|:---|:---|
| **Final Configuration** | [`configs/final_model.yaml`](file:///d:/TectonicSense/configs/final_model.yaml) | Frozen production model specification |
| **Final Model Binary** | [`models/final/tectonicsense_xgboost.joblib`](file:///d:/TectonicSense/models/final/tectonicsense_xgboost.joblib) | Production XGBoost estimator binary |
| **Feature Schema** | [`models/final/feature_names.json`](file:///d:/TectonicSense/models/final/feature_names.json) | Canonical 26-feature ordered list |
| **Label Mapping** | [`models/final/label_mapping.json`](file:///d:/TectonicSense/models/final/label_mapping.json) | Integer to PB2002 class mapping |
| **Operational Metadata** | [`models/final/metadata.json`](file:///d:/TectonicSense/models/final/metadata.json) | Hyperparameters, versions, training duration |
| **Inference Engine** | [`scripts/predict.py`](file:///d:/TectonicSense/scripts/predict.py) | Production prediction pipeline |
| **Verification Suite** | [`scripts/verify_final_model.py`](file:///d:/TectonicSense/scripts/verify_final_model.py) | Independent 16-check verification script |
| **Evaluation Metrics** | [`results/phase_4_final_model/final_model_evaluation.json`](file:///d:/TectonicSense/results/phase_4_final_model/final_model_evaluation.json) | Machine-readable final evaluation metrics |
| **Consolidated Summary** | [`results/phase_4_final_model/FINAL_MODEL_SUMMARY.md`](file:///d:/TectonicSense/results/phase_4_final_model/FINAL_MODEL_SUMMARY.md) | Narrative final evaluation summary |
| **Production Documentation** | [`docs/phase_4_final_model.md`](file:///d:/TectonicSense/docs/phase_4_final_model.md) | Definitive Phase 4 system documentation |

---

## 16. Phase 5 Readiness

With Phase 4 complete, TectonicSense has transitioned from an experimental research codebase to a production-verified machine learning system. The serialized model binary, strict input validator, and reusable inference pipeline provide the foundation for Phase 5 deployment:

- **FastAPI REST Service**: Exposing `/health`, `/predict`, and `/predict-batch` endpoints with Pydantic request models.
- **Interactive UI**: Building a Streamlit or Gradio application with interactive map coordinates and confidence gauges.
- **Docker Containerization**: Packaging the inference pipeline into a lightweight, reproducible container.
- **Deployment**: Local container serving, cloud deployment, or publishing to Hugging Face Spaces.
