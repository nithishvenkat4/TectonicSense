# TectonicSense Final Model Summary

**Project**: TectonicSense  
**Scientific Definition**: Explainable Machine Learning for Tectonic Plate Boundary Classification Using Seismic and Geospatial Features  
**Production Artifact**: `models/final/tectonicsense_xgboost.joblib`  
**Evaluation Status**: FINAL  

---

## 1. Final Model

The final production machine learning model selected for TectonicSense is **Tuned XGBoost** (`xgboost.sklearn.XGBClassifier`), trained on the established 70% stratified training split (167,829 observations).

- **Architecture**: Gradient Boosted Decision Trees (`XGBClassifier`)
- **Framework Version**: XGBoost `3.0.5`
- **Output Classes (7)**: PB2002 Plate Boundary Classes:
  - `CCB` (Continental Convergent Boundary)
  - `CRB` (Continental Rift Boundary)
  - `CTF` (Continental Transform Fault)
  - `OCB` (Oceanic Convergent Boundary)
  - `OSR` (Oceanic Spreading Ridge)
  - `OTF` (Oceanic Transform Fault)
  - `SUB` (Subduction Zone)
- **Input Feature Count**: Exactly **26 features** in canonical order (excluding identifiers `boundary_id` and `boundary_step_id` to eliminate synthetic leakage).
- **Frozen Hyperparameters**:
  ```yaml
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
  ```
- **Serialized Artifact**: [`models/final/tectonicsense_xgboost.joblib`](file:///d:/TectonicSense/models/final/tectonicsense_xgboost.joblib) (842,753 bytes, ~823 KB)
- **Training Time**: 22.89 seconds (final production run)

---

## 2. Random Stratified Test Performance

The finalized model was evaluated on the untouched, held-out test partition ([`data/processed/splits/test.csv`](file:///d:/TectonicSense/data/processed/splits/test.csv)), comprising 35,964 earthquake observations.

### Global Test Benchmark
| Metric | Value |
|:---|:---:|
| **Test Observations** | 35,964 |
| **Accuracy** | **0.999889** (99.99%) |
| **Macro Precision** | **0.999821** |
| **Macro Recall** | **0.999761** |
| **Macro F1 Score** | **0.999791** (~0.9998) |
| **Weighted Precision** | **0.999889** |
| **Weighted Recall** | **0.999889** |
| **Weighted F1 Score** | **0.999889** |
| **Total Misclassifications** | 4 / 35,964 (0.0111% error rate) |
| **Total Inference Time** | 0.4962 seconds |
| **Per-Sample Latency** | 13.80 microseconds |

### Per-Class Test Performance Breakdown
| Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **CCB** | 0.999511 | 1.000000 | 0.999756 | 4,092 |
| **CRB** | 1.000000 | 1.000000 | 1.000000 | 3,948 |
| **CTF** | 1.000000 | 1.000000 | 1.000000 | 2,614 |
| **OCB** | 1.000000 | 0.998639 | 0.999319 | 1,470 |
| **OSR** | 1.000000 | 0.999746 | 0.999873 | 3,931 |
| **OTF** | 0.999233 | 1.000000 | 0.999617 | 2,607 |
| **SUB** | 1.000000 | 0.999942 | 0.999971 | 17,302 |

---

## 3. Boundary-Aware Generalization

To rigorously assess performance beyond spatial proximity interpolation, a **boundary-aware group disjoint experiment** was executed in Phase 3. The dataset was partitioned by PB2002 `boundary_id`, ensuring **zero boundary overlap** across splits.

### Disjoint Partition Structure
- **Training Boundaries**: 127 boundaries (145,287 observations)
- **Validation Boundaries**: 27 boundaries (53,380 observations)
- **Test Boundaries**: 28 boundaries (41,090 observations)
- **Boundary Overlap (Train $\cap$ Val $\cap$ Test)**: **0 (Strictly Disjoint)**

### Performance Under Boundary Separation
| Metric | Random Stratified Test | Boundary-Aware Test |
|:---|:---:|:---:|
| **Accuracy** | 99.99% | **79.94%** |
| **Macro Precision** | 99.98% | **79.89%** |
| **Macro Recall** | 99.98% | **80.78%** |
| **Macro F1 Score** | **99.98%** | **79.35%** |
| **Weighted F1 Score** | 99.99% | **80.05%** |

### Scientific Significance
In random stratified splitting, individual seismic events along the same continuous geological fault or trench system are distributed between train and test splits. The model can thus exploit localized geospatial coordinates. 

When boundaries are strictly isolated, the model must rely purely on generalized geophysical and kinematic signatures to classify unfamiliar plate segments. Achieving **79.35% Macro F1** demonstrates **meaningful out-of-boundary generalization under the evaluated PB2002 split**, confirming that the model learns genuine tectonic patterns rather than solely memorizing localized boundary coordinates.

---

## 4. Feature Ablation

To determine the empirical contribution of distinct feature families, four nested feature configurations were tested on the held-out test split:

| Configuration | Features Included | Feature Count | Accuracy | Macro F1 | Weighted F1 | Training Time |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Group 1: Seismic Only** | `magnitude`, `magnitude_squared`, `depth_km`, `depth_abs_km` | 4 | 50.89% | **20.28%** | 40.62% | 19.07s |
| **Group 2: Seismic + Geographic** | Group 1 + Coordinates (`lon`, `lat`, sine/cos encodings, distances) | 12 | 95.34% | **92.73%** | 95.31% | 26.78s |
| **Group 3: Limited Tectonic** | Group 2 + Azimuths, kinematics, relative velocity | 21 | 99.98% | **99.96%** | 99.98% | 22.15s |
| **Group 4: Full Feature Set** | Group 3 + Bathymetry, crustal age, orogen flag | 26 | 99.99% | **99.98%** | 99.99% | 20.26s |

### Scientific Finding
**Seismic features alone provided substantially weaker classification performance under the evaluated feature configuration (Macro F1 = 20.28%).** Hypocenter depth and magnitude alone cannot disambiguate transform faults from spreading centers or subduction zones. Spatial coordinates boost performance to 92.73%, while the inclusion of PB2002-derived kinematic and bathymetric variables yields 99.98% Macro F1.

---

## 5. Model Benchmarking

During Phase 2, nine candidate model architectures were rigorously benchmarked on the 35,964-sample validation partition:

| Model Architecture | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Training Time | Selection Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Tuned XGBoost** | **0.9999** | **0.9998** | **0.9998** | **0.9998** | **0.9999** | 19.00s | **SELECTED FINAL** |
| **Tuned LightGBM** | 0.9998 | 0.9997 | 0.9997 | 0.9997 | 0.9998 | 19.35s | Benchmark |
| **HistGradientBoosting** | 0.9998 | 0.9997 | 0.9997 | 0.9997 | 0.9998 | 30.19s | Benchmark |
| **LightGBM (Default)** | 0.9998 | 0.9996 | 0.9996 | 0.9996 | 0.9998 | 19.54s | Benchmark |
| **XGBoost (Default)** | 0.9998 | 0.9997 | 0.9996 | 0.9996 | 0.9998 | 21.25s | Benchmark |
| **Random Forest** | 0.9998 | 0.9997 | 0.9996 | 0.9996 | 0.9998 | 22.70s | Benchmark |
| **CatBoost** | 0.9983 | 0.9974 | 0.9969 | 0.9972 | 0.9983 | 43.23s | Benchmark |
| **Logistic Regression** | 0.9410 | 0.9060 | 0.9020 | 0.9036 | 0.9407 | 21.92s | Linear Baseline |
| **Dummy Classifier** | 0.4811 | 0.0687 | 0.1429 | 0.0928 | 0.3126 | 0.10s | Trivial Baseline |

### Rationale for Selecting Tuned XGBoost
Tuned XGBoost was established as the best-performing model among the evaluated candidates under the experimental setup:
1. **Top Macro F1**: Highest balanced performance across all seven classes (0.9998).
2. **Compact Serialized Footprint**: 842 KB binary (`joblib`), ideal for lightweight deployment.
3. **Execution Latency**: 13.8 microseconds per sample inference, enabling high-throughput microservices.
4. **Stable Calibration**: Precise multiclass probability estimates (`multi:softprob`).
5. **Deployment Ecosystem**: Robust, cross-platform enterprise support.

---

## 6. Explainability

SHAP (`shap.TreeExplainer`) analysis on 2,000 stratified test observations identified the physical drivers of model predictions:

| Rank | Feature | Mean Absolute SHAP | Physical Interpretation |
|:---:|:---|:---:|:---|
| 1 | `divergent_component` | **2.0318** | Distinguishes convergent vs. divergent/transform plate motions |
| 2 | `boundary_elevation_m` | **1.4451** | Separates deep ocean trenches (subduction) from ridges and continents |
| 3 | `absolute_divergent_component` | **0.8775** | Magnitude of normal motion across plate boundary interface |
| 4 | `seafloor_age_ma` | **0.4082** | Younger lithosphere along ridges (OSR); older at continental margins |
| 5 | `absolute_right_lateral_component` | **0.3891** | Distinguishes strike-slip transform boundaries (`OTF`, `CTF`) |
| 6 | `relative_velocity` | **0.2626** | Total tectonic plate convergence/separation rate |
| 7 | `tectonic_motion_magnitude` | **0.2201** | Total kinematic motion vector magnitude |

Pure seismic features (`depth_km`: 0.0138; `magnitude`: 0.0038) exhibit minor direct SHAP importance, indicating that kinematics and bathymetry dominate the regime assignment.

---

## 7. Scientific Interpretation

TectonicSense classifies **tectonic plate boundary regimes** associated with seismic and geospatial observations by integrating:
1. Earthquake hypocentral coordinates and magnitude.
2. Geospatial proximity to known PB2002 fault segments.
3. PB2002 kinematic motion vectors (relative velocity, divergence, strike-slip components).
4. Crustal and geophysical properties (elevation/bathymetry, seafloor age, orogeny).

> [!CAUTION]
> **Definitive Scientific Scope Framing**:
> - TectonicSense is **NOT an earthquake prediction system**.
> - It does **NOT forecast when or where an earthquake will happen**.
> - It is **NOT deterministic earthquake forecasting**.
> - It is an **environmental characterization and regime classification system**.

---

## 8. Limitations

A scientifically rigorous evaluation must transparently disclose the limitations of the model:

1. **PB2002-Derived Ground Truth**: Target labels originate from the geological PB2002 model rather than direct in-situ physical measurements.
2. **Metadata Circularity**: Several highly predictive features (kinematics, boundary distance, azimuth) are derived from the same PB2002 database as the ground truth labels. The model therefore partly reconstructs PB2002 metadata relationships.
3. **Spatial Autocorrelation in Random Splits**: Random stratified splitting permits earthquakes from the same continuous boundary to exist in both training and test sets.
4. **Generalization Drop Under Boundary Separation**: When tested on entirely unseen boundaries, Macro F1 drops from **99.98% to 79.35%**, highlighting the challenge of out-of-boundary transfer.
5. **No Causal Physical Claims**: Tree-based feature attributions reflect statistical associations, not underlying geodynamic causality.
6. **No Universal Geographic Generalization**: The model is conditioned on the global PB2002 fault geometry and cannot be assumed to transfer to unmapped or intraplate seismic zones.

---

## 9. Production Readiness

The TectonicSense model artifact has been fully prepared and verified for operational deployment:

- **Model Artifact**: Serialized at [`models/final/tectonicsense_xgboost.joblib`](file:///d:/TectonicSense/models/final/tectonicsense_xgboost.joblib).
- **Canonical Schema**: Enforced via [`models/final/feature_names.json`](file:///d:/TectonicSense/models/final/feature_names.json) (26 features) and [`models/final/label_mapping.json`](file:///d:/TectonicSense/models/final/label_mapping.json) (7 classes).
- **Reusable Inference Engine**: [`scripts/predict.py`](file:///d:/TectonicSense/scripts/predict.py) provides single and batch prediction methods with automatic column alignment.
- **Independent Verification**: [`scripts/verify_final_model.py`](file:///d:/TectonicSense/scripts/verify_final_model.py) confirmed:
  - 16 / 16 independent test assertions passed.
  - Strict determinism across repeated executions ($\Delta = 0.00\text{e}+00$).
  - Fail-safe rejection of missing, non-numeric, `NaN`, infinite, or malformed inputs.
  - Machine-readable verification log saved at [`results/phase_4_final_model/final_model_verification.json`](file:///d:/TectonicSense/results/phase_4_final_model/final_model_verification.json).

---

## 10. Preparation for Phase 5 Deployment

With the model verified and consolidated, the system is ready for Phase 5 operationalization:

- **FastAPI REST API**: Serving `/predict` and `/predict-batch` endpoints.
- **Interactive UI**: Streamlit or Gradio interface with dynamic map visualization and probability gauges.
- **Containerization**: Production Docker image with isolated dependencies and health checks.
- **Deployment Targets**: Local containerized hosting, cloud microservice, or Hugging Face Spaces.
