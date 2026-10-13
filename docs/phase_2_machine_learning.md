# Phase 2 — Machine Learning: Systematic Model Benchmarking & Optimization

## 1. Phase Objective
The primary objective of **Phase 2 — Machine Learning** is to design, implement, and benchmark a reproducible, rigorous multiclass classification pipeline for tectonic plate boundary classification (`boundary_class`). TectonicSense classifies tectonic plate-boundary regimes associated with seismic and geospatial observations. 

> [!IMPORTANT]
> **Scientific Scope Framing**: This project does NOT perform exact earthquake prediction or earthquake forecasting. It solves an academic multiclass geophysical classification task: mapping observed earthquake events and spatial-kinematic features to their corresponding Peter Bird (PB2002) plate boundary tectonic regimes.

The modeling pipeline systematically progresses through five key stages:
1. **Baseline Models**: Dummy Classifier, Scaled Multinomial Logistic Regression, and Random Forest.
2. **Advanced Ensemble & Gradient Boosting Models**: XGBoost, LightGBM, CatBoost, and HistGradientBoostingClassifier.
3. **Hyperparameter Optimization**: Optuna-based sequential hyperparameter optimization using the TPE sampler (30 trials each for XGBoost and LightGBM).
4. **Rigorous Validation Benchmarking**: Full consolidation across all 9 model configurations evaluated on validation data.
5. **Best Phase 2 Candidate Selection & Serialization**: Deterministic export of the best candidate model, label encodings, feature metadata, and validation checks.

---

## 2. Dataset Specification
- **Primary Dataset**: `data/processed/tectonicsense_ml_dataset.csv`
- **Total Observations**: 239,757 seismic events.
- **USGS Acquisition Period**: Global events from 2010 to 2025 with magnitude $M \ge 4.0$.
- **Tectonic Model**: PB2002 plate boundary model (Bird, 2003). Note: Association used 20 candidate PB2002 segments and point-to-great-circle-arc distance refinement (not an oversimplified planar nearest-neighbor distance).
- **Missing Values**: Exactly 0 missing values across all predictive columns.
- **Duplicate Observations**: 5 duplicate feature rows were detected; event IDs remain unique. They were retained pending further investigation.
- **Feature Count**: 26 predictive features.
- **Strict Identifier Exclusion**: `event_id`, `boundary_id`, and `boundary_step_id` are strictly excluded from all model training and validation matrices to prevent data leakage and spatial memorization.

---

## 3. Feature Information
The 26 predictive features represent four distinct geophysical domains:

1. **Earthquake Characteristics (4 features)**:
   - `magnitude`: Reported earthquake magnitude ($M \ge 4.0$).
   - `magnitude_squared`: Quadratic magnitude non-linear scaling.
   - `depth_km`: Hypocentral depth in kilometers.
   - `depth_abs_km`: Absolute focal depth in kilometers.

2. **Global Spatial Coordinates & Cyclical Encodings (6 features)**:
   - `longitude`, `latitude`: Raw geographic coordinates in degrees.
   - `longitude_sin`, `longitude_cos`: Cyclical spherical harmonic encoding of longitude ($\sin(\lambda), \cos(\lambda)$).
   - `latitude_sin`, `latitude_cos`: Cyclical spherical harmonic encoding of latitude ($\sin(\phi), \cos(\phi)$).

3. **Boundary Proximity & Geometry (5 features)**:
   - `boundary_distance_km`: Great-circle arc distance to nearest refined PB2002 boundary segment.
   - `log_boundary_distance_km`: $\ln(1 + d)$ variance-stabilizing transformation.
   - `boundary_length_km`: Length of associated boundary segment in kilometers.
   - `boundary_azimuth_sin`, `boundary_azimuth_cos`: Cyclical sine and cosine of boundary strike angle.

4. **Kinematics & Plate Motion Dynamics (8 features)**:
   - `relative_velocity`: Relative plate motion rate (mm/yr).
   - `velocity_azimuth_sin`, `velocity_azimuth_cos`: Plate motion vector direction cyclical encodings.
   - `divergent_component`: Spreading/converging normal velocity component (mm/yr).
   - `right_lateral_component`: Dextral strike-slip velocity component (mm/yr).
   - `absolute_divergent_component`: Normal motion magnitude $|v_{\text{div}}|$.
   - `absolute_right_lateral_component`: Transform shear magnitude $|v_{\text{trans}}|$.
   - `tectonic_motion_magnitude`: Total plate motion vector magnitude $\sqrt{v_{\text{div}}^2 + v_{\text{trans}}^2}$.

5. **Geophysical Context (3 features)**:
   - `boundary_elevation_m`: Mean bathymetry/topography elevation along boundary segment (meters).
   - `seafloor_age_ma`: Lithospheric age of adjacent oceanic seafloor (Ma).
   - `orogen_flag`: Binary indicator (0/1) for plate boundaries traversing diffuse continental orogens.

---

## 4. Target Classes & Distribution
The prediction target is `boundary_class`, representing the 7 Peter Bird (PB2002) tectonic boundary classes:
- **SUB** (Subduction Zone): 115,347 observations (48.11%) — Dominant class reflecting intense circum-Pacific and island-arc seismicity.
- **CCB** (Continental Convergent Boundary): 27,281 observations (11.38%).
- **CRB** (Continental Rift Boundary): 26,316 observations (10.98%).
- **OSR** (Oceanic Spreading Ridge): 26,206 observations (10.93%).
- **CTF** (Continental Transform Fault): 17,431 observations (7.27%).
- **OTF** (Oceanic Transform Fault): 17,378 observations (7.25%).
- **OCB** (Oceanic Convergent Boundary): 9,798 observations (4.09%) — Minority class.

Because SUB constitutes ~48.1% of the observations and OCB constitutes ~4.1%, accuracy alone is insufficient. **Macro F1 score** is maintained as the primary evaluation metric across all experiments.

---

## 5. Dataset Splitting & Data Leakage Safeguards
A stratified 70% / 15% / 15% random split was utilized with a fixed random state (`random_state=42`):
- **Training Subset**: 167,829 observations (`data/processed/splits/train.csv`)
- **Validation Subset**: 35,964 observations (`data/processed/splits/validation.csv`)
- **Test Subset**: 35,964 observations (`data/processed/splits/test.csv`) — **HELD OUT UNTOUCHED**.

### Leakage Safeguards
- The test subset was never evaluated, accessed, or leaked during model training, selection, or hyperparameter optimization.
- Scalers (e.g., `StandardScaler` in the logistic regression pipeline) were fit exclusively on training data.
- Identifiers (`event_id`, `boundary_id`, `boundary_step_id`) were omitted from all model inputs.

---

## 6. Baseline Models Summary
Trained using `scripts/train_baselines.py`:
- **DummyClassifier** (`strategy="most_frequent"`): Validation Macro F1 of **0.0928** (Validation Accuracy: 0.4811). Sanity anchor.
- **LogisticRegression** (`StandardScaler` + `max_iter=1000` Pipeline): Validation Macro F1 of **0.9036** (Validation Accuracy: 0.9410, Training Time: 21.92s).
- **RandomForestClassifier** (`n_estimators=100`, `max_depth=25`, `n_jobs=-1`): Validation Macro F1 of **0.9996** (Validation Accuracy: 0.9998, Training Time: 22.70s).

---

## 7. Advanced Gradient Boosting Models Summary
Trained using `scripts/train_advanced_models.py`:
- **XGBoost (Default)**: Validation Macro F1 of **0.9996**, Validation Accuracy of 0.9998, Training Time: 21.25s.
- **LightGBM (Default)**: Validation Macro F1 of **0.9996**, Validation Accuracy of 0.9998, Training Time: 19.54s.
- **CatBoost (Default)**: Validation Macro F1 of **0.9972**, Validation Accuracy of 0.9983, Training Time: 43.23s.
- **HistGradientBoosting (Default)**: Validation Macro F1 of **0.9997**, Validation Accuracy of 0.9998, Training Time: 30.19s.

---

## 8. Optuna-Based Sequential Hyperparameter Optimization (TPE Sampler)
Conducted using `scripts/optimize_phase2_models.py`:
- **Sampler**: Optuna `TPESampler(seed=42)` (Tree-structured Parzen Estimator).
- **Objective**: Maximize validation set Macro F1 score strictly on training/validation splits.
- **Trial Budget**: 30 trials for XGBoost + 30 trials for LightGBM (60 total trials executed).

### Search Space & Optimal Parameters

#### XGBoost Optimization (30 Trials, 505.2s total duration)
- Best Trial: **#28**
- Best Validation Macro F1 during tuning: **0.999800**
- Optimal Parameters:
  - `n_estimators`: 120
  - `max_depth`: 9
  - `learning_rate`: 0.155952
  - `subsample`: 0.7
  - `colsample_bytree`: 0.8
  - `min_child_weight`: 2
  - `reg_lambda`: 0.035938
  - `tree_method`: "hist"
  - `objective`: "multi:softprob"

#### LightGBM Optimization (30 Trials, 573.9s total duration)
- Best Trial: **#12**
- Best Validation Macro F1 during tuning: **0.999724**
- Optimal Parameters:
  - `n_estimators`: 140
  - `num_leaves`: 40
  - `max_depth`: 11
  - `learning_rate`: 0.112346
  - `min_child_samples`: 60
  - `subsample`: 0.9
  - `colsample_bytree`: 0.6
  - `reg_lambda`: 0.002385
  - `objective`: "multiclass"

---

## 9. Consolidated Phase 2 Model Benchmark (Validation Set)

The final consolidated comparison across all 9 evaluated models, sorted primarily by Validation Macro F1 descending:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Training Time (s) |
|:------|:---------|:----------------|:-------------|:---------|:------------|:------------------|
| **Tuned XGBoost** | **0.9999** | **0.9998** | **0.9998** | **0.9998** | **0.9999** | 19.00 |
| **Tuned LightGBM** | 0.9998 | 0.9997 | 0.9997 | 0.9997 | 0.9998 | 19.35 |
| **HistGradientBoosting** | 0.9998 | 0.9997 | 0.9997 | 0.9997 | 0.9998 | 30.19 |
| **LightGBM (Default)** | 0.9998 | 0.9996 | 0.9996 | 0.9996 | 0.9998 | 19.54 |
| **XGBoost (Default)** | 0.9998 | 0.9997 | 0.9996 | 0.9996 | 0.9998 | 21.25 |
| **RandomForest** | 0.9998 | 0.9997 | 0.9996 | 0.9996 | 0.9998 | 22.70 |
| **CatBoost** | 0.9983 | 0.9974 | 0.9969 | 0.9972 | 0.9983 | 43.23 |
| **LogisticRegression** | 0.9410 | 0.9060 | 0.9020 | 0.9036 | 0.9407 | 21.92 |
| **DummyClassifier** | 0.4811 | 0.0687 | 0.1429 | 0.0928 | 0.3126 | 0.10 |

---

## 10. Best Phase 2 Candidate Selection
**Selected Model**: **Tuned XGBoost**
- **Validation Macro F1**: **0.9998**
- **Validation Accuracy**: **0.9999**
- **Validation Macro Recall**: **0.9998**
- **Validation Macro Precision**: **0.9998**
- **Validation Weighted F1**: **0.9999**
- **Training Duration**: 19.00 seconds
- **Inference Latency**: 0.2448 seconds on 35,964 validation observations (~6.8 microseconds per event)

### Selection Rationale & Tie-Breaking
1. **Macro F1 Supremacy**: Tuned XGBoost achieved the highest validation Macro F1 score (0.9998) among all 9 models.
2. **Equitable Minority Representation**: Achieved 0.9997 F1 on the minority class `OCB` (1,469 events) and 0.9999 F1 on the majority class `SUB` (17,303 events).
3. **Computational Efficiency**: Faster inference latency (0.2448s) than Tuned LightGBM (1.4575s) and HistGradientBoosting (1.8265s).

---

## 11. Model Artifacts & Serialization
All candidate model artifacts are saved in `models/phase_2_best_candidate/`:
- `best_candidate_model.joblib`: Trained model pipeline (compressed joblib format).
- `metadata.json`: Complete specification including feature names, hyperparameters, validation metrics, and dataset sizes.
- `feature_names.json`: Exact list of the 26 predictive features.
- `label_mapping.json`: Integer to PB2002 string class mappings (`0: CCB`, `1: CRB`, `2: CTF`, `3: OCB`, `4: OSR`, `5: OTF`, `6: SUB`).

Functionality was verified using `scripts/verify_phase_2_model.py`.

---

## 12. Phase 2 Visualizations
Publication-quality figures saved under `results/phase_2_machine_learning/figures/`:
1. `model_macro_f1_comparison.png`: Bar chart of Macro F1 scores across all 9 models.
2. `model_accuracy_comparison.png`: Validation accuracy comparison.
3. `model_precision_recall_comparison.png`: Grouped bar chart comparing Macro Precision vs. Macro Recall.
4. `model_training_time_comparison.png`: Computational latency in seconds.
5. `best_candidate_class_performance.png`: Per-class precision, recall, and F1-score for Tuned XGBoost.
6. `optimization_history.png`: Optuna convergence trajectory across 30 trials each for XGBoost and LightGBM.

---

## 13. Limitations & Scientific Caveats
1. **Random Stratified Split Characteristic**: Random splitting causes observations from the same tectonic boundaries to appear in both train and validation subsets. A true out-of-boundary generalization experiment (`boundary_id` grouped split) will be conducted in Phase 3.
2. **PB2002 Association Dependency**: Features such as `relative_velocity`, `boundary_elevation_m`, and `seafloor_age_ma` are derived from the associated PB2002 segment. Consequently, tree ensembles easily identify the boundary regime when kinematic and geospatial variables are provided.
3. **Validation vs. Test**: Validation scores reflect model development performance and must not be confused with final unbiased test generalization.
