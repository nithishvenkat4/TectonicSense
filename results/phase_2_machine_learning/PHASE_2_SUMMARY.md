# TectonicSense — Phase 2 Machine Learning Summary

## 1. Objective
The objective of Phase 2 was to build a rigorous, reproducible machine learning benchmarking and optimization pipeline for tectonic plate boundary classification (`boundary_class`) across seven Peter Bird (PB2002) tectonic regimes. The modeling progressed systematically through:
1. Baseline models (DummyClassifier, LogisticRegression, RandomForest)
2. Advanced gradient boosting models (XGBoost, LightGBM, CatBoost, HistGradientBoosting)
3. Optuna-based sequential hyperparameter optimization using the TPE sampler
4. Multi-model consolidated benchmarking
5. Best candidate selection, serialization, and verification

> [!IMPORTANT]
> **Scientific Scope**: This task is the classification of tectonic plate-boundary regimes associated with seismic and geospatial observations. It is NOT exact earthquake prediction or earthquake forecasting.

---

## 2. Dataset
- **Engineered Dataset**: `data/processed/tectonicsense_ml_dataset.csv`
- **Total Observations**: 239,757 seismic events from the USGS catalog (2010–2025, $M \ge 4.0$).
- **Association Model**: PB2002 plate-boundary model (Bird, 2003) using 20 candidate PB2002 segments and point-to-great-circle-arc distance refinement.
- **Predictive Features**: 26 domain-engineered features (earthquake energy, spherical spatial encodings, boundary proximity, kinematics, seafloor age, elevation, orogen flag).
- **Leakage Prevention**: `event_id`, `boundary_id`, and `boundary_step_id` were strictly excluded from predictive feature matrices.
- **Data Integrity**: 0 missing values. 5 duplicate feature rows were detected; event IDs remain unique. They were retained pending further investigation.
- **Stratified Split**:
  - Training (70%): 167,829 samples (`data/processed/splits/train.csv`)
  - Validation (15%): 35,964 samples (`data/processed/splits/validation.csv`)
  - Test (15%): 35,964 samples (`data/processed/splits/test.csv`) — **held out completely untouched**.
- **Class Distribution**: SUB (48.11%), CCB (11.38%), CRB (10.98%), OSR (10.93%), CTF (7.27%), OTF (7.25%), OCB (4.09%).

---

## 3. Baseline Models
Evaluated on the validation set using `scripts/train_baselines.py`:
- **DummyClassifier** (`strategy="most_frequent"`):
  - Validation Accuracy: **0.4811** | Macro Precision: **0.0687** | Macro Recall: **0.1429** | Macro F1: **0.0928** | Weighted F1: **0.3126** | Training Time: **0.10s**
- **LogisticRegression** (`StandardScaler` pipeline, `max_iter=1000`):
  - Validation Accuracy: **0.9410** | Macro Precision: **0.9060** | Macro Recall: **0.9020** | Macro F1: **0.9036** | Weighted F1: **0.9407** | Training Time: **21.92s**
- **RandomForest** (`n_estimators=100`, `max_depth=25`, `n_jobs=-1`):
  - Validation Accuracy: **0.9998** | Macro Precision: **0.9997** | Macro Recall: **0.9996** | Macro F1: **0.9996** | Weighted F1: **0.9998** | Training Time: **22.70s**

---

## 4. Advanced Models
Evaluated on the validation set using `scripts/train_advanced_models.py`:
- **XGBoost (Default)**:
  - Validation Accuracy: **0.9998** | Macro Precision: **0.9997** | Macro Recall: **0.9996** | Macro F1: **0.9996** | Weighted F1: **0.9998** | Training Time: **21.25s**
- **LightGBM (Default)**:
  - Validation Accuracy: **0.9998** | Macro Precision: **0.9996** | Macro Recall: **0.9996** | Macro F1: **0.9996** | Weighted F1: **0.9998** | Training Time: **19.54s**
- **CatBoost (Default)**:
  - Validation Accuracy: **0.9983** | Macro Precision: **0.9974** | Macro Recall: **0.9969** | Macro F1: **0.9972** | Weighted F1: **0.9983** | Training Time: **43.23s**
- **HistGradientBoosting (Default)**:
  - Validation Accuracy: **0.9998** | Macro Precision: **0.9997** | Macro Recall: **0.9997** | Macro F1: **0.9997** | Weighted F1: **0.9998** | Training Time: **30.19s**

---

## 5. Hyperparameter Optimization
Performed using `scripts/optimize_phase2_models.py` via **Optuna-based sequential hyperparameter optimization using the TPE sampler**:
- Sampler: `TPESampler(seed=42)`
- Target Metric: Validation Macro F1
- Search Budget: 30 trials for XGBoost (505.2s), 30 trials for LightGBM (573.9s) — 60 trials total.

### Optimal Parameters Obtained
- **XGBoost** (Best Trial #28, Best Tuning Val Macro F1 = 0.999800):
  - `n_estimators`: 120
  - `max_depth`: 9
  - `learning_rate`: 0.155952
  - `subsample`: 0.7
  - `colsample_bytree`: 0.8
  - `min_child_weight`: 2
  - `reg_lambda`: 0.035938
- **LightGBM** (Best Trial #12, Best Tuning Val Macro F1 = 0.999724):
  - `n_estimators`: 140
  - `num_leaves`: 40
  - `max_depth`: 11
  - `learning_rate`: 0.112346
  - `min_child_samples`: 60
  - `subsample`: 0.9
  - `colsample_bytree`: 0.6
  - `reg_lambda`: 0.002385

---

## 6. Final Validation Comparison
Consolidated benchmark across all 9 evaluated models, ranked primarily by Validation Macro F1 descending:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Training Time |
|:------|:---------|:----------------|:-------------|:---------|:------------|:--------------|
| **Tuned XGBoost** | **0.9999** | **0.9998** | **0.9998** | **0.9998** | **0.9999** | **19.00s** |
| **Tuned LightGBM** | 0.9998 | 0.9997 | 0.9997 | 0.9997 | 0.9998 | 19.35s |
| **HistGradientBoosting** | 0.9998 | 0.9997 | 0.9997 | 0.9997 | 0.9998 | 30.19s |
| **LightGBM** | 0.9998 | 0.9996 | 0.9996 | 0.9996 | 0.9998 | 19.54s |
| **XGBoost** | 0.9998 | 0.9997 | 0.9996 | 0.9996 | 0.9998 | 21.25s |
| **RandomForest** | 0.9998 | 0.9997 | 0.9996 | 0.9996 | 0.9998 | 22.70s |
| **CatBoost** | 0.9983 | 0.9974 | 0.9969 | 0.9972 | 0.9983 | 43.23s |
| **LogisticRegression** | 0.9410 | 0.9060 | 0.9020 | 0.9036 | 0.9407 | 21.92s |
| **DummyClassifier** | 0.4811 | 0.0687 | 0.1429 | 0.0928 | 0.3126 | 0.10s |

---

## 7. Best Candidate
- **Model**: **Tuned XGBoost**
- **Why Selected**:
  1. **Highest Validation Macro F1 (0.9998)**: Superior balanced performance across all classes.
  2. **High Macro Precision (0.9998) and Recall (0.9998)**: Zero class neglect on minority classes (`OCB` F1: 0.9997, `SUB` F1: 0.9999).
  3. **High Overall Accuracy (0.9999)** and Weighted F1 (0.9999).
  4. **Fast Latency**: Lowest validation inference time (0.2448s across 35,964 samples, ~6.8 $\mu$s per event) and 19.00s training duration.
- **Validation Metrics**:
  - Accuracy: 0.9999
  - Macro Precision: 0.9998
  - Macro Recall: 0.9998
  - Macro F1: 0.9998
  - Weighted F1: 0.9999
  - Training Time: 19.00s
  - Inference Time: 0.2448s
- **Hyperparameters**:
  `n_estimators=120`, `max_depth=9`, `learning_rate=0.155952`, `subsample=0.7`, `colsample_bytree=0.8`, `min_child_weight=2`, `reg_lambda=0.035938`, `tree_method="hist"`, `objective="multi:softprob"`.

---

## 8. Model Serialization
The best candidate model and all supporting metadata are serialized under:
`models/phase_2_best_candidate/`

Files:
- `models/phase_2_best_candidate/best_candidate_model.joblib`: Serialized model pipeline.
- `models/phase_2_best_candidate/metadata.json`: Specification, versioning, hyperparameter dictionary, and validation results.
- `models/phase_2_best_candidate/feature_names.json`: Complete 26-feature list.
- `models/phase_2_best_candidate/label_mapping.json`: PB2002 target class mapping dictionary.

Verified using `scripts/verify_phase_2_model.py`.

---

## 9. Key Findings
1. **Tree Ensembles Outperform Linear Baselines**: Multinomial Logistic Regression reached 0.9036 Macro F1, whereas all tree-based ensemble methods surpassed 0.9970 Macro F1.
2. **Optuna Tuning Provides Measurable Gains**: TPE optimization raised XGBoost Macro F1 from 0.9996 to 0.9998 and Accuracy from 0.9998 to 0.9999, reducing misclassifications by over 50%.
3. **No Minority Class Collapse**: Despite SUB representing 48.1% of the data and OCB representing only 4.1%, all classes achieved F1 scores exceeding 0.9996 in the tuned candidate model.
4. **Computational Feasibility**: Histogram-based boosting enabled training 167,829 samples in 19 seconds.

---

## 10. Limitations
1. **Class Imbalance**: Subduction zones (SUB) constitute nearly half of all global earthquakes, whereas OCB represents 4.09%. While Macro F1 remains high under the current split, real-world generalization requires ongoing vigilance.
2. **PB2002 Association Dependency**: The features include plate motion kinematics (`relative_velocity`, `divergent_component`, `right_lateral_component`) and geophysical attributes (`seafloor_age_ma`, `boundary_elevation_m`) derived from the PB2002 association. These provide strong discriminant signals.
3. **Random Stratified Split Limitation**: The primary split is stratified randomly by observation. As a result, earthquakes occurring along the same boundary fault segment can appear in both train and validation splits.
4. **Potential Feature-Label Relationship Through PB2002 Metadata**: The association pipeline links earthquakes to the nearest refined boundary segment. If a segment's physical motion is well-defined, the boundary class is predictably aligned with the motion vector components.
5. **Validation Metrics Are Not Final Unbiased Test Performance**: The validation metrics reported here guided model development and tuning. Final unbiased test evaluation must be performed on the untouched test set (`data/processed/splits/test.csv`) in Phase 3.

---

## 11. Phase 3 Readiness
Phase 2 is fully complete. The following assets are validated and ready for Phase 3:
- Fully trained and serialized candidate model: `models/phase_2_best_candidate/best_candidate_model.joblib`
- Verification script: `scripts/verify_phase_2_model.py`
- Benchmarking tables and JSON metrics: `results/phase_2_machine_learning/model_comparison/`
- Completely untouched test split: `data/processed/splits/test.csv` (reserved for final evaluation)
- Out-of-boundary grouping identifiers ready for boundary-aware generalization experiments (`boundary_id`).
