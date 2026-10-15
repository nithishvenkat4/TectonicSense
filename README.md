# TectonicSense

## Explainable Machine Learning for Tectonic Plate Boundary Classification Using Seismic and Geospatial Features

TectonicSense is an end-to-end machine learning research and deployment
project for classifying the tectonic environment associated with seismic
observations into seven PB2002-derived plate-boundary regimes.

The project combines global earthquake observations from the United
States Geological Survey (USGS) with the PB2002 digital plate-boundary
model and engineered seismic, geographic, geometric, kinematic, and
geophysical features. A tuned XGBoost classifier is used as the final
production model, with SHAP-based explainability, boundary-aware
evaluation, automated model verification, a FastAPI inference service,
Docker containerization, AWS ECR/ECS Fargate deployment, and a
model-focused web interface.

> **Scientific scope:** TectonicSense is a tectonic plate-boundary
> classification system. It is not an earthquake prediction, earthquake
> forecasting, or future-magnitude prediction system.

------------------------------------------------------------------------

## 1. Project Overview

### Problem Statement

Earthquake observations contain information about the geological and
tectonic environment in which seismic activity occurs. However,
classifying these observations into meaningful tectonic boundary regimes
requires combining multiple heterogeneous information sources,
including:

-   Seismic characteristics
-   Geographic coordinates
-   Distance to mapped plate boundaries
-   Boundary geometry
-   Plate-motion information
-   Boundary elevation and seafloor age
-   Orogenic information

TectonicSense formulates this as a seven-class multiclass classification
problem using PB2002 boundary classifications as the target labels.

### Project Objective

The primary objective is to develop an explainable and deployable
machine learning system that:

1.  Processes large-scale global earthquake observations.
2.  Associates seismic observations with PB2002 plate-boundary segments.
3.  Constructs a rigorous 26-feature machine learning representation.
4.  Benchmarks multiple machine learning algorithms.
5.  Optimizes an advanced gradient-boosting model.
6.  Evaluates both conventional random-split performance and
    boundary-disjoint generalization.
7.  Explains model decisions using SHAP.
8.  Provides reproducible production inference.
9.  Exposes the model through a REST API.
10. Deploys the API as a Dockerized AWS ECS Fargate service.
11. Provides a web interface for interactive classification.

------------------------------------------------------------------------

## 2. Key Results

The project was evaluated using two complementary evaluation protocols.

### Random Stratified Test

The final model was evaluated on 35,964 completely held-out test
observations.

  Metric                                    Result
  -------------------- ---------------------------
  Accuracy                                99.9889%
  Macro Precision                         99.9821%
  Macro Recall                            99.9761%
  Macro F1                                99.9791%
  Weighted F1                             99.9889%
  Misclassifications                    4 / 35,964
  Error Rate                               0.0111%
  Inference Latency      13.80 microseconds/sample

### Boundary-Aware Evaluation

A stricter group-disjoint evaluation was performed using `boundary_id`,
ensuring that test boundaries were completely unseen during training.

  Metric              Random Stratified   Boundary-Aware
  ----------------- ------------------- ----------------
  Accuracy                       99.99%           79.94%
  Macro Precision                99.98%           79.89%
  Macro Recall                   99.98%           80.78%
  Macro F1                       99.98%           79.35%
  Weighted F1                    99.99%           80.05%

The difference demonstrates why spatial and structural group-aware
evaluation is important for tectonic machine learning. The random split
measures performance when related observations from mapped boundaries
can occur in both training and testing, while the boundary-aware
experiment provides a more conservative assessment of transfer to unseen
PB2002 boundaries.

------------------------------------------------------------------------

## 3. Target Classes

TectonicSense predicts seven PB2002-derived boundary classes:

  Class   Description
  ------- --------------------------------
  `CCB`   Continental Collision Boundary
  `CRB`   Continental Rift Boundary
  `CTF`   Continental Transform Fault
  `OCB`   Oceanic-Continental Boundary
  `OSR`   Oceanic Spreading Ridge
  `OTF`   Oceanic Transform Fault
  `SUB`   Subduction Boundary

The original PB2002 representation may contain starred boundary labels
associated with orogen markers. These are handled separately through the
`orogen_flag` feature rather than being treated as additional target
classes.

------------------------------------------------------------------------

## 4. Dataset

### USGS Earthquake Dataset

Global earthquake observations were collected from USGS for:

-   Time period: 2010--2025
-   Minimum magnitude: 4.0
-   Maximum observed magnitude: 9.1
-   Total raw observations: 239,853
-   Clean earthquake observations: 239,757
-   Unique event IDs: 239,853 in the original collection
-   Missing values after cleaning: None

The combined cleaned earthquake dataset is:

``` text
data/processed/usgs_earthquakes_clean.csv
```

### PB2002 Dataset

The project uses the PB2002 digital plate-boundary model.

Relevant source files:

``` text
data/raw/pb2002/pb2002_boundaries.dig
data/raw/pb2002/pb2002_steps.dat
```

The PB2002 step dataset contains boundary geometry and tectonic
information including:

-   Boundary ID
-   Step ID
-   Start/end coordinates
-   Segment length
-   Segment azimuth
-   Relative velocity
-   Velocity azimuth
-   Divergent component
-   Right-lateral component
-   Elevation
-   Seafloor age
-   Boundary class

The parsed PB2002 dataset contains 5,819 boundary-step records
representing 182 unique boundaries.

### Earthquake-to-Boundary Association

Earthquake observations were associated with PB2002 boundary segments
using a refined spatial association procedure.

The procedure first identifies candidate boundary segments using
midpoint-based spatial indexing and then evaluates
point-to-great-circle-arc distances for the candidate segments.

The refined association dataset contains:

``` text
239,757 earthquake observations
```

The mean association distance is approximately:

``` text
121.203 km
```

with a median of approximately:

``` text
69.168 km
```

------------------------------------------------------------------------

## 5. Feature Engineering

The final model uses exactly 26 features.

### Seismic Features

1.  `magnitude`
2.  `depth_km`

### Engineered Seismic Features

3.  `magnitude_squared`
4.  `depth_abs_km`

### Geographic Features

5.  `longitude`
6.  `latitude`
7.  `longitude_sin`
8.  `longitude_cos`
9.  `latitude_sin`
10. `latitude_cos`

Cyclical encodings are used to represent the periodic nature of
longitude and provide smooth numerical representations of geographic
coordinates.

### Boundary Geometry Features

11. `boundary_distance_km`
12. `log_boundary_distance_km`
13. `boundary_length_km`
14. `boundary_azimuth_sin`
15. `boundary_azimuth_cos`

### Tectonic and Geophysical Features

16. `relative_velocity`
17. `velocity_azimuth_sin`
18. `velocity_azimuth_cos`
19. `divergent_component`
20. `right_lateral_component`
21. `absolute_divergent_component`
22. `absolute_right_lateral_component`
23. `tectonic_motion_magnitude`
24. `boundary_elevation_m`
25. `seafloor_age_ma`
26. `orogen_flag`

The identifiers `boundary_id` and `boundary_step_id` are deliberately
excluded from model input to reduce direct identifier-based label
leakage.

------------------------------------------------------------------------

## 6. Feature Engineering Formulation

The tectonic motion magnitude is represented as:

\[ V\_{tectonic} = `\sqrt{
V_{divergent}^{2} +
V_{right-lateral}^{2}
}`{=tex} \]

Absolute motion components are also generated:

\[ V\_{div,abs} = \|V\_{divergent}\| \]

\[ V\_{rl,abs} = \|V\_{right-lateral}\| \]

Angular variables are encoded using sine and cosine transformations.

For longitude:

\[ lon\_{sin} = `\sin`{=tex}(lon) \]

\[ lon\_{cos} = `\cos`{=tex}(lon) \]

Similarly:

\[ lat\_{sin} = `\sin`{=tex}(lat) \]

\[ lat\_{cos} = `\cos`{=tex}(lat) \]

These transformations allow the model to represent angular continuity
without treating geographic angles as ordinary linear variables.

------------------------------------------------------------------------

## 7. Data Splitting

The main machine learning dataset contains:

``` text
239,757 observations
26 features
7 target classes
```

A stratified split with random seed `42` was used:

  Partition      Samples   Percentage
  ------------ --------- ------------
  Training       167,829          70%
  Validation      35,964          15%
  Test            35,964          15%
  Total          239,757         100%

The validation set was used during model selection and hyperparameter
optimization.

The test set remained untouched until final evaluation.

A separate boundary-aware evaluation was performed using `boundary_id`
grouping so that train, validation, and test contained completely
disjoint boundary groups.

------------------------------------------------------------------------

## 8. Machine Learning Models

The project evaluates multiple baseline and advanced algorithms.

### Baseline Models

-   Dummy Classifier
-   Logistic Regression
-   Random Forest

### Advanced Models

-   XGBoost
-   LightGBM
-   CatBoost
-   HistGradientBoosting

The benchmark was designed to compare simple baselines with modern
gradient-boosting methods.

------------------------------------------------------------------------

## 9. Model Benchmark

Validation results:

  Model                    Accuracy   Macro F1   Weighted F1
  ---------------------- ---------- ---------- -------------
  Tuned XGBoost              0.9999     0.9998        0.9999
  Tuned LightGBM             0.9998     0.9998        0.9998
  HistGradientBoosting       0.9998     0.9997        0.9998
  LightGBM                   0.9998     0.9996        0.9998
  XGBoost                    0.9998     0.9996        0.9998
  Random Forest              0.9998     0.9996        0.9998
  CatBoost                   0.9983     0.9972        0.9983
  Logistic Regression        0.9410     0.9036        0.9407
  Dummy Classifier           0.4811     0.1429        0.3126

Tuned XGBoost was selected as the final production architecture.

------------------------------------------------------------------------

## 10. Hyperparameter Optimization

Optuna was used to optimize XGBoost and LightGBM.

Optimization characteristics:

-   Framework: Optuna
-   Sampler: TPE
-   Random seed: 42
-   Trials per advanced candidate: 30
-   Objective: Validation Macro F1

The final XGBoost configuration is:

  Parameter                         Value
  -------------------- ------------------
  `n_estimators`                      120
  `max_depth`                           9
  `learning_rate`            0.1559524636
  `subsample`                         0.7
  `colsample_bytree`                  0.8
  `min_child_weight`                    2
  `reg_lambda`               0.0359375612
  `objective`            `multi:softprob`
  `eval_metric`                `mlogloss`
  `tree_method`                    `hist`
  `random_state`                       42
  `n_jobs`                             -1

Framework version:

``` text
XGBoost 3.0.5
```

Estimator:

``` text
XGBClassifier
```

------------------------------------------------------------------------

## 11. Final Production Model

The final production model is stored under:

``` text
models/final/
```

Artifacts:

``` text
models/final/
├── tectonicsense_xgboost.joblib
├── feature_names.json
├── label_mapping.json
└── metadata.json
```

Model binary size:

``` text
842,753 bytes
```

Approximately:

``` text
823 KB
```

The model expects exactly 26 features and produces probabilities for all
seven target classes.

------------------------------------------------------------------------

## 12. Explainability

SHAP TreeExplainer was used to interpret the final model.

SHAP analysis was performed on 2,000 stratified test observations.

Top features by mean absolute SHAP value:

    Rank Feature                                  Mean
  ------ ------------------------------------ --------
       1 `divergent_component`                  2.0318
       2 `boundary_elevation_m`                 1.4451
       3 `absolute_divergent_component`         0.8775
       4 `seafloor_age_ma`                      0.4082
       5 `absolute_right_lateral_component`     0.3891

The results indicate that tectonic motion characteristics and boundary
geophysical properties strongly influence the model's classification
decisions.

However, SHAP values represent statistical model associations. They
should not be interpreted as proof of causal geological mechanisms.

------------------------------------------------------------------------

## 13. Feature Ablation

Feature ablation was used to investigate where the high classification
performance originates.

  Feature Configuration     Accuracy   Macro F1
  ----------------------- ---------- ----------
  Seismic only                50.89%     20.28%
  Seismic + Geographic        95.34%     92.73%
  Limited tectonic            99.98%     99.96%
  Full 26 features            99.99%     99.98%

This experiment demonstrates that geographic and tectonic information
contributes substantially more to boundary classification than seismic
magnitude/depth information alone under the evaluated representation.

------------------------------------------------------------------------

## 14. Error Analysis

Only four errors occurred in the random held-out test set.

  Event          True   Predicted     Confidence
  -------------- ------ ----------- ------------
  `us20009vsi`   OCB    CCB               0.9817
  `us6000pzw2`   OSR    OTF               0.9794
  `us1000e7ye`   OCB    OTF               0.5577
  `usb000fsh5`   SUB    CCB               0.8174

The errors occur in physically ambiguous or transition-like
environments, including ridge-transform intersections, oblique
convergence/transform zones, and distal events near tectonic regime
transitions.

------------------------------------------------------------------------

## 15. Scientific Interpretation

TectonicSense learns an empirical mapping between seismic observations,
geographic position, PB2002 boundary geometry, and PB2002-derived
tectonic metadata.

The high random-split performance should therefore not be interpreted as
independent discovery of tectonic boundary types from seismic
observations alone.

Several predictor variables, including relative velocity and boundary
geometry, are derived from the same PB2002 tectonic framework used to
define the target classes.

This creates an important methodological consideration: the model
demonstrates strong classification capability within the represented
PB2002 feature space, but its performance on completely unseen tectonic
structures is more modest.

The boundary-aware Macro F1 of 79.35% provides the more conservative
generalization result.

------------------------------------------------------------------------

## 16. Limitations

The project has several important limitations.

### PB2002-Derived Ground Truth

The target classes are derived from the PB2002 geological model rather
than directly measured physical labels.

### Metadata Dependence

Several model features originate from PB2002 metadata. Consequently, the
model partly learns the mapping encoded by the geological model itself.

### Spatial Autocorrelation

Random splitting allows observations associated with the same boundary
to occur in both training and testing.

### Boundary-Aware Generalization

The drop from approximately 99.98% Macro F1 to 79.35% Macro F1
demonstrates the importance of testing on completely unseen boundary
groups.

### No Earthquake Forecasting

The model does not predict:

-   Whether an earthquake will occur
-   When an earthquake will occur
-   Where a future earthquake will occur
-   The magnitude of a future earthquake

### Geographic Scope

The system is based on mapped PB2002 plate boundaries and may not
generalize equally well to diffuse intraplate seismicity or tectonic
settings poorly represented by the source model.

------------------------------------------------------------------------

## 17. Production Inference Pipeline

The production inference engine is implemented in:

``` text
scripts/predict.py
```

The inference system provides:

-   Single-observation prediction
-   Batch prediction
-   Exact feature-schema validation
-   Automatic feature ordering
-   Numeric type validation
-   NaN rejection
-   Infinite-value rejection
-   Probability outputs
-   Predicted class
-   Predicted class ID
-   Confidence score

The predictor is loaded as a singleton so that the model binary is not
unnecessarily reloaded for every API request.

------------------------------------------------------------------------

## 18. Model Verification

An independent verification suite is implemented in:

``` text
scripts/verify_final_model.py
```

The final model passed:

``` text
16 / 16 verification checks
```

Verification categories include:

1.  Model artifact integrity
2.  Feature schema
3.  Label mapping
4.  Model architecture
5.  Expected feature count
6.  Expected class count
7.  Single inference
8.  Batch inference
9.  Probability normalization
10. Deterministic inference
11. Missing-feature rejection
12. Non-numeric input rejection
13. NaN rejection
14. Infinite-value rejection
15. Truncated-schema rejection
16. Repeated prediction consistency

------------------------------------------------------------------------

## 19. REST API

The inference service is implemented using:

-   FastAPI
-   Uvicorn
-   Pydantic

### API Endpoints

  Endpoint           Method   Purpose
  ------------------ -------- -----------------------------------
  `/health`          GET      Service health check
  `/model-info`      GET      Model metadata
  `/predict`         POST     Single observation classification
  `/predict-batch`   POST     Batch classification
  `/docs`            GET      Swagger API documentation
  `/redoc`           GET      ReDoc API documentation
  `/openapi.json`    GET      OpenAPI specification

### API Validation

The API enforces:

-   Exactly 26 model features
-   Numeric feature values
-   No NaN
-   No infinity
-   No unexpected attributes
-   No `boundary_id`
-   No `boundary_step_id`
-   No `boundary_class`
-   Batch size between 1 and 1000

Automated API testing achieved:

``` text
17 / 17 tests passed
```

------------------------------------------------------------------------

## 20. Live Deployment

### Frontend

Production frontend:

https://tectonicsense.vercel.app

The frontend provides a model-focused interface for:

1.  Entering an observation
2.  Selecting tectonic presets
3.  Viewing the tectonic map
4.  Sending the observation to the API
5.  Receiving the predicted class
6.  Viewing confidence
7.  Viewing seven-class probabilities
8.  Viewing model feature importance

### Backend API

Production FastAPI service:

http://54.225.30.119:8000

### Swagger Documentation

http://54.225.30.119:8000/docs

### Health Endpoint

http://54.225.30.119:8000/health

### Model Information

http://54.225.30.119:8000/model-info

> The backend is currently exposed through the public IP of the ECS
> Fargate task. Because a Fargate task public IP can change if the task
> is replaced, a production-hardened architecture should place the
> service behind an Application Load Balancer with HTTPS and a stable
> DNS name.

------------------------------------------------------------------------

## 21. Cloud Architecture

The deployed system follows this architecture:

``` text
User
  |
  v
Vercel Frontend
  |
  | HTTPS / API proxy
  v
AWS ECS Fargate
  |
  v
FastAPI
  |
  v
Production Predictor
  |
  v
Tuned XGBoost Model
  |
  v
Tectonic Boundary Classification
```

### Container Registry

Docker images are stored in Amazon Elastic Container Registry (ECR).

Repository:

``` text
222907083247.dkr.ecr.us-east-1.amazonaws.com/ml-app
```

TectonicSense image tag:

``` text
tectonicsense
```

### ECS

Cluster:

``` text
tectonicsense-cluster
```

Service:

``` text
tectonicsense-api-service
```

Task definition:

``` text
tectonicsense-api:1
```

Launch type:

``` text
AWS Fargate
```

Container port:

``` text
8000
```

CPU:

``` text
512
```

Memory:

``` text
1024 MB
```

Cloud logging:

``` text
/ecs/tectonicsense-api
```

------------------------------------------------------------------------

## 22. Docker

The backend is containerized using a Python 3.11 slim image.

Build:

``` bash
docker build -t tectonicsense-api:local .
```

Run:

``` bash
docker run --rm -d \
  --name tectonicsense-api \
  -p 8000:8000 \
  tectonicsense-api:local
```

Health check:

``` bash
curl http://localhost:8000/health
```

The Docker image contains:

``` text
api/
scripts/
configs/
models/final/
requirements.txt
```

Large research datasets, notebooks, generated results, and frontend
dependencies are excluded from the production image.

------------------------------------------------------------------------

## 23. Project Structure

``` text
TectonicSense/
│
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── tests/
│
├── app/
│
├── configs/
│   └── final_model.yaml
│
├── data/
│   ├── raw/
│   │   ├── usgs/
│   │   └── pb2002/
│   ├── interim/
│   └── processed/
│
├── docs/
│
├── frontend/
│
├── models/
│   ├── final/
│   │   ├── tectonicsense_xgboost.joblib
│   │   ├── feature_names.json
│   │   ├── label_mapping.json
│   │   └── metadata.json
│   └── phase_2_best_candidate/
│
├── notebooks/
│
├── results/
│   ├── phase_2_machine_learning/
│   ├── phase_3_evaluation/
│   ├── phase_4_final_model/
│   └── phase_5_api/
│
├── scripts/
│   ├── feature_engineering.py
│   ├── create_dataset_splits.py
│   ├── train_baselines.py
│   ├── train_advanced_models.py
│   ├── optimize_phase2_models.py
│   ├── compare_phase2_models.py
│   ├── serialize_best_candidate.py
│   ├── finalize_phase2.py
│   ├── evaluate_phase3.py
│   ├── feature_ablation.py
│   ├── boundary_aware_evaluation.py
│   ├── shap_analysis.py
│   ├── error_analysis.py
│   ├── train_final_model.py
│   ├── predict.py
│   └── verify_final_model.py
│
├── tests/
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

## 24. Reproducibility

### Create Environment

``` bash
python -m venv .venv
```

Windows PowerShell:

``` powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

### Production Inference Self-Test

``` bash
python scripts/predict.py
```

### Model Verification

``` bash
python scripts/verify_final_model.py
```

### API Tests

``` bash
pytest api/tests -v
```

### Run FastAPI Locally

``` bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Then open:

``` text
http://127.0.0.1:8000/docs
```

------------------------------------------------------------------------

## 25. Research and Development Pipeline

The project was developed through controlled phases.

### Phase 1 --- Data Engineering

-   USGS data collection
-   PB2002 parsing
-   Data cleaning
-   Spatial association
-   Feature engineering
-   Dataset validation

### Phase 2 --- Machine Learning

-   Baseline models
-   Advanced models
-   Benchmarking
-   Optuna optimization
-   Candidate model selection

### Phase 3 --- Scientific Evaluation

-   Untouched test evaluation
-   SHAP analysis
-   Feature ablation
-   Boundary-aware evaluation
-   Error analysis

### Phase 4 --- Production Model

-   Frozen configuration
-   Final training
-   Artifact serialization
-   Production inference
-   Independent verification

### Phase 5 --- Deployment

-   FastAPI service
-   API testing
-   Docker containerization
-   Amazon ECR
-   Amazon ECS Fargate
-   Vercel frontend
-   End-to-end deployment validation

------------------------------------------------------------------------

## 26. Technology Stack

### Programming

-   Python
-   TypeScript
-   JavaScript

### Machine Learning

-   XGBoost
-   LightGBM
-   CatBoost
-   Scikit-learn
-   Optuna
-   SHAP

### Data Processing

-   Pandas
-   NumPy
-   SciPy

### Visualization

-   Matplotlib
-   Seaborn
-   SHAP visualizations
-   MapLibre

### Backend

-   FastAPI
-   Uvicorn
-   Pydantic

### Frontend

-   React
-   TypeScript
-   Vite
-   MapLibre

### Deployment

-   Docker
-   Amazon ECR
-   Amazon ECS
-   AWS Fargate
-   Vercel

### Reproducibility and Testing

-   Pytest
-   JSON configuration
-   YAML configuration
-   Joblib model serialization

------------------------------------------------------------------------

## 27. Important Generated Artifacts

Important project outputs include:

``` text
data/processed/tectonicsense_ml_dataset.csv
data/processed/usgs_earthquakes_clean.csv
```

Final model:

``` text
models/final/tectonicsense_xgboost.joblib
```

Final configuration:

``` text
configs/final_model.yaml
```

Final model evaluation:

``` text
results/phase_4_final_model/final_model_evaluation.json
```

Final verification:

``` text
results/phase_4_final_model/final_model_verification.json
```

API test report:

``` text
results/phase_5_api/api_test_report.json
```

Production documentation:

``` text
docs/phase_4_final_model.md
docs/phase_5_api.md
```

------------------------------------------------------------------------

## 28. Deployment Verification

The deployed system was verified at multiple levels.

### Model

``` text
16 / 16 model verification checks passed
```

### API

``` text
17 / 17 API tests passed
```

### Docker

The production Docker image was successfully built and locally tested.

### AWS

The container was successfully:

``` text
Built
    ->
Tagged
    ->
Pushed to ECR
    ->
Pulled by ECS
    ->
Started on Fargate
    ->
Exposed through FastAPI
```

### Frontend

The Vercel frontend was successfully connected to the deployed ECS API
and verified through the production API proxy.

------------------------------------------------------------------------

## 29. Future Scope

Potential future work includes:

1.  Evaluate alternative tectonic models such as updated plate-motion
    frameworks.
2.  Introduce additional seismic waveform-derived features.
3.  Incorporate focal mechanisms and moment tensors.
4.  Incorporate geological and geophysical raster datasets.
5.  Investigate spatial cross-validation strategies beyond PB2002
    boundary groups.
6.  Evaluate performance on independently sourced tectonic labels.
7.  Improve uncertainty estimation and probability calibration.
8.  Investigate geographically balanced training strategies.
9.  Evaluate temporal robustness on future earthquake observations.
10. Add an AWS Application Load Balancer for stable HTTPS deployment.
11. Add a custom DNS endpoint for the production API.
12. Introduce automated model monitoring and drift detection.
13. Expand interpretability from global SHAP analysis to local
    event-level explanations.
14. Compare XGBoost with neural and transformer-based geospatial
    architectures.

------------------------------------------------------------------------

## 30. References

The project is based on established scientific and machine learning
resources, including:

-   Bird, P., "An updated digital model of plate boundaries,"
    Geochemistry, Geophysics, Geosystems, 2003.
-   U.S. Geological Survey earthquake catalog and earthquake data
    services.
-   Chen, T. and Guestrin, C., "XGBoost: A Scalable Tree Boosting
    System," Proceedings of the 22nd ACM SIGKDD International Conference
    on Knowledge Discovery and Data Mining, 2016.
-   Lundberg, S. M. and Lee, S.-I., "A Unified Approach to Interpreting
    Model Predictions," Advances in Neural Information Processing
    Systems, 2017.
-   Akiba, T., et al., "Optuna: A Next-generation Hyperparameter
    Optimization Framework," Proceedings of the 25th ACM SIGKDD
    International Conference on Knowledge Discovery and Data Mining,
    2019.
-   Ke, G., et al., "LightGBM: A Highly Efficient Gradient Boosting
    Decision Tree," Advances in Neural Information Processing Systems,
    2017.
-   Prokhorenkova, L., et al., "CatBoost: Unbiased Boosting with
    Categorical Features," Advances in Neural Information Processing
    Systems, 2018.

Full IEEE-formatted references are provided in the project report.

------------------------------------------------------------------------

## 31. Project Status

Current project status:

``` text
Data Engineering                  COMPLETE
Feature Engineering               COMPLETE
Baseline Benchmarking             COMPLETE
Advanced Model Benchmarking       COMPLETE
Hyperparameter Optimization       COMPLETE
Scientific Evaluation             COMPLETE
SHAP Explainability               COMPLETE
Boundary-Aware Evaluation         COMPLETE
Final Model Training              COMPLETE
Model Verification                16/16 PASSED
FastAPI Development                COMPLETE
API Testing                        17/17 PASSED
Dockerization                      COMPLETE
Amazon ECR Deployment              COMPLETE
Amazon ECS Fargate Deployment      COMPLETE
Frontend Deployment                COMPLETE
End-to-End Verification            COMPLETE
```

------------------------------------------------------------------------

## 32. Deployment Links

### Production Application

https://tectonicsense.vercel.app

### Production API

http://54.225.30.119:8000

### Swagger Documentation

http://54.225.30.119:8000/docs

### ReDoc

http://54.225.30.119:8000/redoc

### API Health

http://54.225.30.119:8000/health

### Model Information

http://54.225.30.119:8000/model-info

------------------------------------------------------------------------

## 33. Author and Guidance

### Project Author

**Nithish V**\
M.Sc. Artificial Intelligence and Machine Learning\
Department of Computing (AI&ML)\
Coimbatore Institute of Technology\
Coimbatore, India

### Project Guidance

**Ms. P. Hemashree**\
Associate Professor\
Department of Computing (AI&ML)\
Coimbatore Institute of Technology\
Coimbatore, India

------------------------------------------------------------------------

## 34. License and Academic Use

This repository is developed as an academic machine learning project.

The datasets and external resources used by the project remain subject
to their respective licenses, terms of use, and attribution
requirements.

The PB2002 geological model and USGS earthquake data should be cited
according to their original sources when the project or derived results
are used in academic work.

------------------------------------------------------------------------

## 35. Citation

If you use TectonicSense in academic work, please cite the associated
project report and the original data/model sources.

**TectonicSense: Explainable Machine Learning for Tectonic Plate
Boundary Classification Using Seismic and Geospatial Features**

Nithish V, Department of Computing (AI&ML), Coimbatore Institute of
Technology, Coimbatore, India.

------------------------------------------------------------------------

## 36. Final Note

TectonicSense demonstrates a complete machine learning lifecycle:

``` text
Scientific Problem
        |
        v
Data Acquisition
        |
        v
Data Cleaning
        |
        v
Spatial Association
        |
        v
Feature Engineering
        |
        v
Model Benchmarking
        |
        v
Hyperparameter Optimization
        |
        v
Explainability
        |
        v
Boundary-Aware Validation
        |
        v
Final Model
        |
        v
FastAPI
        |
        v
Docker
        |
        v
Amazon ECR
        |
        v
Amazon ECS Fargate
        |
        v
Production Web Interface
```

The project therefore covers not only model development, but also
scientific validation, explainability, reproducibility, API engineering,
containerization, cloud deployment, and end-to-end inference.
