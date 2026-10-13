# TectonicSense

**Explainable Machine Learning for Tectonic Plate Boundary Classification Using Seismic and Geospatial Features**

TectonicSense is an Advanced Machine Learning Lab project that investigates whether seismic, spatial, and tectonic features can be used to classify earthquake-associated tectonic environments.

## Project Status

Phase 0 — Project structure initialized.

## Planned Pipeline

USGS Earthquake Data + PB2002 Plate Boundary Data
→ Spatial Dataset Construction
→ Feature Engineering
→ Exploratory Data Analysis
→ Baseline Models
→ Advanced Ensemble Models
→ Optuna Optimization
→ Ablation Study
→ SHAP Explainability
→ FastAPI + Streamlit
→ Docker
→ Cloud Deployment

## Repository Structure

```text
TectonicSense/
├── data/
├── notebooks/
├── src/
├── models/
├── results/
├── configs/
├── api/
├── app/
├── tests/
├── docs/
├── requirements.txt
├── Dockerfile
└── README.md
```

## Important Scientific Scope

This project is intended to classify tectonic environments associated with seismic events. It does **not** claim to predict future earthquakes.

## Data Sources

Planned primary sources:
- USGS earthquake catalog / ComCat
- PB2002 global plate-boundary model

Dataset details will be documented after source inspection.

## Development Rule

Each major milestone will be:
1. Implemented
2. Tested
3. Documented
4. Committed to Git
5. Pushed to GitHub
