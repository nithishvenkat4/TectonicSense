# Methodology

## Planned Workflow

1. Acquire and inspect source datasets.
2. Clean and validate seismic and tectonic data.
3. Perform spatial association between earthquake events and plate-boundary segments.
4. Construct target labels.
5. Engineer seismic, spatial, tectonic, temporal, and—if sufficiently available—focal-mechanism features.
6. Perform exploratory data analysis.
7. Train baseline models.
8. Train advanced ensemble models.
9. Perform hyperparameter optimization.
10. Conduct feature-group ablation experiments.
11. Apply SHAP-based explainability and error analysis.
12. Package the final inference pipeline.
13. Deploy using FastAPI, Streamlit, Docker, and a cloud platform.

## Planned Model Progression

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

## Planned Explainability

- SHAP global feature importance
- SHAP local explanations
- Error analysis

## Planned Ablation

- Seismic features only
- Seismic + spatial features
- Seismic + spatial + tectonic features
- Full feature set
