# TectonicSense — Phase 2 Consolidated Model Comparison

Evaluated strictly on the validation set (`data/processed/splits/validation.csv`). Test set held out.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Training Time (s) |
|:------|:---------|:----------------|:-------------|:---------|:------------|:------------------|
| Tuned XGBoost | 0.9999 | 0.9998 | 0.9998 | 0.9998 | 0.9999 | 19.00 |
| Tuned LightGBM | 0.9998 | 0.9997 | 0.9997 | 0.9997 | 0.9998 | 19.35 |
| HistGradientBoosting | 0.9998 | 0.9997 | 0.9997 | 0.9997 | 0.9998 | 30.19 |
| LightGBM | 0.9998 | 0.9996 | 0.9996 | 0.9996 | 0.9998 | 19.54 |
| XGBoost | 0.9998 | 0.9997 | 0.9996 | 0.9996 | 0.9998 | 21.25 |
| RandomForest | 0.9998 | 0.9997 | 0.9996 | 0.9996 | 0.9998 | 22.70 |
| CatBoost | 0.9983 | 0.9974 | 0.9969 | 0.9972 | 0.9983 | 43.23 |
| LogisticRegression | 0.9410 | 0.9060 | 0.9020 | 0.9036 | 0.9407 | 21.92 |
| DummyClassifier | 0.4811 | 0.0687 | 0.1429 | 0.0928 | 0.3126 | 0.10 |
