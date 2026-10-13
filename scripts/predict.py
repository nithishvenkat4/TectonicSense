"""TectonicSense - Production Inference Engine

Phase 4: Final Model & Productionization
Provides a unified, reusable inference pipeline for single and batch predictions.
Used directly by API (FastAPI), interactive UI (Streamlit), and batch services.

Artifacts Loaded:
  - models/final/tectonicsense_xgboost.joblib
  - models/final/feature_names.json
  - models/final/label_mapping.json
  - models/final/metadata.json
"""

from pathlib import Path
import json
from typing import Dict, List, Union, Any
import numpy as np
import pandas as pd
import joblib

# Deterministic project root resolution
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "final" / "tectonicsense_xgboost.joblib"
DEFAULT_FEATURES_PATH = ROOT_DIR / "models" / "final" / "feature_names.json"
DEFAULT_LABELS_PATH = ROOT_DIR / "models" / "final" / "label_mapping.json"
DEFAULT_METADATA_PATH = ROOT_DIR / "models" / "final" / "metadata.json"
TEST_DATA_PATH = ROOT_DIR / "data" / "processed" / "splits" / "test.csv"


class TectonicPredictor:
    """Production inference engine for TectonicSense PB2002 plate boundary classification."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        features_path: Path = DEFAULT_FEATURES_PATH,
        labels_path: Path = DEFAULT_LABELS_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH,
    ):
        self.model_path = Path(model_path)
        self.features_path = Path(features_path)
        self.labels_path = Path(labels_path)
        self.metadata_path = Path(metadata_path)

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load model binary, feature schema, label mappings, and metadata."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model binary not found at: {self.model_path}")
        if not self.features_path.exists():
            raise FileNotFoundError(f"Feature schema not found at: {self.features_path}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Label mapping not found at: {self.labels_path}")

        # 1. Load model
        self.model = joblib.load(self.model_path)

        # 2. Load feature names
        with open(self.features_path, "r", encoding="utf-8") as f:
            feat_data = json.load(f)
            self.feature_names: List[str] = feat_data["features"]
            self.expected_feature_count: int = feat_data.get("feature_count", len(self.feature_names))

        # 3. Load label mapping (mapping string integer keys to class names)
        with open(self.labels_path, "r", encoding="utf-8") as f:
            raw_labels = json.load(f)
            self.label_mapping: Dict[int, str] = {int(k): str(v) for k, v in raw_labels.items()}
            self.inverse_label_mapping: Dict[str, int] = {v: k for k, v in self.label_mapping.items()}

        # 4. Load metadata if available
        self.metadata: Dict[str, Any] = {}
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def validate_and_format_input(self, input_data: Union[Dict[str, Any], pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Validate input schema, data types, nulls, and ensure exact feature ordering."""
        # Normalize input to DataFrame
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.Series):
            df = pd.DataFrame([input_data.to_dict()])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise TypeError(
                f"Unsupported input type: {type(input_data)}. Expected dict, pd.Series, or pd.DataFrame."
            )

        if df.empty:
            raise ValueError("Input dataset is empty.")

        # Check for missing features
        input_cols = set(df.columns)
        expected_cols = set(self.feature_names)
        missing_features = [f for f in self.feature_names if f not in input_cols]
        if missing_features:
            raise ValueError(
                f"Missing {len(missing_features)} required feature(s): {missing_features}"
            )

        # Extract only the 26 required features in exact established order
        df_ordered = df[self.feature_names].copy()

        # Check for non-numeric types and convert
        for col in self.feature_names:
            try:
                df_ordered[col] = pd.to_numeric(df_ordered[col], errors="raise")
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Non-numeric value encountered in feature '{col}': {e}"
                ) from e

        # Validate no NaN values
        nan_counts = df_ordered.isna().sum().sum()
        if nan_counts > 0:
            nan_cols = df_ordered.columns[df_ordered.isna().any()].tolist()
            raise ValueError(
                f"NaN values detected in input features: {nan_cols} (total NaNs: {nan_counts})"
            )

        # Validate no infinite values
        inf_mask = np.isinf(df_ordered.values)
        if np.any(inf_mask):
            inf_cols = [self.feature_names[j] for j in np.where(inf_mask)[1]]
            raise ValueError(
                f"Infinite (inf/-inf) values detected in input features: {set(inf_cols)}"
            )

        # Confirm exact feature count
        if df_ordered.shape[1] != self.expected_feature_count:
            raise ValueError(
                f"Feature count mismatch: Expected {self.expected_feature_count}, got {df_ordered.shape[1]}"
            )

        return df_ordered

    def predict_batch(
        self,
        input_data: Union[pd.DataFrame, List[Dict[str, Any]]],
        include_probabilities: bool = True
    ) -> List[Dict[str, Any]]:
        """Run batch inference on multiple observations while preserving input order."""
        if isinstance(input_data, list):
            input_df = pd.DataFrame(input_data)
        else:
            input_df = input_data

        df_validated = self.validate_and_format_input(input_df)

        # Model inference (probabilities)
        probs_matrix = self.model.predict_proba(df_validated)
        pred_indices = np.argmax(probs_matrix, axis=1)

        results: List[Dict[str, Any]] = []
        n_classes = len(self.label_mapping)

        for i in range(len(df_validated)):
            pred_idx = int(pred_indices[i])
            pred_class = self.label_mapping[pred_idx]
            # Exact confidence = probability of predicted class
            confidence = float(probs_matrix[i, pred_idx])

            record: Dict[str, Any] = {
                "predicted_class": pred_class,
                "predicted_class_id": pred_idx,
                "confidence": confidence,
            }

            if include_probabilities:
                record["class_probabilities"] = {
                    self.label_mapping[c_idx]: float(probs_matrix[i, c_idx])
                    for c_idx in range(n_classes)
                }

            results.append(record)

        return results

    def predict_single(
        self,
        features: Union[Dict[str, Any], pd.Series, pd.DataFrame],
        include_probabilities: bool = True
    ) -> Dict[str, Any]:
        """Run inference on a single observation. Returns structured prediction dictionary."""
        batch_results = self.predict_batch(features, include_probabilities=include_probabilities)
        return batch_results[0]


# Module-level singleton instance for fast reuse in API / UI
_PREDICTOR_INSTANCE = None


def get_predictor() -> TectonicPredictor:
    """Return a singleton TectonicPredictor instance."""
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = TectonicPredictor()
    return _PREDICTOR_INSTANCE


def predict_single(features: Union[Dict[str, Any], pd.Series, pd.DataFrame]) -> Dict[str, Any]:
    """Convenience function for single record prediction."""
    return get_predictor().predict_single(features)


def predict_batch(dataframe: pd.DataFrame, include_probabilities: bool = True) -> List[Dict[str, Any]]:
    """Convenience function for batch record prediction."""
    return get_predictor().predict_batch(dataframe, include_probabilities=include_probabilities)


def main():
    """CLI self-test demonstrating model loading, input validation, and single/batch inference."""
    print("=" * 70)
    print("TECTONICSENSE PRODUCTION INFERENCE ENGINE (CLI SELF-TEST)")
    print("=" * 70)

    predictor = get_predictor()
    print(f"Model loaded:            {predictor.metadata.get('model_name', 'XGBoost')}")
    print(f"Framework version:       XGBoost {predictor.metadata.get('xgboost_version', '3.x')}")
    print(f"Expected feature count:  {predictor.expected_feature_count}")
    print(f"Classes ({len(predictor.label_mapping)}):         {list(predictor.label_mapping.values())}")

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Test split not found at: {TEST_DATA_PATH}")

    # Load 1 valid test observation
    test_df = pd.read_csv(TEST_DATA_PATH, nrows=5)
    sample_series = test_df.iloc[0].drop("boundary_class")
    true_label = test_df.iloc[0]["boundary_class"]

    print("\n--- Single Observation Inference Test ---")
    print(f"Sample test row 1 (True Class: {true_label})")
    
    result = predictor.predict_single(sample_series)
    
    print(f"\nPrediction:  {result['predicted_class']} (Class ID: {result['predicted_class_id']})")
    print(f"Confidence:  {result['confidence']:.6f}")
    print("\nClass Probabilities:")
    prob_sum = 0.0
    for cls_name, prob in result["class_probabilities"].items():
        print(f"  {cls_name}: {prob:.6f}")
        prob_sum += prob

    print(f"\nSum of probabilities: {prob_sum:.6f} (Normalized: {np.isclose(prob_sum, 1.0)})")
    assert np.isclose(prob_sum, 1.0, atol=1e-5), "Probabilities do not sum to 1.0!"
    assert result["confidence"] == result["class_probabilities"][result["predicted_class"]], "Confidence mismatch!"

    print("\n--- Batch Inference Test (5 rows) ---")
    batch_features = test_df.drop(columns=["boundary_class"])
    batch_results = predictor.predict_batch(batch_features)
    
    for i, res in enumerate(batch_results):
        true_cls = test_df.iloc[i]["boundary_class"]
        print(f"Row {i+1}: True={true_cls:<4} | Predicted={res['predicted_class']:<4} | Conf={res['confidence']:.4f} | Match={res['predicted_class'] == true_cls}")

    print("\n" + "=" * 70)
    print("INFERENCE PIPELINE SELF-TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
