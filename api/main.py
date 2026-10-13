"""TectonicSense - Production FastAPI Backend

Phase 5A: FastAPI Production API
Provides high-performance, strictly-validated REST endpoints for PB2002 tectonic
plate boundary classification using the verified Phase 4 production model.

Endpoints:
  - GET  /health          Service health check
  - GET  /model-info      Safe model metadata and schema specification
  - POST /predict         Single-observation classification
  - POST /predict-batch   Batch classification (up to 1,000 records)
"""

import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

# Ensure project root is available on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.predict import get_predictor, TectonicPredictor
from api.schemas import (
    SeismicFeatureInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)

# Configure logging
logger = logging.getLogger("tectonicsense_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Global predictor reference
_predictor: Optional[TectonicPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for single model initialization on startup."""
    global _predictor
    try:
        logger.info("Initializing TectonicSense production predictor...")
        _predictor = get_predictor()
        logger.info(
            "Predictor loaded successfully: %s (%d features, %d classes)",
            _predictor.metadata.get("model_name", "Tuned XGBoost"),
            _predictor.expected_feature_count,
            len(_predictor.label_mapping),
        )
    except Exception as e:
        logger.error("Failed to load model during startup: %s", e, exc_info=True)
        _predictor = None
    yield
    logger.info("Shutting down TectonicSense API.")


app = FastAPI(
    title="TectonicSense API",
    description="Production inference API for PB2002 tectonic plate boundary regime classification.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

import math
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    """Sanitize validation errors ensuring no un-serializable NaN or Inf floats in response."""
    errors = []
    for err in exc.errors():
        err_copy = dict(err)
        raw_input = err_copy.get("input")
        if isinstance(raw_input, float) and (math.isnan(raw_input) or math.isinf(raw_input)):
            err_copy["input"] = str(raw_input)
        errors.append(err_copy)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": jsonable_encoder(errors)},
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Returns the operational status of the service and confirms the production ML model is loaded.",
)
def health_check():
    """Verify service and model readiness."""
    global _predictor
    if _predictor is None:
        try:
            _predictor = get_predictor()
        except Exception as e:
            logger.error("Health check failed to initialize predictor: %s", e)
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "degraded",
                    "service": "TectonicSense API",
                    "model_loaded": False,
                },
            )

    return HealthResponse(
        status="healthy",
        service="TectonicSense API",
        model_loaded=True,
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model Metadata & Specification",
    description="Provides safe architectural metadata, input feature count, framework version, and class labels.",
)
def get_model_info():
    """Expose safe model metadata without revealing internal filesystem paths."""
    global _predictor
    if _predictor is None:
        _predictor = get_predictor()

    meta = _predictor.metadata
    return ModelInfoResponse(
        model_name=meta.get("model_name", "Tuned XGBoost"),
        estimator=meta.get("estimator", "XGBClassifier"),
        framework=meta.get("framework", "xgboost"),
        framework_version=meta.get("xgboost_version", "3.0.5"),
        feature_count=_predictor.expected_feature_count,
        class_count=len(_predictor.label_mapping),
        class_names=list(_predictor.label_mapping.values()),
        model_version=meta.get("model_version", "1.0.0-final"),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Single Observation Prediction",
    description="Accepts exactly 26 seismic and geospatial features and returns the predicted PB2002 class and probability distribution.",
)
def predict_single_observation(payload: SeismicFeatureInput):
    """Run inference on a single observation."""
    global _predictor
    if _predictor is None:
        _predictor = get_predictor()

    try:
        data_dict = payload.model_dump()
        result = _predictor.predict_single(data_dict)

        return PredictionResponse(
            predicted_class=result["predicted_class"],
            class_id=result.get("class_id", result.get("predicted_class_id")),
            confidence=result["confidence"],
            class_probabilities=result["class_probabilities"],
        )
    except ValueError as e:
        logger.warning("Input validation error during single prediction: %s", e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Internal error during single prediction: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed due to an internal server error.",
        )


@app.post(
    "/predict-batch",
    response_model=BatchPredictionResponse,
    summary="Batch Observations Prediction",
    description="Accepts a list of observations (up to 1,000 items) and returns ordered prediction records.",
)
def predict_batch_observations(payload: BatchPredictionRequest):
    """Run inference on multiple observations while preserving input order."""
    global _predictor
    if _predictor is None:
        _predictor = get_predictor()

    try:
        observations_data = [obs.model_dump() for obs in payload.observations]
        batch_results = _predictor.predict_batch(observations_data)

        predictions = [
            PredictionResponse(
                predicted_class=r["predicted_class"],
                class_id=r.get("class_id", r.get("predicted_class_id")),
                confidence=r["confidence"],
                class_probabilities=r["class_probabilities"],
            )
            for r in batch_results
        ]

        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions),
        )
    except ValueError as e:
        logger.warning("Validation error during batch prediction: %s", e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Internal error during batch prediction: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed due to an internal server error.",
        )
