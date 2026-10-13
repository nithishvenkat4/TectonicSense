"""TectonicSense - API Pydantic Schemas

Defines request and response schemas for FastAPI endpoints.
Strictly enforces the 26-feature input schema, numeric data types,
rejection of extraneous/leakage fields, and finite values (no NaN/Inf).
"""

import math
from typing import Dict, List
from pydantic import BaseModel, Field, ConfigDict, field_validator


class SeismicFeatureInput(BaseModel):
    """Input payload representing exactly the 26 required seismic and geospatial features.
    
    Target labels and discrete segment IDs (boundary_id, boundary_step_id, boundary_class)
    are strictly forbidden.
    """
    model_config = ConfigDict(extra="forbid")

    magnitude: float = Field(..., description="Earthquake magnitude (Mw/Mb)")
    magnitude_squared: float = Field(..., description="Squared earthquake magnitude")
    depth_km: float = Field(..., description="Hypocentral depth in kilometers")
    depth_abs_km: float = Field(..., description="Absolute hypocentral depth in kilometers")
    longitude: float = Field(..., description="Epicenter longitude (-180 to 180 degrees)")
    latitude: float = Field(..., description="Epicenter latitude (-90 to 90 degrees)")
    longitude_sin: float = Field(..., description="Cyclical sine encoding of longitude")
    longitude_cos: float = Field(..., description="Cyclical cosine encoding of longitude")
    latitude_sin: float = Field(..., description="Cyclical sine encoding of latitude")
    latitude_cos: float = Field(..., description="Cyclical cosine encoding of latitude")
    boundary_distance_km: float = Field(..., description="Geodesic distance to nearest PB2002 boundary segment (km)")
    log_boundary_distance_km: float = Field(..., description="Natural log distance ln(1 + distance_km)")
    boundary_length_km: float = Field(..., description="Length of nearest PB2002 boundary segment (km)")
    boundary_azimuth_sin: float = Field(..., description="Sine of boundary strike azimuth")
    boundary_azimuth_cos: float = Field(..., description="Cosine of boundary strike azimuth")
    relative_velocity: float = Field(..., description="Relative plate velocity (mm/yr)")
    velocity_azimuth_sin: float = Field(..., description="Sine of velocity azimuth vector")
    velocity_azimuth_cos: float = Field(..., description="Cosine of velocity azimuth vector")
    divergent_component: float = Field(..., description="Normal velocity component (+ = spreading, - = convergence)")
    right_lateral_component: float = Field(..., description="Strike-slip velocity component (+ = right-lateral, - = left-lateral)")
    absolute_divergent_component: float = Field(..., description="Absolute normal velocity magnitude (mm/yr)")
    absolute_right_lateral_component: float = Field(..., description="Absolute strike-slip velocity magnitude (mm/yr)")
    tectonic_motion_magnitude: float = Field(..., description="Euclidean magnitude of tectonic velocity vector (mm/yr)")
    boundary_elevation_m: float = Field(..., description="Surface elevation or trench bathymetry at boundary (m)")
    seafloor_age_ma: float = Field(..., description="Oceanic crustal age in million years (Ma)")
    orogen_flag: float = Field(..., description="PB2002 orogen indicator (1.0 = orogen, 0.0 = non-orogen)")

    @field_validator("*", mode="after")
    @classmethod
    def validate_finite_numeric(cls, v: float) -> float:
        """Validate that all features are finite numeric values (no NaN, inf, or -inf)."""
        if isinstance(v, (int, float)) and (math.isnan(v) or math.isinf(v)):
            raise ValueError("Numeric value must be finite; NaN, inf, and -inf are strictly forbidden.")
        return v


class PredictionResponse(BaseModel):
    """Single observation prediction output."""
    predicted_class: str = Field(..., description="Predicted PB2002 tectonic boundary class (e.g. SUB, OSR, CTF)")
    class_id: int = Field(..., description="Integer class identifier (0 to 6)")
    confidence: float = Field(..., description="Estimated probability of predicted class (0.0 to 1.0)")
    class_probabilities: Dict[str, float] = Field(..., description="Calibrated probability distribution across all 7 PB2002 classes")


class BatchPredictionRequest(BaseModel):
    """Batch prediction input payload with strict size limiting."""
    observations: List[SeismicFeatureInput] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of observations to classify (1 to 1000 items)",
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction output containing ordered predictions."""
    predictions: List[PredictionResponse] = Field(..., description="Ordered list of prediction results")
    count: int = Field(..., description="Total number of evaluated observations")


class HealthResponse(BaseModel):
    """Health check status output."""
    status: str = Field(..., description="Operational status: healthy or degraded")
    service: str = Field(..., description="Service identifier name")
    model_loaded: bool = Field(..., description="Flag indicating model binary is initialized")


class ModelInfoResponse(BaseModel):
    """Exposes safe model metadata and schema specification."""
    model_name: str = Field(..., description="Model architecture name")
    estimator: str = Field(..., description="Estimator class name")
    framework: str = Field(..., description="Underlying machine learning framework")
    framework_version: str = Field(..., description="Framework version")
    feature_count: int = Field(..., description="Total number of required features")
    class_count: int = Field(..., description="Total number of target classes")
    class_names: List[str] = Field(..., description="List of recognized PB2002 class names")
    model_version: str = Field(..., description="Model artifact release version")
