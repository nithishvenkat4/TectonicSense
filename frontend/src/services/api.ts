import type {
  RawObservationInput,
  Production26Features,
  PredictionResponse,
  HealthResponse,
  ModelInfoResponse,
} from '../types';

const API_BASE_URL =
  (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_URL) ||
  'http://127.0.0.1:8000';

/**
 * Deterministically derives the exact 26 production features from user primary inputs
 * using the project's canonical mathematical transformations.
 */
export function deriveFeatures(raw: RawObservationInput): Production26Features {
  const lonRad = (raw.longitude * Math.PI) / 180;
  const latRad = (raw.latitude * Math.PI) / 180;
  const bAzRad = (raw.boundary_azimuth_deg * Math.PI) / 180;
  const vAzRad = (raw.velocity_azimuth_deg * Math.PI) / 180;

  return {
    magnitude: raw.magnitude,
    magnitude_squared: Number((raw.magnitude ** 2).toFixed(6)),
    depth_km: raw.depth_km,
    depth_abs_km: Math.abs(raw.depth_km),
    longitude: raw.longitude,
    latitude: raw.latitude,
    longitude_sin: Number(Math.sin(lonRad).toFixed(6)),
    longitude_cos: Number(Math.cos(lonRad).toFixed(6)),
    latitude_sin: Number(Math.sin(latRad).toFixed(6)),
    latitude_cos: Number(Math.cos(latRad).toFixed(6)),
    boundary_distance_km: raw.boundary_distance_km,
    log_boundary_distance_km: Number(Math.log1p(Math.max(0, raw.boundary_distance_km)).toFixed(6)),
    boundary_length_km: raw.boundary_length_km,
    boundary_azimuth_sin: Number(Math.sin(bAzRad).toFixed(6)),
    boundary_azimuth_cos: Number(Math.cos(bAzRad).toFixed(6)),
    relative_velocity: raw.relative_velocity,
    velocity_azimuth_sin: Number(Math.sin(vAzRad).toFixed(6)),
    velocity_azimuth_cos: Number(Math.cos(vAzRad).toFixed(6)),
    divergent_component: raw.divergent_component,
    right_lateral_component: raw.right_lateral_component,
    absolute_divergent_component: Math.abs(raw.divergent_component),
    absolute_right_lateral_component: Math.abs(raw.right_lateral_component),
    tectonic_motion_magnitude: Number(Math.hypot(raw.divergent_component, raw.right_lateral_component).toFixed(6)),
    boundary_elevation_m: raw.boundary_elevation_m,
    seafloor_age_ma: raw.seafloor_age_ma,
    orogen_flag: Number(raw.orogen_flag),
  };
}

/**
 * Service health check.
 */
export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/health`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });
  if (!res.ok) {
    throw new Error(`Health check returned HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Model specification & safe metadata.
 */
export async function getModelInfo(): Promise<ModelInfoResponse> {
  const res = await fetch(`${API_BASE_URL}/model-info`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });
  if (!res.ok) {
    throw new Error(`Model info returned HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Execute single observation inference against FastAPI /predict.
 */
export async function predictSingle(features: Production26Features): Promise<PredictionResponse> {
  const res = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(features),
  });

  if (!res.ok) {
    let errorDetail = `Prediction failed with HTTP ${res.status}`;
    try {
      const errorJson = await res.json();
      if (errorJson.detail) {
        errorDetail = typeof errorJson.detail === 'string'
          ? errorJson.detail
          : JSON.stringify(errorJson.detail);
      }
    } catch {
      // ignore json parse errors
    }
    throw new Error(errorDetail);
  }

  return res.json();
}
