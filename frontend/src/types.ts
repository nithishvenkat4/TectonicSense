export type PB2002Class = 'CCB' | 'CRB' | 'CTF' | 'OCB' | 'OSR' | 'OTF' | 'SUB';

export interface RawObservationInput {
  magnitude: number;
  depth_km: number;
  latitude: number;
  longitude: number;
  boundary_distance_km: number;
  boundary_length_km: number;
  boundary_azimuth_deg: number;
  relative_velocity: number;
  velocity_azimuth_deg: number;
  divergent_component: number;
  right_lateral_component: number;
  boundary_elevation_m: number;
  seafloor_age_ma: number;
  orogen_flag: number;
}

export interface Production26Features {
  magnitude: number;
  magnitude_squared: number;
  depth_km: number;
  depth_abs_km: number;
  longitude: number;
  latitude: number;
  longitude_sin: number;
  longitude_cos: number;
  latitude_sin: number;
  latitude_cos: number;
  boundary_distance_km: number;
  log_boundary_distance_km: number;
  boundary_length_km: number;
  boundary_azimuth_sin: number;
  boundary_azimuth_cos: number;
  relative_velocity: number;
  velocity_azimuth_sin: number;
  velocity_azimuth_cos: number;
  divergent_component: number;
  right_lateral_component: number;
  absolute_divergent_component: number;
  absolute_right_lateral_component: number;
  tectonic_motion_magnitude: number;
  boundary_elevation_m: number;
  seafloor_age_ma: number;
  orogen_flag: number;
}

export interface PredictionResponse {
  predicted_class: PB2002Class;
  class_id: number;
  confidence: number;
  class_probabilities: Record<PB2002Class, number>;
}

export interface HealthResponse {
  status: string;
  service: string;
  model_loaded: boolean;
}

export interface ModelInfoResponse {
  model_name: string;
  estimator: string;
  framework: string;
  framework_version: string;
  feature_count: number;
  class_count: number;
  class_names: PB2002Class[];
  model_version: string;
}

export interface ObservationPreset {
  name: string;
  location: string;
  class: PB2002Class;
  data: RawObservationInput;
}
