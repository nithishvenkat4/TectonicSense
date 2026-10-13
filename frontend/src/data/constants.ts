import type { PB2002Class } from '../types';

export interface ClassMetadata {
  code: PB2002Class;
  name: string;
  type: string;
  description: string;
  color: string;
}

export const PB2002_CLASSES: Record<PB2002Class, ClassMetadata> = {
  SUB: {
    code: 'SUB',
    name: 'Subduction Zone',
    type: 'Convergent (Oceanic-Continental / Oceanic-Oceanic)',
    description: 'One tectonic plate descends beneath an overriding plate into the mantle, accompanied by deep ocean trenches, arc volcanism, and deep Wadati-Benioff seismicity.',
    color: '#b91c1c', // deep terracotta red
  },
  OSR: {
    code: 'OSR',
    name: 'Oceanic Spreading Ridge',
    type: 'Divergent (Oceanic)',
    description: 'Tectonic plates pull apart at oceanic spreading centers where upwelling magma forms new oceanic lithosphere with shallow hydrothermal activity.',
    color: '#0284c7', // oceanic cyan/blue
  },
  OTF: {
    code: 'OTF',
    name: 'Oceanic Transform Fault',
    type: 'Transform (Strike-Slip Oceanic)',
    description: 'Horizontal strike-slip faulting offsetting oceanic ridge segments along conservative plate boundaries.',
    color: '#d97706', // amber/orange
  },
  CRB: {
    code: 'CRB',
    name: 'Continental Rift Boundary',
    type: 'Divergent (Continental)',
    description: 'Incipient extensional boundary where a continental landmass is stretching and rifting apart (e.g. East African Rift).',
    color: '#059669', // green
  },
  CTF: {
    code: 'CTF',
    name: 'Continental Transform Fault',
    type: 'Transform (Strike-Slip Continental)',
    description: 'Strike-slip motion accommodating lateral displacement between continental plates (e.g. San Andreas Fault, North Anatolian Fault).',
    color: '#ea580c', // rust orange
  },
  CCB: {
    code: 'CCB',
    name: 'Continental Collision Boundary',
    type: 'Convergent (Continent-Continent)',
    description: 'Convergence of two continental landmasses producing intense crustal shortening, mountain building (orogeny), and shallow thrust faulting (e.g. Himalayas).',
    color: '#4f46e5', // indigo
  },
  OCB: {
    code: 'OCB',
    name: 'Oceanic Collision Boundary',
    type: 'Convergent (Diffuse Oceanic)',
    description: 'Diffuse compressive zone between oceanic plates where lithospheric bucking and non-subducting convergence occur.',
    color: '#7c3aed', // purple
  },
};

export interface ShapFeature {
  feature: string;
  label: string;
  importance: number;
  description: string;
}

export const GLOBAL_SHAP_FEATURES: ShapFeature[] = [
  {
    feature: 'divergent_component',
    label: 'Divergent Component',
    importance: 2.0318,
    description: 'Distinguishes normal plate spreading from convergent subduction/collision motions.',
  },
  {
    feature: 'boundary_elevation_m',
    label: 'Boundary Elevation / Bathymetry',
    importance: 1.4451,
    description: 'Separates deep subduction ocean trenches (-6km to -11km) from shallow ridges and continental plateaus.',
  },
  {
    feature: 'absolute_divergent_component',
    label: 'Absolute Divergence Rate',
    importance: 0.8775,
    description: 'Magnitude of normal displacement perpendicular to the fault interface.',
  },
  {
    feature: 'seafloor_age_ma',
    label: 'Seafloor Crustal Age',
    importance: 0.4082,
    description: 'Youngest along mid-ocean ridges (0 Ma); oldest along ancient subducting oceanic margins and continental margins.',
  },
  {
    feature: 'absolute_right_lateral_component',
    label: 'Absolute Strike-Slip Rate',
    importance: 0.3891,
    description: 'Dominant kinetic discriminator for transform faulting (OTF and CTF).',
  },
  {
    feature: 'relative_velocity',
    label: 'Relative Plate Velocity',
    importance: 0.2626,
    description: 'Overall relative plate displacement speed across the boundary segment.',
  },
  {
    feature: 'tectonic_motion_magnitude',
    label: 'Tectonic Motion Vector Magnitude',
    importance: 0.2201,
    description: 'Total Euclidean magnitude combining divergent and transform velocity vectors.',
  },
];

export interface ModelBenchmark {
  model: string;
  macroF1: number;
  accuracy: number;
  trainingTimeSec: number;
  status: 'Final Production' | 'Candidate' | 'Baseline';
}

export const MODEL_BENCHMARKS: ModelBenchmark[] = [
  { model: 'Tuned XGBoost', macroF1: 0.9998, accuracy: 0.9999, trainingTimeSec: 19.0, status: 'Final Production' },
  { model: 'Tuned LightGBM', macroF1: 0.9997, accuracy: 0.9998, trainingTimeSec: 19.35, status: 'Candidate' },
  { model: 'HistGradientBoosting', macroF1: 0.9997, accuracy: 0.9998, trainingTimeSec: 30.19, status: 'Candidate' },
  { model: 'LightGBM (Default)', macroF1: 0.9996, accuracy: 0.9998, trainingTimeSec: 19.54, status: 'Candidate' },
  { model: 'XGBoost (Default)', macroF1: 0.9996, accuracy: 0.9998, trainingTimeSec: 21.25, status: 'Candidate' },
  { model: 'Random Forest', macroF1: 0.9996, accuracy: 0.9998, trainingTimeSec: 22.70, status: 'Candidate' },
  { model: 'CatBoost', macroF1: 0.9972, accuracy: 0.9983, trainingTimeSec: 43.23, status: 'Candidate' },
  { model: 'Logistic Regression', macroF1: 0.9036, accuracy: 0.9410, trainingTimeSec: 21.92, status: 'Baseline' },
  { model: 'Dummy Classifier', macroF1: 0.0928, accuracy: 0.4811, trainingTimeSec: 0.10, status: 'Baseline' },
];

export const GENERALIZATION_BENCHMARKS = {
  randomStratified: {
    macroF1: 0.9998,
    accuracy: 0.9999,
    testSamples: 35964,
    errors: 4,
    description: 'Standard random split across all 239k global earthquakes; provides in-boundary interpolation benchmark.',
  },
  boundaryAware: {
    macroF1: 0.7935,
    accuracy: 0.7994,
    testBoundaries: 28,
    trainBoundaries: 127,
    boundaryOverlap: 0,
    description: 'Group-disjoint evaluation isolating complete PB2002 boundary entities; tests true out-of-boundary transfer on unfamiliar plate segments.',
  },
};
