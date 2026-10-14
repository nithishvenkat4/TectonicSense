import React from 'react';
import type { RawObservationInput, PB2002Class } from '../types';
import { PRESETS } from '../data/presets';
import { PB2002_CLASSES } from '../data/constants';
import { Compass, Play, RotateCcw } from 'lucide-react';

interface ObservationFormProps {
  values: RawObservationInput;
  onChange: (values: RawObservationInput) => void;
  onSubmit: () => void;
  isLoading: boolean;
  onReset: () => void;
  predictedClass?: PB2002Class | null;
}

export const ObservationForm: React.FC<ObservationFormProps> = ({
  values,
  onChange,
  onSubmit,
  isLoading,
  onReset,
  predictedClass,
}) => {
  const handleInputChange = (field: keyof RawObservationInput, val: string | number) => {
    const num = typeof val === 'number' ? val : parseFloat(val);
    onChange({
      ...values,
      [field]: isNaN(num) ? 0 : num,
    });
  };

  const handlePresetSelect = (presetName: string) => {
    const found = PRESETS.find((p) => p.name === presetName);
    if (found) {
      onChange({ ...found.data });
    }
  };

  return (
    <form
      id="observation-form"
      className="observation-form-container"
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit();
      }}
    >
      <div className="section-header">
        <div className="section-title-wrap">
          <Compass size={16} className="section-icon" />
          <h2 className="section-title">Observation Parameters</h2>
        </div>
      </div>

      {/* Preset Selector */}
      <div className="preset-selector-row">
        <label htmlFor="preset-select" className="preset-label">
          Reference Geological Event
        </label>
        <div className="preset-input-group">
          <select
            id="preset-select"
            className="preset-select"
            onChange={(e) => handlePresetSelect(e.target.value)}
            defaultValue={PRESETS[0].name}
          >
            {PRESETS.map((p) => (
              <option key={p.name} value={p.name}>
                [{p.class}] {p.name} — {p.location}
              </option>
            ))}
          </select>
          <button
            id="btn-classify-top"
            type="submit"
            disabled={isLoading}
            className="btn-primary btn-quick"
            title="Execute classification"
          >
            <Play size={13} />
            <span>{isLoading ? '...' : 'Classify'}</span>
          </button>
        </div>
      </div>

      <div className="form-grid">
        {/* Hypocenter Parameters */}
        <fieldset className="form-group-card">
          <legend className="group-legend">1. Hypocenter & Coordinates</legend>
          <div className="input-row">
            <div className="input-field">
              <label htmlFor="input-latitude">Latitude (°)</label>
              <input
                id="input-latitude"
                type="number"
                step="any"
                min="-90"
                max="90"
                value={values.latitude}
                onChange={(e) => handleInputChange('latitude', e.target.value)}
                required
              />
            </div>
            <div className="input-field">
              <label htmlFor="input-longitude">Longitude (°)</label>
              <input
                id="input-longitude"
                type="number"
                step="any"
                min="-180"
                max="180"
                value={values.longitude}
                onChange={(e) => handleInputChange('longitude', e.target.value)}
                required
              />
            </div>
          </div>
          <div className="input-row">
            <div className="input-field">
              <label htmlFor="input-magnitude">Magnitude (Mw)</label>
              <input
                id="input-magnitude"
                type="number"
                step="any"
                min="1.0"
                max="10.0"
                value={values.magnitude}
                onChange={(e) => handleInputChange('magnitude', e.target.value)}
                required
              />
            </div>
            <div className="input-field">
              <label htmlFor="input-depth">Focal Depth (km)</label>
              <input
                id="input-depth"
                type="number"
                step="any"
                min="0"
                max="750"
                value={values.depth_km}
                onChange={(e) => handleInputChange('depth_km', e.target.value)}
                required
              />
            </div>
          </div>
        </fieldset>

        {/* Boundary Geometry */}
        <fieldset className="form-group-card">
          <legend className="group-legend">2. PB2002 Boundary Geometry</legend>
          <div className="input-row">
            <div className="input-field">
              <label htmlFor="input-boundary-distance">Distance to Fault (km)</label>
              <input
                id="input-boundary-distance"
                type="number"
                step="any"
                min="0"
                max="5000"
                value={values.boundary_distance_km}
                onChange={(e) => handleInputChange('boundary_distance_km', e.target.value)}
                required
              />
            </div>
            <div className="input-field">
              <label htmlFor="input-boundary-length">Segment Length (km)</label>
              <input
                id="input-boundary-length"
                type="number"
                step="any"
                min="0"
                max="5000"
                value={values.boundary_length_km}
                onChange={(e) => handleInputChange('boundary_length_km', e.target.value)}
                required
              />
            </div>
          </div>
          <div className="input-row">
            <div className="input-field">
              <label htmlFor="input-boundary-azimuth">Fault Strike Azimuth (°)</label>
              <input
                id="input-boundary-azimuth"
                type="number"
                step="any"
                min="0"
                max="360"
                value={values.boundary_azimuth_deg}
                onChange={(e) => handleInputChange('boundary_azimuth_deg', e.target.value)}
                required
              />
            </div>
            <div className="input-field">
              <label htmlFor="input-elevation">Boundary Elevation / Depth (m)</label>
              <input
                id="input-elevation"
                type="number"
                step="any"
                min="-12000"
                max="9000"
                value={values.boundary_elevation_m}
                onChange={(e) => handleInputChange('boundary_elevation_m', e.target.value)}
                required
              />
            </div>
          </div>
        </fieldset>

        {/* Kinematics & Vector Components */}
        <fieldset className="form-group-card">
          <legend className="group-legend">3. Relative Plate Kinematics</legend>
          <div className="input-row">
            <div className="input-field">
              <label htmlFor="input-relative-velocity">Relative Velocity (mm/yr)</label>
              <input
                id="input-relative-velocity"
                type="number"
                step="any"
                min="0"
                max="250"
                value={values.relative_velocity}
                onChange={(e) => handleInputChange('relative_velocity', e.target.value)}
                required
              />
            </div>
            <div className="input-field">
              <label htmlFor="input-velocity-azimuth">Velocity Azimuth (°)</label>
              <input
                id="input-velocity-azimuth"
                type="number"
                step="any"
                min="0"
                max="360"
                value={values.velocity_azimuth_deg}
                onChange={(e) => handleInputChange('velocity_azimuth_deg', e.target.value)}
                required
              />
            </div>
          </div>
          <div className="input-row">
            <div className="input-field">
              <label htmlFor="input-divergent-comp">Divergent Component (mm/yr)</label>
              <input
                id="input-divergent-comp"
                type="number"
                step="any"
                min="-250"
                max="250"
                value={values.divergent_component}
                onChange={(e) => handleInputChange('divergent_component', e.target.value)}
                required
              />
            </div>
            <div className="input-field">
              <label htmlFor="input-right-lateral-comp">Right-Lateral Motion (mm/yr)</label>
              <input
                id="input-right-lateral-comp"
                type="number"
                step="any"
                min="-250"
                max="250"
                value={values.right_lateral_component}
                onChange={(e) => handleInputChange('right_lateral_component', e.target.value)}
                required
              />
            </div>
          </div>
        </fieldset>

        {/* Crustal Setting */}
        <fieldset className="form-group-card">
          <legend className="group-legend">4. Crustal & Geological Setting</legend>
          <div className="input-row">
            <div className="input-field">
              <label htmlFor="input-seafloor-age">Seafloor Crustal Age (Ma)</label>
              <input
                id="input-seafloor-age"
                type="number"
                step="any"
                min="0"
                max="300"
                value={values.seafloor_age_ma}
                onChange={(e) => handleInputChange('seafloor_age_ma', e.target.value)}
                required
              />
            </div>
            <div className="input-field">
              <label htmlFor="input-orogen-flag">Orogen Belt Flag</label>
              <select
                id="input-orogen-flag"
                value={values.orogen_flag}
                onChange={(e) => handleInputChange('orogen_flag', parseInt(e.target.value, 10))}
              >
                <option value={0}>0 — Oceanic / Stable Craton</option>
                <option value={1}>1 — Active Orogen Belt</option>
              </select>
            </div>
          </div>
        </fieldset>
      </div>

      {/* Action Row */}
      <div className="form-actions-row">
        <button
          id="btn-classify"
          type="submit"
          disabled={isLoading}
          className="btn-primary"
        >
          <Play size={14} />
          <span>{isLoading ? 'Classifying via FastAPI...' : 'Execute PB2002 Classification'}</span>
        </button>

        <button
          id="btn-reset"
          type="button"
          onClick={onReset}
          disabled={isLoading}
          className="btn-secondary"
          title="Reset to default event"
        >
          <RotateCcw size={13} />
          <span>Reset Defaults</span>
        </button>

        {predictedClass && (
          <div className="quick-result-pill" style={{ borderColor: PB2002_CLASSES[predictedClass]?.color }}>
            <span className="pill-dot" style={{ backgroundColor: PB2002_CLASSES[predictedClass]?.color }} />
            <span className="pill-text">Latest: <strong>{predictedClass}</strong> ({PB2002_CLASSES[predictedClass]?.name})</span>
          </div>
        )}
      </div>
    </form>
  );
};
