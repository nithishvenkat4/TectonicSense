import { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header';
import { ObservationForm } from './components/ObservationForm';
import { TectonicMap } from './components/TectonicMap';
import { PredictionResult } from './components/PredictionResult';
import { ModelExplanation } from './components/ModelExplanation';
import { BenchmarksView } from './components/BenchmarksView';

import { PRESETS } from './data/presets';
import { checkHealth, getModelInfo, deriveFeatures, predictSingle } from './services/api';
import type { RawObservationInput, PredictionResponse, ModelInfoResponse } from './types';
import { AlertCircle, RefreshCw } from 'lucide-react';
import './index.css';

export function App() {
  const [currentTab, setCurrentTab] = useState<'classify' | 'model' | 'benchmarks'>('classify');
  const [observation, setObservation] = useState<RawObservationInput>(PRESETS[0].data);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);

  // Check backend health and model metadata on mount
  const verifyBackend = useCallback(async () => {
    try {
      const health = await checkHealth();
      setApiConnected(health.status === 'healthy' && health.model_loaded);
      const info = await getModelInfo();
      setModelInfo(info);
      setError(null);
    } catch {
      setApiConnected(false);
    }
  }, []);

  useEffect(() => {
    verifyBackend();
  }, [verifyBackend]);

  // Handle map click updating coordinates
  const handleCoordinateChange = (lat: number, lon: number) => {
    setObservation((prev) => ({
      ...prev,
      latitude: lat,
      longitude: lon,
    }));
  };

  // Reset to initial Kermadec reference preset
  const handleReset = () => {
    setObservation({ ...PRESETS[0].data });
    setPrediction(null);
    setError(null);
  };

  // Execute single observation inference
  const handleClassify = async () => {
    setIsLoading(true);
    setError(null);

    try {
      // 1. Deterministically derive the canonical 26 production features
      const features = deriveFeatures(observation);

      // 2. Transmit to FastAPI inference backend POST /predict
      const result = await predictSingle(features);

      // 3. Update prediction state
      setPrediction(result);
      setApiConnected(true);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Classification request failed';
      setError(msg);
      // Re-check backend health
      try {
        await checkHealth();
      } catch {
        setApiConnected(false);
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-root">
      <Header
        currentTab={currentTab}
        onTabChange={setCurrentTab}
        apiConnected={apiConnected}
      />

      <main className="main-content">
        {/* Backend Unreachable Alert Banner */}
        {apiConnected === false && (
          <div className="system-alert-banner">
            <div className="alert-content">
              <AlertCircle size={16} className="alert-icon" />
              <span>
                FastAPI inference backend is not detected at <code>http://127.0.0.1:8000</code>.
                Make sure the backend server is running via <code>uvicorn api.main:app --port 8000</code>.
              </span>
            </div>
            <button
              type="button"
              onClick={verifyBackend}
              className="retry-btn"
              title="Retry connection to backend"
            >
              <RefreshCw size={13} />
              <span>Retry</span>
            </button>
          </div>
        )}

        {/* Inference Error Notification */}
        {error && (
          <div className="error-alert-banner">
            <AlertCircle size={16} className="alert-icon" />
            <div className="error-text">
              <strong>Prediction Error: </strong>
              <span>{error}</span>
            </div>
          </div>
        )}

        {/* View Switcher */}
        {currentTab === 'classify' && (
          <div className="workspace-layout">
            {/* Left Column: Observation Parameter Input */}
            <div className="workspace-column left-column">
              <ObservationForm
                values={observation}
                onChange={setObservation}
                onSubmit={handleClassify}
                isLoading={isLoading}
                onReset={handleReset}
                predictedClass={prediction?.predicted_class}
              />
            </div>

            {/* Right Column: Spatial Context & Classification Result */}
            <div className="workspace-column right-column">
              <div className="panel-card map-panel">
                <div className="panel-header">
                  <h3 className="panel-title">PB2002 Spatial Context Map</h3>
                  <span className="panel-subtitle">
                    Epicenter: {observation.latitude.toFixed(4)}°, {observation.longitude.toFixed(4)}°
                  </span>
                </div>
                <TectonicMap
                  latitude={observation.latitude}
                  longitude={observation.longitude}
                  onCoordinateChange={handleCoordinateChange}
                  predictedClass={prediction?.predicted_class}
                />
              </div>

              <PredictionResult
                prediction={prediction}
                isLoading={isLoading}
                divergentComponent={observation.divergent_component}
                elevationM={observation.boundary_elevation_m}
              />
            </div>
          </div>
        )}

        {currentTab === 'model' && (
          <div className="tab-pane">
            <ModelExplanation modelInfo={modelInfo} />
          </div>
        )}

        {currentTab === 'benchmarks' && (
          <div className="tab-pane">
            <BenchmarksView />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
