import React from 'react';
import { Activity, Layers, Database } from 'lucide-react';

interface HeaderProps {
  currentTab: 'classify' | 'model' | 'benchmarks';
  onTabChange: (tab: 'classify' | 'model' | 'benchmarks') => void;
  apiConnected: boolean | null;
}

export const Header: React.FC<HeaderProps> = ({
  currentTab,
  onTabChange,
  apiConnected,
}) => {
  return (
    <header className="site-header">
      <div className="header-container">
        {/* Left: Branding */}
        <div className="branding">
          <span className="brand-title">TECTONICSENSE</span>
          <span className="brand-subtitle">
            Tectonic Boundary Classification
          </span>
        </div>

        {/* Right: Navigation Tabs & Backend Status */}
        <div className="header-actions">
          <nav className="tab-nav">
            <button
              id="tab-classify"
              type="button"
              onClick={() => onTabChange('classify')}
              className={`tab-btn ${currentTab === 'classify' ? 'active' : ''}`}
            >
              <Activity size={14} className="tab-icon" />
              <span>Classification</span>
            </button>
            <button
              id="tab-model"
              type="button"
              onClick={() => onTabChange('model')}
              className={`tab-btn ${currentTab === 'model' ? 'active' : ''}`}
            >
              <Layers size={14} className="tab-icon" />
              <span>Model & SHAP</span>
            </button>
            <button
              id="tab-benchmarks"
              type="button"
              onClick={() => onTabChange('benchmarks')}
              className={`tab-btn ${currentTab === 'benchmarks' ? 'active' : ''}`}
            >
              <Database size={14} className="tab-icon" />
              <span>Validation & Benchmarks</span>
            </button>
          </nav>

          {/* API Backend Health Indicator */}
          <div
            id="backend-status-badge"
            className="status-badge"
            title={
              apiConnected === true
                ? 'FastAPI ML Inference Engine connected at http://127.0.0.1:8000'
                : apiConnected === false
                ? 'Backend unreachable. Ensure uvicorn api.main:app is running'
                : 'Verifying FastAPI health...'
            }
          >
            <span
              className={`status-dot ${
                apiConnected === true
                  ? 'status-online'
                  : apiConnected === false
                  ? 'status-offline'
                  : 'status-checking'
              }`}
            />
            <span className="status-text">
              {apiConnected === true
                ? 'Backend: Connected'
                : apiConnected === false
                ? 'Backend: Offline'
                : 'Connecting...'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};
