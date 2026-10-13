import React from 'react';
import { ShieldCheck } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="site-footer">
      <div className="footer-container">
        {/* Academic & Scientific Scope Notice */}
        <div className="footer-scope-card">
          <div className="scope-header">
            <ShieldCheck size={16} className="text-slate-600" />
            <span className="scope-title">Scientific Classification Scope & Disclaimer</span>
          </div>
          <p className="scope-text">
            TectonicSense classifies geological tectonic plate boundary regimes (PB2002 taxonomy)
            based on observed kinematic and spatial parameters.
            <strong> This system classifies boundary tectonic regimes and does NOT predict, forecast, or provide early warning for earthquake events.</strong>
          </p>
        </div>

        {/* Citations & Metadata */}
        <div className="footer-bottom-row">
          <div className="footer-citations text-xs text-slate-500">
            <span>Reference: Bird, P. (2003) <em>An updated digital model of plate boundaries</em>, G-Cubed, 4(3), 1027.</span>
            <span className="citation-sep">•</span>
            <span>USGS Earthquake Hazards Program</span>
          </div>
          <div className="footer-tech-stack text-xs text-slate-500 font-mono">
            <span>XGBoost 3.0.5</span>
            <span className="citation-sep">•</span>
            <span>FastAPI REST API</span>
            <span className="citation-sep">•</span>
            <span>MapLibre GL</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
