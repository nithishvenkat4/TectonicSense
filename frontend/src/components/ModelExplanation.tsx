import React from 'react';
import { GLOBAL_SHAP_FEATURES } from '../data/constants';
import type { ModelInfoResponse } from '../types';
import { Layers, Info, CheckCircle, BarChart3 } from 'lucide-react';

interface ModelExplanationProps {
  modelInfo: ModelInfoResponse | null;
}

export const ModelExplanation: React.FC<ModelExplanationProps> = ({ modelInfo }) => {
  const maxImportance = Math.max(...GLOBAL_SHAP_FEATURES.map((f) => f.importance));

  return (
    <div className="model-explanation-container">
      {/* Top Header */}
      <div className="section-header">
        <div className="section-title-wrap">
          <Layers size={18} className="section-icon" />
          <h2 className="section-title">Model Architecture & Global SHAP Feature Importance</h2>
        </div>
        <span className="section-subtitle">
          Geophysical feature attribution calculated via TreeSHAP on test observations
        </span>
      </div>

      {/* Model Specifications Card */}
      <div className="model-specs-grid">
        <div className="spec-card">
          <span className="spec-label">Model Architecture</span>
          <span className="spec-val font-semibold">{modelInfo?.model_name || 'Tuned XGBoost (Final Production)'}</span>
          <span className="spec-sub">Framework: {modelInfo?.framework || 'XGBoost'} {modelInfo?.framework_version || '3.0.5'}</span>
        </div>
        <div className="spec-card">
          <span className="spec-label">Input Dimension</span>
          <span className="spec-val font-mono">{modelInfo?.feature_count || 26} Engineered Features</span>
          <span className="spec-sub">Deterministic spatial & kinematic derivation</span>
        </div>
        <div className="spec-card">
          <span className="spec-label">Target Taxonomy</span>
          <span className="spec-val font-mono">{modelInfo?.class_count || 7} PB2002 Boundary Classes</span>
          <span className="spec-sub">Peter Bird (2002) global plate model</span>
        </div>
        <div className="spec-card">
          <span className="spec-label">Model Verification</span>
          <span className="spec-val font-semibold text-emerald-700">Phase 4 Certified</span>
          <span className="spec-sub">Deterministic frozen artifact (xgb_model.json)</span>
        </div>
      </div>

      {/* SHAP Feature Importance Chart */}
      <div className="shap-card">
        <div className="shap-header">
          <div className="flex items-center space-x-2">
            <BarChart3 size={16} className="text-slate-700" />
            <h3 className="shap-title">Global Feature Attribution (Mean |SHAP Value|)</h3>
          </div>
          <span className="shap-sub text-xs text-slate-500">
            Higher values indicate greater influence on PB2002 classification
          </span>
        </div>

        <div className="shap-list">
          {GLOBAL_SHAP_FEATURES.map((feat, idx) => {
            const widthPct = ((feat.importance / maxImportance) * 100).toFixed(1);

            return (
              <div key={feat.feature} className="shap-item">
                <div className="shap-item-top">
                  <div className="shap-name-col">
                    <span className="shap-rank font-mono">#{idx + 1}</span>
                    <span className="shap-label font-medium">{feat.label}</span>
                    <code className="shap-code font-mono text-[11px] text-slate-500">
                      {feat.feature}
                    </code>
                  </div>
                  <span className="shap-val font-mono font-semibold text-slate-800">
                    {feat.importance.toFixed(4)}
                  </span>
                </div>

                <div className="shap-bar-track">
                  <div
                    className="shap-bar-fill"
                    style={{ width: `${widthPct}%` }}
                  />
                </div>

                <p className="shap-desc">{feat.description}</p>
              </div>
            );
          })}
        </div>
      </div>

      {/* Physical Mechanism Documentation */}
      <div className="physics-card">
        <div className="physics-header">
          <Info size={16} className="text-slate-700" />
          <h3 className="physics-title">Geophysical Grounding & Physical Interpretability</h3>
        </div>
        <div className="physics-body text-xs text-slate-600 space-y-2">
          <p>
            <strong>1. Divergent Component Dominance: </strong>
            The single most decisive discriminator in PB2002 boundary classification is the normal velocity vector.
            Subduction zones (SUB) and collision boundaries (CCB/OCB) consistently exhibit strongly negative divergent components
            (crustal shortening and lithospheric convergence), whereas oceanic spreading ridges (OSR) and continental rifts (CRB)
            exhibit positive divergent components.
          </p>
          <p>
            <strong>2. Topographic & Bathymetric Discrimination: </strong>
            Oceanic trenches reach extreme abyssal depths (-6,000 m to -11,000 m), providing an unambiguous morphological signature
            for oceanic subduction that easily separates it from continental collision belts (e.g. Himalayas, Tibetan Plateau &gt; +4,000 m)
            and mid-ocean ridges (-2,500 m).
          </p>
          <p>
            <strong>3. Strike-Slip Vector Kinematics: </strong>
            Oceanic transform faults (OTF) and continental transforms (CTF) are characterized by dominant lateral shear motion
            with minimal divergence, enabling clean separation from pure spreading or convergent boundaries.
          </p>
        </div>
        <div className="mt-3 pt-2 border-t border-slate-100 flex items-center space-x-2 text-[11px] text-emerald-800">
          <CheckCircle size={14} className="text-emerald-600" />
          <span>All 26 production features are derived without circular target leakage.</span>
        </div>
      </div>
    </div>
  );
};
