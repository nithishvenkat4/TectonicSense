import React from 'react';
import type { PredictionResponse, PB2002Class } from '../types';
import { PB2002_CLASSES } from '../data/constants';
import { CheckCircle2, ShieldCheck, Compass, Info } from 'lucide-react';

interface PredictionResultProps {
  prediction: PredictionResponse | null;
  isLoading: boolean;
  divergentComponent: number;
  elevationM: number;
}

export const PredictionResult: React.FC<PredictionResultProps> = ({
  prediction,
  isLoading,
  divergentComponent,
  elevationM,
}) => {
  if (isLoading) {
    return (
      <div className="prediction-card loading">
        <div className="loading-spinner-wrap">
          <div className="loading-spinner" />
          <span className="loading-text">Executing XGBoost inference on FastAPI backend...</span>
          <span className="loading-subtext">Evaluating 26 deterministic geometric and kinematic features</span>
        </div>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="prediction-card empty">
        <Compass size={32} className="empty-icon" />
        <h3 className="empty-title">Awaiting Observation Classification</h3>
        <p className="empty-desc">
          Click on the interactive map or select an observation preset, then press{' '}
          <strong>"Execute PB2002 Classification"</strong> to run inference with the production XGBoost model.
        </p>
      </div>
    );
  }

  const meta = PB2002_CLASSES[prediction.predicted_class] || {
    code: prediction.predicted_class,
    name: prediction.predicted_class,
    type: 'Tectonic Boundary',
    description: 'PB2002 geological boundary classification',
    color: '#334155',
  };

  const confidencePct = (prediction.confidence * 100).toFixed(2);
  const orderedClasses: PB2002Class[] = ['SUB', 'OSR', 'OTF', 'CRB', 'CTF', 'CCB', 'OCB'];

  return (
    <div id="prediction-results-card" className="prediction-card active">
      {/* Primary Result Banner */}
      <div className="result-banner" style={{ borderLeftColor: meta.color }}>
        <div className="result-top-line">
          <div className="result-badge-wrap">
            <span
              className="class-badge"
              style={{ backgroundColor: meta.color, color: '#ffffff' }}
            >
              {meta.code}
            </span>
            <div className="class-title-wrap">
              <h3 className="class-name">{meta.name}</h3>
              <span className="class-type">{meta.type}</span>
            </div>
          </div>
          <div className="confidence-pill" title="Model softmax confidence">
            <ShieldCheck size={14} className="confidence-icon" />
            <span className="confidence-label">Confidence:</span>
            <span className="confidence-val font-mono">{confidencePct}%</span>
          </div>
        </div>

        <p className="class-description">{meta.description}</p>

        {/* Physics-based context interpretation */}
        <div className="physics-interpretation">
          <Info size={14} className="info-icon" />
          <div className="physics-text">
            <strong>Geophysical Interpretation: </strong>
            {divergentComponent < -20 ? (
              <span>
                Strong negative divergent component ({divergentComponent.toFixed(1)} mm/yr) confirms acute lithospheric convergence.
              </span>
            ) : divergentComponent > 5 ? (
              <span>
                Positive divergent component (+{divergentComponent.toFixed(1)} mm/yr) confirms crustal extension and spreading.
              </span>
            ) : (
              <span>
                Near-zero divergence with strike-slip displacement indicates lateral plate conservative motion.
              </span>
            )}
            {elevationM < -4000 ? (
              <span> Deep bathymetry ({elevationM.toFixed(0)} m) aligns with oceanic trench/abyssal morphology.</span>
            ) : elevationM > 1000 ? (
              <span> High boundary elevation (+{elevationM.toFixed(0)} m) indicates continental orogenic crust.</span>
            ) : null}
          </div>
        </div>
      </div>

      {/* 7-Class Probability Distribution */}
      <div className="probabilities-section">
        <div className="probabilities-header">
          <div className="flex items-center space-x-1.5">
            <CheckCircle2 size={14} className="text-slate-600" />
            <h4 className="probabilities-title">PB2002 Multi-Class Probability Distribution (7 Classes)</h4>
          </div>
          <span className="probabilities-subtitle font-mono">Softmax Probabilities</span>
        </div>

        <div className="prob-table-container">
          <table className="prob-table">
            <thead>
              <tr>
                <th className="col-code">Code</th>
                <th className="col-name">Regime Type</th>
                <th className="col-bar">Relative Likelihood</th>
                <th className="col-val">Probability</th>
              </tr>
            </thead>
            <tbody>
              {orderedClasses.map((cls) => {
                const prob = prediction.class_probabilities[cls] || 0;
                const probPct = (prob * 100).toFixed(2);
                const isWinner = cls === prediction.predicted_class;
                const classMeta = PB2002_CLASSES[cls];

                return (
                  <tr
                    key={cls}
                    className={`prob-row ${isWinner ? 'winner-row' : ''}`}
                    style={isWinner ? { backgroundColor: `${classMeta.color}0a` } : {}}
                  >
                    <td className="col-code">
                      <span
                        className="prob-code-badge"
                        style={{
                          backgroundColor: isWinner ? classMeta.color : '#f1f5f9',
                          color: isWinner ? '#ffffff' : '#334155',
                        }}
                      >
                        {cls}
                      </span>
                    </td>
                    <td className="col-name">
                      <div className="prob-name-wrap">
                        <span className={`prob-name ${isWinner ? 'font-semibold text-slate-900' : 'text-slate-700'}`}>
                          {classMeta.name}
                        </span>
                        <span className="prob-category text-[11px] text-slate-500">
                          {classMeta.type.split(' ')[0]}
                        </span>
                      </div>
                    </td>
                    <td className="col-bar">
                      <div className="prob-bar-track">
                        <div
                          className="prob-bar-fill"
                          style={{
                            width: `${Math.max(0.5, prob * 100)}%`,
                            backgroundColor: classMeta.color,
                          }}
                        />
                      </div>
                    </td>
                    <td className="col-val font-mono">
                      <span className={isWinner ? 'font-bold text-slate-900' : 'text-slate-600'}>
                        {probPct}%
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
