import React from 'react';
import { Database, Award } from 'lucide-react';
import { MODEL_BENCHMARKS, GENERALIZATION_BENCHMARKS } from '../data/constants';

export const BenchmarksView: React.FC = () => {
  return (
    <div className="benchmarks-view-container">
      {/* Section Header */}
      <div className="section-header">
        <div className="section-title-wrap">
          <Database size={18} className="section-icon" />
          <h2 className="section-title">Model Evaluation & Generalization Benchmarks</h2>
        </div>
        <span className="section-subtitle">
          Rigorous benchmarking across candidate architectures and spatial split strategies
        </span>
      </div>

      {/* Model Comparison Table */}
      <div className="benchmark-card">
        <div className="benchmark-card-header">
          <div className="flex items-center space-x-2">
            <Award size={16} className="text-amber-600" />
            <h3 className="benchmark-title">Production Candidate Benchmarking (Test Set: N=35,964)</h3>
          </div>
          <span className="benchmark-sub text-xs text-slate-500">
            Phase 2 and Phase 3 comparative model evaluation
          </span>
        </div>

        <div className="benchmark-table-wrap">
          <table className="benchmark-table">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Status</th>
                <th>Macro F1 Score</th>
                <th>Overall Accuracy</th>
                <th>Training Time (s)</th>
              </tr>
            </thead>
            <tbody>
              {MODEL_BENCHMARKS.map((m) => {
                const isSelected = m.status === 'Final Production';
                return (
                  <tr
                    key={m.model}
                    className={`bench-row ${isSelected ? 'selected-model-row' : ''}`}
                  >
                    <td className="font-medium">
                      <div className="flex items-center space-x-2">
                        {isSelected && (
                          <span className="inline-block w-2 h-2 rounded-full bg-emerald-500" />
                        )}
                        <span className={isSelected ? 'font-bold text-slate-900' : 'text-slate-800'}>
                          {m.model}
                        </span>
                      </div>
                    </td>
                    <td>
                      <span
                        className={`status-pill ${
                          isSelected
                            ? 'status-prod'
                            : m.status === 'Candidate'
                            ? 'status-cand'
                            : 'status-base'
                        }`}
                      >
                        {m.status}
                      </span>
                    </td>
                    <td className="font-mono font-semibold">
                      {(m.macroF1 * 100).toFixed(2)}%
                    </td>
                    <td className="font-mono">
                      {(m.accuracy * 100).toFixed(2)}%
                    </td>
                    <td className="font-mono text-slate-600">
                      {m.trainingTimeSec.toFixed(1)}s
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Generalization Benchmark Cards */}
      <div className="generalization-grid">
        <div className="gen-card">
          <div className="gen-card-header">
            <h4 className="gen-title">Random Stratified Split</h4>
            <span className="gen-tag text-emerald-800 bg-emerald-50">In-Boundary Interpolation</span>
          </div>
          <div className="gen-metrics">
            <div className="gen-metric">
              <span className="gen-num font-mono">{(GENERALIZATION_BENCHMARKS.randomStratified.macroF1 * 100).toFixed(2)}%</span>
              <span className="gen-lbl">Macro F1 Score</span>
            </div>
            <div className="gen-metric">
              <span className="gen-num font-mono">{GENERALIZATION_BENCHMARKS.randomStratified.errors}</span>
              <span className="gen-lbl">Errors / 35,964</span>
            </div>
          </div>
          <p className="gen-desc">{GENERALIZATION_BENCHMARKS.randomStratified.description}</p>
        </div>

        <div className="gen-card">
          <div className="gen-card-header">
            <h4 className="gen-title">Boundary-Aware Group Split</h4>
            <span className="gen-tag text-indigo-800 bg-indigo-50">Out-of-Boundary Transfer</span>
          </div>
          <div className="gen-metrics">
            <div className="gen-metric">
              <span className="gen-num font-mono">{(GENERALIZATION_BENCHMARKS.boundaryAware.macroF1 * 100).toFixed(2)}%</span>
              <span className="gen-lbl">Macro F1 Score</span>
            </div>
            <div className="gen-metric">
              <span className="gen-num font-mono">28</span>
              <span className="gen-lbl">Disjoint Test Boundaries</span>
            </div>
          </div>
          <p className="gen-desc">{GENERALIZATION_BENCHMARKS.boundaryAware.description}</p>
        </div>
      </div>
    </div>
  );
};
