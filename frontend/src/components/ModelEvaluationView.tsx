import React, { useState } from 'react'
import {
  BarChart2,
  CheckCircle,
  HelpCircle,
  LineChart as LineChartIcon,
  ShieldAlert,
} from 'lucide-react'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { ModelEvaluation, ModelInfo } from '../types'
import { MetricCard } from './MetricCard'

interface ModelEvaluationViewProps {
  evaluation: ModelEvaluation | null
  modelInfo: ModelInfo | null
}

type ChartType = 'roc' | 'pr' | 'threshold' | 'calibration' | 'lift' | 'distribution'

export const ModelEvaluationView: React.FC<ModelEvaluationViewProps> = ({
  evaluation,
  modelInfo,
}) => {
  const [selectedChart, setSelectedChart] = useState<ChartType>('roc')

  if (!evaluation || !modelInfo) {
    return (
      <div className="bg-panel rounded-lg border border-border p-12 text-center">
        <p className="text-sm text-ink-muted">Loading model evaluation and validation artifacts...</p>
      </div>
    )
  }

  const { holdout, candidates, selection, permutation_importance } = evaluation
  const ci = holdout.confidence_intervals
  const atThresh = holdout.at_decision_threshold
  const cm = atThresh.confusion_matrix

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 bg-panel p-4 rounded-lg border border-border">
        <div>
          <h2 className="text-base font-semibold text-ink">Empirical Model Evaluation & Benchmarks</h2>
          <p className="text-xs text-ink-muted mt-0.5">
            Holdout test results on {holdout.rows.toLocaleString()} customers ({holdout.positives} churned,{' '}
            {((holdout.positives / holdout.rows) * 100).toFixed(1)}% base rate).
          </p>
        </div>
        <div className="flex items-center gap-2">
          {evaluation.synthetic_data ? (
            <span className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-amber-50 text-amber-800 border border-amber-200 text-xs font-semibold">
              <ShieldAlert className="w-3.5 h-3.5" />
              Synthetic Fixture Data
            </span>
          ) : (
            <span className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-emerald-50 text-emerald-800 border border-emerald-200 text-xs font-semibold">
              <CheckCircle className="w-3.5 h-3.5" />
              Verified Public Dataset
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <MetricCard
          label="ROC-AUC"
          value={holdout.ranking.roc_auc.toFixed(4)}
          confidenceInterval={ci.roc_auc}
          subtext="Area under ROC"
          variant="highlight"
        />
        <MetricCard
          label="PR-AUC"
          value={holdout.ranking.pr_auc.toFixed(4)}
          confidenceInterval={ci.pr_auc}
          subtext="Precision-Recall Area"
          variant="highlight"
        />
        <MetricCard
          label="Brier Score"
          value={holdout.ranking.brier.toFixed(4)}
          subtext="Calibration loss (lower is better)"
        />
        <MetricCard
          label="F1 Score"
          value={atThresh.f1.toFixed(4)}
          confidenceInterval={ci.f1}
          subtext={`At threshold ${atThresh.threshold.toFixed(4)}`}
        />
        <MetricCard
          label="Precision"
          value={`${(atThresh.precision * 100).toFixed(1)}%`}
          subtext="True positive rate in flagged"
        />
        <MetricCard
          label="Recall"
          value={`${(atThresh.recall * 100).toFixed(1)}%`}
          subtext="Churners captured"
        />
      </div>

      <div className="bg-panel rounded-lg border border-border p-5 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-border pb-3">
          <div className="flex items-center gap-2">
            <LineChartIcon className="w-4 h-4 text-terracotta-600" />
            <h3 className="text-sm font-semibold text-ink">Model Diagnostic Visualizations</h3>
          </div>
          <div className="flex flex-wrap items-center gap-1">
            {(
              [
                { id: 'roc', label: 'ROC Curve' },
                { id: 'pr', label: 'Precision-Recall' },
                { id: 'threshold', label: 'Threshold Sweep' },
                { id: 'calibration', label: 'Calibration (Reliability)' },
                { id: 'lift', label: 'Cumulative Gains / Lift' },
                { id: 'distribution', label: 'Score Distribution' },
              ] as const
            ).map((tab) => (
              <button
                key={tab.id}
                onClick={() => setSelectedChart(tab.id)}
                className={`px-2.5 py-1 rounded text-xs transition-colors ${
                  selectedChart === tab.id
                    ? 'bg-ink text-canvas font-medium'
                    : 'text-ink-muted hover:text-ink bg-subtle'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        <div className="h-80 w-full pt-2">
          {selectedChart === 'roc' && (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={holdout.curves.roc} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E3DDD4" />
                <XAxis
                  dataKey="fpr"
                  label={{ value: 'False Positive Rate (1 - Specificity)', position: 'insideBottom', offset: -10, fontSize: 11 }}
                  fontSize={11}
                  stroke="#8D99A6"
                  domain={[0, 1]}
                  tickFormatter={(v) => v.toFixed(1)}
                />
                <YAxis
                  label={{ value: 'True Positive Rate (Recall)', angle: -90, position: 'insideLeft', fontSize: 11 }}
                  fontSize={11}
                  stroke="#8D99A6"
                  domain={[0, 1]}
                  tickFormatter={(v) => v.toFixed(1)}
                />
                <Tooltip
                  formatter={(val: unknown) => [Number(val || 0).toFixed(4), 'Rate']}
                  labelFormatter={(lbl) => `FPR: ${Number(lbl).toFixed(4)}`}
                  contentStyle={{ backgroundColor: '#1B242D', borderRadius: '6px', border: 'none', color: '#F8F6F0', fontSize: '12px' }}
                />
                <Line type="monotone" dataKey="tpr" stroke="#D65A31" strokeWidth={2.5} dot={false} name="ROC Curve" />
              </LineChart>
            </ResponsiveContainer>
          )}

          {selectedChart === 'pr' && (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={holdout.curves.precision_recall} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E3DDD4" />
                <XAxis
                  dataKey="recall"
                  label={{ value: 'Recall', position: 'insideBottom', offset: -10, fontSize: 11 }}
                  fontSize={11}
                  stroke="#8D99A6"
                  domain={[0, 1]}
                  tickFormatter={(v) => v.toFixed(1)}
                />
                <YAxis
                  label={{ value: 'Precision', angle: -90, position: 'insideLeft', fontSize: 11 }}
                  fontSize={11}
                  stroke="#8D99A6"
                  domain={[0, 1]}
                  tickFormatter={(v) => v.toFixed(1)}
                />
                <Tooltip
                  formatter={(val: unknown) => [Number(val || 0).toFixed(4), 'Rate']}
                  labelFormatter={(lbl) => `Recall: ${Number(lbl).toFixed(4)}`}
                  contentStyle={{ backgroundColor: '#1B242D', borderRadius: '6px', border: 'none', color: '#F8F6F0', fontSize: '12px' }}
                />
                <ReferenceLine y={holdout.ranking.base_rate} stroke="#8D99A6" strokeDasharray="4 4" label="Base Rate" />
                <Line type="monotone" dataKey="precision" stroke="#BF4820" strokeWidth={2.5} dot={false} name="PR Curve" />
              </LineChart>
            </ResponsiveContainer>
          )}

          {selectedChart === 'threshold' && (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={holdout.curves.threshold_sweep} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E3DDD4" />
                <XAxis
                  dataKey="threshold"
                  label={{ value: 'Decision Threshold', position: 'insideBottom', offset: -10, fontSize: 11 }}
                  fontSize={11}
                  stroke="#8D99A6"
                  domain={[0, 1]}
                  tickFormatter={(v) => v.toFixed(1)}
                />
                <YAxis fontSize={11} stroke="#8D99A6" domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)} />
                <Tooltip
                  formatter={(val: unknown) => [Number(val || 0).toFixed(3), 'Score']}
                  labelFormatter={(lbl) => `Threshold: ${Number(lbl).toFixed(2)}`}
                  contentStyle={{ backgroundColor: '#1B242D', borderRadius: '6px', border: 'none', color: '#F8F6F0', fontSize: '12px' }}
                />
                <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: '12px' }} />
                <ReferenceLine x={atThresh.threshold} stroke="#1B242D" strokeDasharray="3 3" label="Optimal Cutoff" />
                <Line type="monotone" dataKey="precision" stroke="#D97706" strokeWidth={2} dot={false} name="Precision" />
                <Line type="monotone" dataKey="recall" stroke="#15803D" strokeWidth={2} dot={false} name="Recall" />
                <Line type="monotone" dataKey="f1" stroke="#D65A31" strokeWidth={2.5} dot={false} name="F1 Score" />
              </LineChart>
            </ResponsiveContainer>
          )}

          {selectedChart === 'calibration' && (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={holdout.curves.calibration} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E3DDD4" />
                <XAxis
                  dataKey="predicted"
                  label={{ value: 'Mean Predicted Churn Probability', position: 'insideBottom', offset: -10, fontSize: 11 }}
                  fontSize={11}
                  stroke="#8D99A6"
                  domain={[0, 1]}
                  tickFormatter={(v) => v.toFixed(1)}
                />
                <YAxis
                  label={{ value: 'Empirical Churn Fraction', angle: -90, position: 'insideLeft', fontSize: 11 }}
                  fontSize={11}
                  stroke="#8D99A6"
                  domain={[0, 1]}
                  tickFormatter={(v) => v.toFixed(1)}
                />
                <Tooltip
                  formatter={(val: unknown) => [Number(val || 0).toFixed(3), 'Rate']}
                  labelFormatter={(lbl) => `Predicted: ${Number(lbl).toFixed(3)}`}
                  contentStyle={{ backgroundColor: '#1B242D', borderRadius: '6px', border: 'none', color: '#F8F6F0', fontSize: '12px' }}
                />
                <Line type="monotone" dataKey="observed" stroke="#D65A31" strokeWidth={2.5} dot={{ r: 4 }} name="Observed Rate" />
              </LineChart>
            </ResponsiveContainer>
          )}

          {selectedChart === 'lift' && (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={holdout.curves.cumulative_gains} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E3DDD4" />
                <XAxis
                  dataKey="population_share"
                  label={{ value: 'Fraction of Customer Base Contacted', position: 'insideBottom', offset: -10, fontSize: 11 }}
                  fontSize={11}
                  stroke="#8D99A6"
                  domain={[0, 1]}
                  tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                />
                <YAxis
                  label={{ value: 'Fraction of Total Churners Captured', angle: -90, position: 'insideLeft', fontSize: 11 }}
                  fontSize={11}
                  stroke="#8D99A6"
                  domain={[0, 1]}
                  tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                />
                <Tooltip
                  formatter={(val: unknown) => [`${(Number(val || 0) * 100).toFixed(1)}%`, 'Captured Churners']}
                  labelFormatter={(lbl) => `Targeting top ${(Number(lbl) * 100).toFixed(0)}%`}
                  contentStyle={{ backgroundColor: '#1B242D', borderRadius: '6px', border: 'none', color: '#F8F6F0', fontSize: '12px' }}
                />
                <Line type="monotone" dataKey="churners_captured" stroke="#15803D" strokeWidth={2.5} dot={{ r: 4 }} name="Gains Curve" />
              </LineChart>
            </ResponsiveContainer>
          )}

          {selectedChart === 'distribution' && (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={holdout.score_histogram} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E3DDD4" />
                <XAxis
                  dataKey="lower"
                  fontSize={11}
                  stroke="#8D99A6"
                  tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                />
                <YAxis fontSize={11} stroke="#8D99A6" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1B242D', borderRadius: '6px', border: 'none', color: '#F8F6F0', fontSize: '12px' }}
                />
                <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: '12px' }} />
                <Area type="monotone" dataKey="retained" stackId="1" stroke="#15803D" fill="#15803D" fillOpacity={0.4} name="Retained Customers" />
                <Area type="monotone" dataKey="churned" stackId="1" stroke="#DC2626" fill="#DC2626" fillOpacity={0.6} name="Churned Customers" />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-8 bg-panel rounded-lg border border-border p-5 space-y-4">
          <div className="border-b border-border pb-3">
            <h3 className="text-sm font-semibold text-ink">Candidate Model Comparison Matrix</h3>
            <p className="text-xs text-ink-muted mt-0.5">
              Selection rule: 1-Standard-Error rule on cross-validated PR-AUC to prioritize parsimony and prevent overfitting.
            </p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead className="bg-subtle border-b border-border">
                <tr>
                  <th className="py-2.5 px-3 font-semibold text-ink">Candidate Architecture</th>
                  <th className="py-2.5 px-3 font-semibold text-ink">Role</th>
                  <th className="py-2.5 px-3 font-semibold text-ink">CV PR-AUC (5-Fold)</th>
                  <th className="py-2.5 px-3 font-semibold text-ink">CV Std Err</th>
                  <th className="py-2.5 px-3 font-semibold text-ink">Test PR-AUC</th>
                  <th className="py-2.5 px-3 font-semibold text-ink">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {candidates.map((cand) => {
                  const isSelected = cand.name === selection.selected
                  const cvPr = cand.cv?.pr_auc?.mean ?? 0
                  const cvStd = (cand.cv?.pr_auc?.std ?? 0) / Math.sqrt(5)
                  const testPr = cand.test?.pr_auc ?? 0

                  return (
                    <tr
                      key={cand.name}
                      className={isSelected ? 'bg-terracotta-50/40 font-medium' : 'hover:bg-canvas'}
                    >
                      <td className="py-2.5 px-3 text-ink">
                        <div>
                          <span className="font-semibold">{cand.label}</span>
                          <p className="text-[11px] text-ink-muted leading-tight mt-0.5">{cand.description}</p>
                        </div>
                      </td>
                      <td className="py-2.5 px-3 text-ink-muted capitalize">{cand.role}</td>
                      <td className="py-2.5 px-3 font-mono text-ink">
                        {cvPr ? cvPr.toFixed(4) : '—'}
                      </td>
                      <td className="py-2.5 px-3 font-mono text-ink-muted">
                        {cvStd ? `±${cvStd.toFixed(4)}` : '—'}
                      </td>
                      <td className="py-2.5 px-3 font-mono text-ink">
                        {testPr ? testPr.toFixed(4) : '—'}
                      </td>
                      <td className="py-2.5 px-3">
                        {isSelected ? (
                          <span className="px-2 py-0.5 rounded bg-terracotta-600 text-white font-bold text-[10px]">
                            SELECTED
                          </span>
                        ) : cand.eligible_for_selection ? (
                          <span className="text-ink-muted text-[11px]">Eligible</span>
                        ) : (
                          <span className="text-ink-muted/70 text-[11px]">Reference</span>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          <div className="p-3 bg-canvas rounded border border-border text-xs text-ink leading-relaxed">
            <span className="font-semibold text-terracotta-700">Selection Rationale: </span>
            {selection.explanation}
          </div>
        </div>

        <div className="lg:col-span-4 bg-panel rounded-lg border border-border p-5 flex flex-col justify-between">
          <div>
            <div className="border-b border-border pb-3">
              <h3 className="text-sm font-semibold text-ink">Holdout Confusion Matrix</h3>
              <p className="text-xs text-ink-muted mt-0.5">
                Cutoff = {atThresh.threshold.toFixed(4)} (F1-optimal)
              </p>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-3 text-center">
              <div className="p-3 rounded border border-emerald-200 bg-emerald-50/50">
                <span className="text-[11px] text-emerald-800 uppercase font-semibold">True Negative</span>
                <p className="text-xl font-bold font-mono text-emerald-900 mt-1">{cm.tn.toLocaleString()}</p>
                <span className="text-[10px] text-ink-muted">Retained identified</span>
              </div>
              <div className="p-3 rounded border border-amber-200 bg-amber-50/50">
                <span className="text-[11px] text-amber-800 uppercase font-semibold">False Positive</span>
                <p className="text-xl font-bold font-mono text-amber-900 mt-1">{cm.fp.toLocaleString()}</p>
                <span className="text-[10px] text-ink-muted">False alarm</span>
              </div>
              <div className="p-3 rounded border border-red-200 bg-red-50/50">
                <span className="text-[11px] text-red-800 uppercase font-semibold">False Negative</span>
                <p className="text-xl font-bold font-mono text-red-900 mt-1">{cm.fn.toLocaleString()}</p>
                <span className="text-[10px] text-ink-muted">Missed churners</span>
              </div>
              <div className="p-3 rounded border border-emerald-200 bg-emerald-50/50">
                <span className="text-[11px] text-emerald-800 uppercase font-semibold">True Positive</span>
                <p className="text-xl font-bold font-mono text-emerald-900 mt-1">{cm.tp.toLocaleString()}</p>
                <span className="text-[10px] text-ink-muted">Caught churners</span>
              </div>
            </div>

            <div className="mt-4 pt-3 border-t border-border space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-ink-muted">Holdout Accuracy:</span>
                <span className="font-mono font-semibold text-ink">{(atThresh.accuracy * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-ink-muted">Holdout Specificity:</span>
                <span className="font-mono font-semibold text-ink">{(atThresh.specificity * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-ink-muted">Holdout Flagged Rate:</span>
                <span className="font-mono font-semibold text-ink">{(atThresh.flagged_rate * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-panel rounded-lg border border-border p-5 space-y-4">
        <div className="border-b border-border pb-3 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <div>
            <div className="flex items-center gap-2">
              <BarChart2 className="w-4 h-4 text-terracotta-600" />
              <h3 className="text-sm font-semibold text-ink">Permutation Feature Importance</h3>
            </div>
            <p className="text-xs text-ink-muted mt-0.5">
              Evaluated on the test split using Average Precision score loss across {permutation_importance.repeats} shuffles.
            </p>
          </div>
          <span className="text-xs text-ink-muted flex items-center gap-1" title="Permutation importance measures how model predictive power drops when a feature is shuffled. It does not prove direct real-world causal impact.">
            <HelpCircle className="w-3.5 h-3.5" />
            Predictive Association Only
          </span>
        </div>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={permutation_importance.features.slice(0, 10)}
              layout="vertical"
              margin={{ top: 10, right: 30, left: 100, bottom: 10 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#E3DDD4" horizontal={false} />
              <XAxis type="number" fontSize={11} stroke="#8D99A6" tickFormatter={(v) => v.toFixed(3)} />
              <YAxis type="category" dataKey="label" fontSize={11} stroke="#1B242D" width={95} />
              <Tooltip
                formatter={(val: unknown) => [Number(val || 0).toFixed(4), 'Mean Drop in PR-AUC']}
                contentStyle={{ backgroundColor: '#1B242D', borderRadius: '6px', border: 'none', color: '#F8F6F0', fontSize: '12px' }}
              />
              <Bar dataKey="importance_mean" fill="#D65A31" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}
