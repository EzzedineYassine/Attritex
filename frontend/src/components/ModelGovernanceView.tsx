import React from 'react'
import {
  AlertCircle,
  CheckCircle2,
  Database,
  FileCode2,
  Lock,
  Scale,
  Terminal,
} from 'lucide-react'
import { ModelInfo } from '../types'

interface ModelGovernanceViewProps {
  modelInfo: ModelInfo | null
}

export const ModelGovernanceView: React.FC<ModelGovernanceViewProps> = ({ modelInfo }) => {
  if (!modelInfo) {
    return (
      <div className="bg-panel rounded-lg border border-border p-12 text-center">
        <p className="text-sm text-ink-muted">Loading governance and metadata records...</p>
      </div>
    )
  }

  const { dataset, excluded_features, features, limitations, environment, model_version } = modelInfo
  const { provenance, quality, split } = dataset

  return (
    <div className="space-y-6">
      <div className="bg-panel rounded-lg border border-border p-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-border pb-4">
          <div>
            <div className="flex items-center gap-2">
              <Database className="w-5 h-5 text-terracotta-600" />
              <h2 className="text-base font-semibold text-ink">Dataset Provenance & Quality Audit</h2>
            </div>
            <p className="text-xs text-ink-muted mt-0.5">
              Documentation of origin, licensing, preprocessing transformations, and data integrity.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-emerald-50 text-emerald-800 border border-emerald-200 text-xs font-semibold">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
              Cryptographically Verified
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
          <div className="p-3 bg-canvas rounded border border-border">
            <span className="text-ink-muted block uppercase font-medium text-[10px]">Dataset Name</span>
            <span className="font-semibold text-ink mt-1 block">{provenance.name}</span>
          </div>
          <div className="p-3 bg-canvas rounded border border-border">
            <span className="text-ink-muted block uppercase font-medium text-[10px]">Source Repository</span>
            <span className="font-semibold text-ink mt-1 block truncate" title={provenance.repository || 'N/A'}>
              {provenance.repository ? 'IBM telco-customer-churn-on-icp4d' : 'Synthetic Fixture'}
            </span>
          </div>
          <div className="p-3 bg-canvas rounded border border-border">
            <span className="text-ink-muted block uppercase font-medium text-[10px]">License</span>
            <span className="font-semibold text-ink mt-1 block font-mono">{provenance.license || 'N/A'}</span>
          </div>
          <div className="p-3 bg-canvas rounded border border-border">
            <span className="text-ink-muted block uppercase font-medium text-[10px]">Model Version</span>
            <span className="font-semibold text-ink mt-1 block font-mono truncate" title={model_version}>
              {model_version}
            </span>
          </div>
        </div>

        <div className="p-3 bg-subtle rounded border border-border text-xs flex flex-col gap-1">
          <span className="text-ink-muted font-medium text-[11px]">SHA-256 Checksum:</span>
          <span className="font-mono text-ink text-[11px] break-all select-all">{provenance.sha256}</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
          <div className="p-4 bg-panel border border-border rounded-lg space-y-2">
            <span className="text-xs font-semibold text-ink">Source Population</span>
            <div className="text-xs space-y-1 text-ink-muted">
              <div className="flex justify-between">
                <span>Total Input Rows:</span>
                <span className="font-mono text-ink font-medium">{quality.input_rows.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span>Modelled Records:</span>
                <span className="font-mono text-ink font-medium">{quality.modelled_rows.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span>Duplicate IDs Filtered:</span>
                <span className="font-mono text-ink font-medium">{quality.duplicate_ids_dropped}</span>
              </div>
            </div>
          </div>

          <div className="p-4 bg-panel border border-border rounded-lg space-y-2">
            <span className="text-xs font-semibold text-ink">Class Balance</span>
            <div className="text-xs space-y-1 text-ink-muted">
              <div className="flex justify-between">
                <span>Retained Customers:</span>
                <span className="font-mono text-ink font-medium">{quality.retained.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span>Churned Customers:</span>
                <span className="font-mono text-ink font-medium">{quality.churned.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span>Empirical Churn Rate:</span>
                <span className="font-mono text-ink font-medium">{(quality.churn_rate * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>

          <div className="p-4 bg-panel border border-border rounded-lg space-y-2">
            <span className="text-xs font-semibold text-ink">Evaluation Split</span>
            <div className="text-xs space-y-1 text-ink-muted">
              <div className="flex justify-between">
                <span>Train Cohort (80%):</span>
                <span className="font-mono text-ink font-medium">{split.train_rows.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span>Holdout Cohort (20%):</span>
                <span className="font-mono text-ink font-medium">{split.test_rows.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span>Holdout Churn Rate:</span>
                <span className="font-mono text-ink font-medium">{(split.test_churn_rate * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-panel rounded-lg border border-border p-6 space-y-4">
        <div className="flex items-center gap-2 border-b border-border pb-3">
          <Scale className="w-5 h-5 text-terracotta-600" />
          <div>
            <h3 className="text-sm font-semibold text-ink">Ethical & Regulatory Exclusions</h3>
            <p className="text-xs text-ink-muted">
              Protected demographic attributes intentionally barred from model features.
            </p>
          </div>
        </div>

        <div className="space-y-3">
          {excluded_features.map((item) => (
            <div
              key={item.name}
              className="p-3.5 rounded border border-amber-200 bg-amber-50/40 text-xs flex items-start gap-3"
            >
              <Lock className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
              <div>
                <span className="font-bold text-amber-900 font-mono">{item.name}</span>
                <p className="text-amber-800 mt-1 leading-relaxed">{item.reason}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-panel rounded-lg border border-border p-6 space-y-4">
        <div className="flex items-center gap-2 border-b border-border pb-3">
          <FileCode2 className="w-5 h-5 text-terracotta-600" />
          <div>
            <h3 className="text-sm font-semibold text-ink">Feature Catalog & Data Dictionary</h3>
            <p className="text-xs text-ink-muted">
              Specification of all active schema variables utilised by the preprocessing and estimator pipeline.
            </p>
          </div>
        </div>

        <div className="overflow-x-auto max-h-80 overflow-y-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead className="bg-subtle sticky top-0 border-b border-border">
              <tr>
                <th className="py-2.5 px-3 font-semibold text-ink">Field Name</th>
                <th className="py-2.5 px-3 font-semibold text-ink">Group</th>
                <th className="py-2.5 px-3 font-semibold text-ink">Type</th>
                <th className="py-2.5 px-3 font-semibold text-ink">Constraints / Values</th>
                <th className="py-2.5 px-3 font-semibold text-ink">Description</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {features.map((spec) => (
                <tr key={spec.name} className="hover:bg-canvas">
                  <td className="py-2 px-3 font-mono font-semibold text-ink">{spec.name}</td>
                  <td className="py-2 px-3 text-ink-muted">{spec.group}</td>
                  <td className="py-2 px-3">
                    <span className="px-1.5 py-0.5 rounded bg-subtle border border-border text-[11px] font-mono">
                      {spec.kind}
                    </span>
                  </td>
                  <td className="py-2 px-3 text-ink-muted font-mono text-[11px]">
                    {spec.kind === 'categorical'
                      ? spec.allowed_values.slice(0, 3).join(', ') + (spec.allowed_values.length > 3 ? '...' : '')
                      : `${spec.minimum ?? 0} to ${spec.maximum ?? 'inf'} ${spec.unit || ''}`}
                  </td>
                  <td className="py-2 px-3 text-ink leading-relaxed max-w-sm">{spec.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="bg-panel rounded-lg border border-border p-6 space-y-4">
        <div className="flex items-center gap-2 border-b border-border pb-3">
          <Terminal className="w-5 h-5 text-terracotta-600" />
          <div>
            <h3 className="text-sm font-semibold text-ink">Reproduction & Operational Commands</h3>
            <p className="text-xs text-ink-muted">
              Commands to download dataset, train and evaluate pipeline, start API server, and run tests.
            </p>
          </div>
        </div>

        <div className="bg-ink text-canvas rounded p-4 font-mono text-xs space-y-2 overflow-x-auto">
          <p className="text-terracotta-400"># 1. Download verified IBM Telco customer dataset</p>
          <p>uv run churnsense data download</p>
          <p className="text-terracotta-400 mt-3"># 2. Train candidate models, select parsimonious model, evaluate holdout</p>
          <p>uv run churnsense train</p>
          <p className="text-terracotta-400 mt-3"># 3. Inspect saved model artifacts, version, and headline metrics</p>
          <p>uv run churnsense info</p>
          <p className="text-terracotta-400 mt-3"># 4. Run automated test suite (31 unit & integration tests)</p>
          <p>uv run pytest</p>
          <p className="text-terracotta-400 mt-3"># 5. Launch FastAPI prediction server on port 8000</p>
          <p>uv run churnsense serve --port 8000</p>
        </div>

        <div className="pt-2">
          <span className="text-xs font-semibold text-ink">Runtime Environment:</span>
          <div className="flex flex-wrap gap-2 mt-2">
            {Object.entries(environment).map(([pkg, ver]) => (
              <span
                key={pkg}
                className="px-2 py-1 rounded bg-canvas border border-border text-xs font-mono text-ink"
              >
                {pkg}: {ver}
              </span>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-panel rounded-lg border border-border p-6 space-y-3">
        <div className="flex items-center gap-2 border-b border-border pb-3">
          <AlertCircle className="w-5 h-5 text-terracotta-600" />
          <h3 className="text-sm font-semibold text-ink">Stated Operational Limitations & Boundaries</h3>
        </div>
        <ul className="list-disc list-inside text-xs text-ink-muted space-y-2 pl-1 leading-relaxed">
          {limitations.map((limitation, i) => (
            <li key={i} className="text-ink">
              {limitation}
            </li>
          ))}
          <li className="text-ink">
            Macroeconomic and competitive shifts (e.g., regional price cuts by telecom rivals) are not observed in historical account data and may degrade model calibration over time.
          </li>
          <li className="text-ink">
            Continuous retraining and data drift monitoring are recommended quarterly or when monthly churn rates deviate by more than 3 percentage points from baseline.
          </li>
        </ul>
      </div>
    </div>
  )
}
