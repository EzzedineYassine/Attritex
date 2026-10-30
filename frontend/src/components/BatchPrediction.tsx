import React, { useRef, useState } from 'react'
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  FileSpreadsheet,
  FileText,
  Filter,
  Loader2,
  UploadCloud,
} from 'lucide-react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { downloadBatchTemplate, downloadScoredBatchCsv, predictBatch } from '../api'
import { BatchPredictionResult } from '../types'
import { MetricCard } from './MetricCard'
import { RiskBadge } from './RiskBadge'

export const BatchPrediction: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [downloading, setDownloading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<BatchPredictionResult | null>(null)
  const [tierFilter, setTierFilter] = useState<'all' | 'high' | 'medium' | 'low' | 'invalid'>('all')
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0]
      if (!file.name.toLowerCase().endsWith('.csv')) {
        setError('Only CSV files are accepted.')
        return
      }
      setSelectedFile(file)
      setError(null)
      setResult(null)
    }
  }

  const handleUploadAndScore = async () => {
    if (!selectedFile) return
    setLoading(true)
    setError(null)

    try {
      const data = await predictBatch(selectedFile)
      setResult(data)
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message)
      } else {
        setError('Batch upload and scoring failed.')
      }
    } finally {
      setLoading(false)
    }
  }

  const handleDownloadTemplate = async () => {
    try {
      const blob = await downloadBatchTemplate()
      const url = window.URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = 'attritex_batch_template.csv'
      document.body.appendChild(anchor)
      anchor.click()
      document.body.removeChild(anchor)
      window.URL.revokeObjectURL(url)
    } catch {
      setError('Failed to download batch CSV template.')
    }
  }

  const handleDownloadScoredCsv = async () => {
    if (!selectedFile) return
    setDownloading(true)
    try {
      const blob = await downloadScoredBatchCsv(selectedFile)
      const url = window.URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      const stem = selectedFile.name.replace(/\.csv$/i, '')
      anchor.download = `${stem}_scored.csv`
      document.body.appendChild(anchor)
      anchor.click()
      document.body.removeChild(anchor)
      window.URL.revokeObjectURL(url)
    } catch {
      setError('Failed to download scored batch CSV.')
    } finally {
      setDownloading(false)
    }
  }

  const filteredRows = (result?.rows || []).filter((row) => {
    if (tierFilter === 'all') return true
    if (tierFilter === 'invalid') return row.status === 'invalid'
    return row.risk_tier === tierFilter
  })

  const predictedRatePct = result
    ? ((result.summary.predicted_churn_rate ?? result.summary.mean_probability ?? 0) * 100)
    : 0
  const baselineRatePct = result
    ? ((result.summary.baseline_churn_rate ?? 0.2654) * 100)
    : 26.5
  const rateDelta = predictedRatePct - baselineRatePct

  return (
    <div className="space-y-6">
      <div className="bg-panel rounded-lg border border-border p-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-border pb-4">
          <div>
            <h2 className="text-base font-semibold text-ink">Batch Customer Risk & ARR Analysis</h2>
            <p className="text-xs text-ink-muted mt-0.5">
              Upload a customer portfolio CSV to generate bulk churn probabilities, aggregate ARR at risk, and an exported scored dataset.
            </p>
          </div>
          <button
            type="button"
            onClick={handleDownloadTemplate}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium text-ink bg-subtle hover:bg-border transition-colors border border-border"
          >
            <Download className="w-3.5 h-3.5" />
            Download CSV Template
          </button>
        </div>

        <div className="flex flex-col sm:flex-row items-center gap-4">
          <input
            type="file"
            accept=".csv"
            ref={fileInputRef}
            onChange={handleFileChange}
            className="hidden"
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="w-full sm:w-auto flex-1 border-2 border-dashed border-border hover:border-terracotta-500 rounded-lg p-5 flex flex-col items-center justify-center gap-2 bg-canvas transition-colors cursor-pointer text-center"
          >
            <UploadCloud className="w-6 h-6 text-terracotta-600" />
            <div className="text-xs">
              <span className="font-semibold text-ink">
                {selectedFile ? selectedFile.name : 'Click to select customer CSV'}
              </span>
              <p className="text-ink-muted mt-0.5">
                {selectedFile
                  ? `${(selectedFile.size / 1024).toFixed(1)} KB`
                  : 'Up to 50,000 customer records (5MB limit)'}
              </p>
            </div>
          </button>

          <button
            type="button"
            disabled={!selectedFile || loading}
            onClick={handleUploadAndScore}
            className="w-full sm:w-auto px-6 py-4 rounded-lg bg-terracotta-600 hover:bg-terracotta-700 text-white font-semibold text-sm transition-colors flex items-center justify-center gap-2 shadow-sm disabled:opacity-50"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Scoring Customers...
              </>
            ) : (
              <>
                <FileSpreadsheet className="w-4 h-4" />
                Process & Score Batch
              </>
            )}
          </button>
        </div>

        {error && (
          <div className="p-3 bg-red-50 border border-red-200 rounded text-xs text-red-800 flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {result && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <MetricCard
              label="Total Records Scored"
              value={result.summary.scored_rows.toLocaleString()}
              subtext={`${result.summary.input_rows.toLocaleString()} uploaded (${result.summary.invalid_rows.toLocaleString()} invalid) · ${result.summary.flagged_rows.toLocaleString()} flagged`}
            />
            <MetricCard
              label="Predicted Churn Rate (vs 26.5% Baseline)"
              value={`${predictedRatePct.toFixed(1)}%`}
              subtext={`${rateDelta >= 0 ? '+' : ''}${rateDelta.toFixed(1)} pts vs ${baselineRatePct.toFixed(1)}% empirical baseline (${result.summary.expected_churners.toFixed(1)} expected churners)`}
              variant={rateDelta > 0 ? 'danger' : 'highlight'}
            />
            <MetricCard
              label="Total Annual Recurring Revenue (ARR) at Risk"
              value={`$${(result.summary.total_arr_at_risk ?? 0).toLocaleString(undefined, {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
              })}`}
              subtext="Sum of (MonthlyCharges × 12 × Churn Probability)"
              variant="highlight"
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-8 bg-panel rounded-lg border border-border p-5 space-y-3">
              <div className="flex items-center justify-between border-b border-border pb-3">
                <div>
                  <h3 className="text-sm font-semibold text-ink">Portfolio Risk Distribution</h3>
                  <p className="text-xs text-ink-muted mt-0.5">
                    Histogram of model-assigned churn probabilities across scored customers.
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-mono text-ink-muted">
                    Decision Threshold: {(result.decision_threshold * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              <div className="h-64 w-full pt-2">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={result.summary.histogram}
                    margin={{ top: 10, right: 10, left: -20, bottom: 20 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#E3DDD4" vertical={false} />
                    <XAxis
                      dataKey="lower"
                      tickFormatter={(val) => `${(val * 100).toFixed(0)}%`}
                      stroke="#8D99A6"
                      fontSize={11}
                    />
                    <YAxis stroke="#8D99A6" fontSize={11} allowDecimals={false} />
                    <Tooltip
                      formatter={(value: unknown) => [`${Number(value || 0)} accounts`, 'Customer Count']}
                      labelFormatter={(label) =>
                        `Score: ${(Number(label) * 100).toFixed(0)}% - ${(
                          (Number(label) + 0.05) *
                          100
                        ).toFixed(0)}%`
                      }
                      contentStyle={{
                        backgroundColor: '#1B242D',
                        borderRadius: '6px',
                        border: 'none',
                        color: '#F8F6F0',
                        fontSize: '12px',
                      }}
                    />
                    <Bar dataKey="count" fill="#D65A31" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="lg:col-span-4 bg-panel rounded-lg border border-border p-5 flex flex-col justify-between">
              <div>
                <h3 className="text-sm font-semibold text-ink border-b border-border pb-3">
                  Tier Allocations
                </h3>
                <div className="space-y-3 mt-4">
                  <div className="p-3 rounded border border-red-200 bg-red-50/50 flex items-center justify-between">
                    <div>
                      <span className="text-xs font-bold text-red-800">High Risk Tier</span>
                      <p className="text-[11px] text-red-700">
                        &gt;= {(result.tier_cutoffs.high * 100).toFixed(1)}%
                      </p>
                    </div>
                    <span className="text-xl font-bold font-mono text-red-800">
                      {result.summary.tier_counts.high.toLocaleString()}
                    </span>
                  </div>

                  <div className="p-3 rounded border border-amber-200 bg-amber-50/50 flex items-center justify-between">
                    <div>
                      <span className="text-xs font-bold text-amber-800">Medium Risk Tier</span>
                      <p className="text-[11px] text-amber-700">
                        {(result.tier_cutoffs.medium * 100).toFixed(1)}% -{' '}
                        {(result.tier_cutoffs.high * 100).toFixed(1)}%
                      </p>
                    </div>
                    <span className="text-xl font-bold font-mono text-amber-800">
                      {result.summary.tier_counts.medium.toLocaleString()}
                    </span>
                  </div>

                  <div className="p-3 rounded border border-emerald-200 bg-emerald-50/50 flex items-center justify-between">
                    <div>
                      <span className="text-xs font-bold text-emerald-800">Low Risk Tier</span>
                      <p className="text-[11px] text-emerald-700">
                        &lt; {(result.tier_cutoffs.medium * 100).toFixed(1)}%
                      </p>
                    </div>
                    <span className="text-xl font-bold font-mono text-emerald-800">
                      {result.summary.tier_counts.low.toLocaleString()}
                    </span>
                  </div>
                </div>
              </div>

              <div className="pt-4 border-t border-border mt-4">
                <button
                  type="button"
                  disabled={downloading}
                  onClick={handleDownloadScoredCsv}
                  className="w-full py-2.5 px-4 rounded bg-ink hover:bg-ink/90 text-canvas text-xs font-semibold flex items-center justify-center gap-2 transition-colors disabled:opacity-50"
                >
                  {downloading ? (
                    <>
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      Generating File...
                    </>
                  ) : (
                    <>
                      <Download className="w-3.5 h-3.5" />
                      Download Scored CSV
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          <div className="bg-panel rounded-lg border border-border p-5 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-border pb-3">
              <div className="flex items-center gap-2">
                <FileText className="w-4 h-4 text-terracotta-600" />
                <h3 className="text-sm font-semibold text-ink">Customer Account Scorecard</h3>
                <span className="text-xs text-ink-muted">({filteredRows.length} shown)</span>
              </div>

              <div className="flex items-center gap-1.5 text-xs">
                <Filter className="w-3.5 h-3.5 text-ink-muted" />
                <span className="text-ink-muted mr-1">Filter:</span>
                {(['all', 'high', 'medium', 'low', 'invalid'] as const).map((filter) => (
                  <button
                    key={filter}
                    onClick={() => setTierFilter(filter)}
                    className={`px-2 py-0.5 rounded capitalize ${
                      tierFilter === filter
                        ? 'bg-ink text-canvas font-medium'
                        : 'text-ink-muted hover:text-ink bg-subtle'
                    }`}
                  >
                    {filter}
                  </button>
                ))}
              </div>
            </div>

            <div className="overflow-x-auto max-h-96 overflow-y-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead className="bg-subtle sticky top-0 border-b border-border">
                  <tr>
                    <th className="py-2.5 px-3 font-semibold text-ink">Row #</th>
                    <th className="py-2.5 px-3 font-semibold text-ink">Customer ID</th>
                    <th className="py-2.5 px-3 font-semibold text-ink">Status</th>
                    <th className="py-2.5 px-3 font-semibold text-ink">Probability</th>
                    <th className="py-2.5 px-3 font-semibold text-ink">ARR at Risk</th>
                    <th className="py-2.5 px-3 font-semibold text-ink">Risk Tier</th>
                    <th className="py-2.5 px-3 font-semibold text-ink">Action Flag</th>
                    <th className="py-2.5 px-3 font-semibold text-ink">Validation Notes</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {filteredRows.length === 0 ? (
                    <tr>
                      <td colSpan={8} className="py-8 text-center text-ink-muted italic">
                        No customer rows match the selected filter.
                      </td>
                    </tr>
                  ) : (
                    filteredRows.map((row) => (
                      <tr key={row.row} className="hover:bg-canvas transition-colors">
                        <td className="py-2 px-3 font-mono text-ink-muted">{row.row}</td>
                        <td className="py-2 px-3 font-medium text-ink font-mono">
                          {row.customer_id || '—'}
                        </td>
                        <td className="py-2 px-3">
                          {row.status === 'scored' ? (
                            <span className="inline-flex items-center gap-1 text-emerald-700 text-[11px] font-medium">
                              <CheckCircle2 className="w-3 h-3" /> Scored
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 text-red-700 text-[11px] font-medium">
                              <AlertTriangle className="w-3 h-3" /> Invalid
                            </span>
                          )}
                        </td>
                        <td className="py-2 px-3 font-mono font-bold text-ink">
                          {row.churn_probability !== null
                            ? `${(row.churn_probability * 100).toFixed(1)}%`
                            : '—'}
                        </td>
                        <td className="py-2 px-3 font-mono text-ink">
                          {row.expected_revenue_at_risk !== null &&
                          row.expected_revenue_at_risk !== undefined
                            ? `$${row.expected_revenue_at_risk.toLocaleString(undefined, {
                                minimumFractionDigits: 2,
                                maximumFractionDigits: 2,
                              })}`
                            : '—'}
                        </td>
                        <td className="py-2 px-3">
                          {row.risk_tier ? <RiskBadge tier={row.risk_tier} /> : '—'}
                        </td>
                        <td className="py-2 px-3">
                          {row.flagged ? (
                            <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-red-100 text-red-800 border border-red-200">
                              FLAGGED
                            </span>
                          ) : (
                            <span className="text-ink-muted text-[11px]">—</span>
                          )}
                        </td>
                        <td className="py-2 px-3 text-ink-muted max-w-xs truncate" title={row.issues}>
                          {row.issues ? (
                            <span className="text-red-700 font-mono text-[11px]">{row.issues}</span>
                          ) : row.imputed_fields.length > 0 ? (
                            <span className="text-ink-muted italic text-[11px]">
                              Imputed: {row.imputed_fields.join(', ')}
                            </span>
                          ) : (
                            <span className="text-emerald-700 text-[11px]">Valid</span>
                          )}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
