import React, { useState } from 'react'
import {
  AlertCircle,
  ArrowDownRight,
  ArrowUpRight,
  CheckCircle,
  DollarSign,
  Info,
  RotateCcw,
  Sparkles,
} from 'lucide-react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { predictSingle } from '../api'
import { FeatureSpec, PredictionResult } from '../types'
import { RiskBadge } from './RiskBadge'

interface SinglePredictionProps {
  features: FeatureSpec[]
  modelLoaded: boolean
}

const PRESETS = {
  highRisk: {
    customerID: 'HIGH-RISK-01',
    SeniorCitizen: '0',
    Partner: 'No',
    Dependents: 'No',
    tenure: 2,
    PhoneService: 'Yes',
    MultipleLines: 'No',
    InternetService: 'Fiber optic',
    OnlineSecurity: 'No',
    OnlineBackup: 'No',
    DeviceProtection: 'No',
    TechSupport: 'No',
    StreamingTV: 'Yes',
    StreamingMovies: 'Yes',
    Contract: 'Month-to-month',
    PaperlessBilling: 'Yes',
    PaymentMethod: 'Electronic check',
    MonthlyCharges: 89.5,
    TotalCharges: 179.0,
  },
  loyal: {
    customerID: 'LOYAL-01',
    SeniorCitizen: '0',
    Partner: 'Yes',
    Dependents: 'Yes',
    tenure: 48,
    PhoneService: 'Yes',
    MultipleLines: 'Yes',
    InternetService: 'DSL',
    OnlineSecurity: 'Yes',
    OnlineBackup: 'Yes',
    DeviceProtection: 'Yes',
    TechSupport: 'Yes',
    StreamingTV: 'No',
    StreamingMovies: 'No',
    Contract: 'Two year',
    PaperlessBilling: 'No',
    PaymentMethod: 'Bank transfer (automatic)',
    MonthlyCharges: 62.0,
    TotalCharges: 2976.0,
  },
  average: {
    customerID: 'SAMPLE-01',
    SeniorCitizen: '0',
    Partner: 'No',
    Dependents: 'No',
    tenure: 24,
    PhoneService: 'Yes',
    MultipleLines: 'No',
    InternetService: 'DSL',
    OnlineSecurity: 'No',
    OnlineBackup: 'Yes',
    DeviceProtection: 'No',
    TechSupport: 'No',
    StreamingTV: 'No',
    StreamingMovies: 'No',
    Contract: 'One year',
    PaperlessBilling: 'Yes',
    PaymentMethod: 'Credit card (automatic)',
    MonthlyCharges: 50.0,
    TotalCharges: 1200.0,
  },
}

export const SinglePrediction: React.FC<SinglePredictionProps> = ({ modelLoaded }) => {
  const [formData, setFormData] = useState<Record<string, unknown>>(PRESETS.highRisk)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({})

  const handleInputChange = (name: string, value: unknown) => {
    setFormData((prev) => {
      const next = { ...prev, [name]: value }

      if (name === 'InternetService') {
        if (value === 'No') {
          next.OnlineSecurity = 'No internet service'
          next.OnlineBackup = 'No internet service'
          next.DeviceProtection = 'No internet service'
          next.TechSupport = 'No internet service'
          next.StreamingTV = 'No internet service'
          next.StreamingMovies = 'No internet service'
        } else if (prev.InternetService === 'No') {
          next.OnlineSecurity = 'No'
          next.OnlineBackup = 'No'
          next.DeviceProtection = 'No'
          next.TechSupport = 'No'
          next.StreamingTV = 'No'
          next.StreamingMovies = 'No'
        }
      }

      if (name === 'PhoneService') {
        if (value === 'No') {
          next.MultipleLines = 'No phone service'
        } else if (prev.PhoneService === 'No') {
          next.MultipleLines = 'No'
        }
      }

      if (name === 'tenure' || name === 'MonthlyCharges') {
        const tenureVal = Number(name === 'tenure' ? value : prev.tenure) || 0
        const monthlyVal = Number(name === 'MonthlyCharges' ? value : prev.MonthlyCharges) || 0
        next.TotalCharges = Math.round(tenureVal * monthlyVal * 100) / 100
      }

      return next
    })

    if (fieldErrors[name]) {
      setFieldErrors((prev) => {
        const updated = { ...prev }
        delete updated[name]
        return updated
      })
    }
  }

  const loadPreset = (presetKey: keyof typeof PRESETS) => {
    setFormData(PRESETS[presetKey])
    setFieldErrors({})
    setError(null)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!modelLoaded) {
      setError('Model artifact is not loaded in backend.')
      return
    }

    setLoading(true)
    setError(null)
    setFieldErrors({})

    try {
      const prediction = await predictSingle(formData)
      setResult(prediction)
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message)
        const apiErr = err as { details?: Array<{ field?: string; message: string }> }
        if (apiErr.details && Array.isArray(apiErr.details)) {
          const map: Record<string, string> = {}
          apiErr.details.forEach((d) => {
            if (d.field) map[d.field] = d.message
          })
          setFieldErrors(map)
        }
      } else {
        setError('An unexpected error occurred during prediction.')
      }
    } finally {
      setLoading(false)
    }
  }

  const isInternetNo = formData.InternetService === 'No'
  const isPhoneNo = formData.PhoneService === 'No'

  const attributionDrivers = result
    ? [
        ...(result.local_attribution?.top_risk_increasing || []),
        ...(result.local_attribution?.top_risk_reducing || []),
      ]
    : []

  const chartDrivers = (attributionDrivers.length > 0 ? attributionDrivers : result?.factors || []).map(
    (item) => ({
      name: item.label,
      effectPct: Number((item.effect * 100).toFixed(2)),
      customerValue: item.customer_value,
      referenceValue: item.reference_value,
    })
  )

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 bg-panel p-4 rounded-lg border border-border">
        <div>
          <h2 className="text-base font-semibold text-ink">Customer Risk Profiler</h2>
          <p className="text-xs text-ink-muted mt-0.5">
            Evaluate churn probability, annual revenue at risk, and local feature attribution for an account.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-ink-muted">Quick Presets:</span>
          <button
            type="button"
            onClick={() => loadPreset('highRisk')}
            className="text-xs px-2.5 py-1 rounded bg-red-50 text-red-700 border border-red-200 hover:bg-red-100 transition-colors font-medium"
          >
            High Risk
          </button>
          <button
            type="button"
            onClick={() => loadPreset('loyal')}
            className="text-xs px-2.5 py-1 rounded bg-emerald-50 text-emerald-800 border border-emerald-200 hover:bg-emerald-100 transition-colors font-medium"
          >
            Loyal Customer
          </button>
          <button
            type="button"
            onClick={() => loadPreset('average')}
            className="text-xs px-2.5 py-1 rounded bg-subtle text-ink hover:bg-border transition-colors font-medium"
          >
            Typical User
          </button>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-panel rounded-lg border border-border p-5 space-y-4">
            <div className="border-b border-border pb-2.5">
              <h3 className="text-sm font-semibold text-ink uppercase tracking-wide">
                1. Account & Contract
              </h3>
            </div>
            <div className="space-y-3.5">
              <div>
                <label className="block text-xs font-medium text-ink mb-1">Customer Identifier</label>
                <input
                  type="text"
                  value={String(formData.customerID || '')}
                  onChange={(e) => handleInputChange('customerID', e.target.value)}
                  placeholder="e.g. CUST-90812"
                  className="w-full px-3 py-1.5 text-sm rounded border border-border bg-canvas focus:outline-none focus:border-terracotta-500"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-ink mb-1">Contract Term</label>
                <select
                  value={String(formData.Contract || 'Month-to-month')}
                  onChange={(e) => handleInputChange('Contract', e.target.value)}
                  className="w-full px-3 py-1.5 text-sm rounded border border-border bg-canvas focus:outline-none focus:border-terracotta-500"
                >
                  <option value="Month-to-month">Month-to-month</option>
                  <option value="One year">One year</option>
                  <option value="Two year">Two year</option>
                </select>
                {fieldErrors.Contract && (
                  <p className="text-xs text-red-600 mt-1">{fieldErrors.Contract}</p>
                )}
              </div>

              <div>
                <label className="block text-xs font-medium text-ink mb-1">
                  Tenure <span className="text-ink-muted">(0 - 120 months)</span>
                </label>
                <input
                  type="number"
                  min="0"
                  max="120"
                  value={Number(formData.tenure ?? 0)}
                  onChange={(e) => handleInputChange('tenure', parseInt(e.target.value, 10) || 0)}
                  className="w-full px-3 py-1.5 text-sm rounded border border-border bg-canvas focus:outline-none focus:border-terracotta-500 font-mono"
                />
                {fieldErrors.tenure && (
                  <p className="text-xs text-red-600 mt-1">{fieldErrors.tenure}</p>
                )}
              </div>

              <div>
                <label className="block text-xs font-medium text-ink mb-1">Payment Method</label>
                <select
                  value={String(formData.PaymentMethod || 'Electronic check')}
                  onChange={(e) => handleInputChange('PaymentMethod', e.target.value)}
                  className="w-full px-3 py-1.5 text-sm rounded border border-border bg-canvas focus:outline-none focus:border-terracotta-500"
                >
                  <option value="Electronic check">Electronic check</option>
                  <option value="Mailed check">Mailed check</option>
                  <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
                  <option value="Credit card (automatic)">Credit card (automatic)</option>
                </select>
                {fieldErrors.PaymentMethod && (
                  <p className="text-xs text-red-600 mt-1">{fieldErrors.PaymentMethod}</p>
                )}
              </div>

              <div>
                <label className="block text-xs font-medium text-ink mb-1">Paperless Billing</label>
                <select
                  value={String(formData.PaperlessBilling || 'Yes')}
                  onChange={(e) => handleInputChange('PaperlessBilling', e.target.value)}
                  className="w-full px-3 py-1.5 text-sm rounded border border-border bg-canvas focus:outline-none focus:border-terracotta-500"
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>
            </div>
          </div>

          <div className="bg-panel rounded-lg border border-border p-5 space-y-4">
            <div className="border-b border-border pb-2.5">
              <h3 className="text-sm font-semibold text-ink uppercase tracking-wide">
                2. Services
              </h3>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-medium text-ink mb-1">Phone Service</label>
                <select
                  value={String(formData.PhoneService || 'Yes')}
                  onChange={(e) => handleInputChange('PhoneService', e.target.value)}
                  className="w-full px-2.5 py-1.5 text-sm rounded border border-border bg-canvas focus:outline-none focus:border-terracotta-500"
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-ink mb-1">Multiple Lines</label>
                <select
                  disabled={isPhoneNo}
                  value={String(formData.MultipleLines || (isPhoneNo ? 'No phone service' : 'No'))}
                  onChange={(e) => handleInputChange('MultipleLines', e.target.value)}
                  className="w-full px-2.5 py-1.5 text-sm rounded border border-border bg-canvas focus:outline-none focus:border-terracotta-500 disabled:opacity-50"
                >
                  {isPhoneNo ? (
                    <option value="No phone service">No phone service</option>
                  ) : (
                    <>
                      <option value="No">No</option>
                      <option value="Yes">Yes</option>
                    </>
                  )}
                </select>
              </div>

              <div className="sm:col-span-2">
                <label className="block text-xs font-medium text-ink mb-1">Internet Service</label>
                <select
                  value={String(formData.InternetService || 'DSL')}
                  onChange={(e) => handleInputChange('InternetService', e.target.value)}
                  className="w-full px-2.5 py-1.5 text-sm rounded border border-border bg-canvas focus:outline-none focus:border-terracotta-500"
                >
                  <option value="DSL">DSL</option>
                  <option value="Fiber optic">Fiber optic</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-ink mb-1">Online Security</label>
                <select
                  disabled={isInternetNo}
                  value={String(formData.OnlineSecurity || (isInternetNo ? 'No internet service' : 'No'))}
                  onChange={(e) => handleInputChange('OnlineSecurity', e.target.value)}
                  className="w-full px-2.5 py-1.5 text-sm rounded border border-border bg-canvas focus:outline-none focus:border-terracotta-500 disabled:opacity-50"
                >
                  {isInternetNo ? (
                    <option value="No internet service">No internet service</option>
                  ) : (
                    <>
                      <option value="No">No</option>
                      <option value="Yes">Yes</option>
                    </>
                  )}
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-ink mb-1">Online Backup</label>
                <select
                  disabled={isInternetNo}
                  value={String(formData.OnlineBackup || (isInternetNo ? 'No internet service' : 'No'))}
                  onChange={(e) => handleInputChange('OnlineBackup', e.target.value)}
                  className="w-full px-2.5 py-1.5 text-sm rounded border border-border bg-canvas focus:outline-none focus:border-terracotta-500 disabled:opacity-50"
                >
                  {isInternetNo ? (
                    <option value="No internet service">No internet service</option>
                  ) : (
                    <>
                      <option value="No">No</option>
                      <option value="Yes">Yes</option>
                    </>
                  )}
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-ink mb-1">Device Protection</label>
                <select
                  disabled={isInternetNo}
                  value={String(formData.DeviceProtection || (isInternetNo ? 'No internet service' : 'No'))}
                  onChange={(e) => handleInputChange('DeviceProtection', e.target.value)}
                  className="w-full px-2.5 py-1.5 text-sm rounded border border-border bg-canvas focus:outline-none focus:border-terracotta-500 disabled:opacity-50"
                >
                  {isInternetNo ? (
                    <option value="No internet service">No internet service</option>
                  ) : (
                    <>
                      <option value="No">No</option>
                      <option value="Yes">Yes</option>
                    </>
                  )}
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-ink mb-1">Tech Support</label>
                <select
                  disabled={isInternetNo}
                  value={String(formData.TechSupport || (isInternetNo ? 'No internet service' : 'No'))}
                  onChange={(e) => handleInputChange('TechSupport', e.target.value)}
                  className="w-full px-2.5 py-1.5 text-sm rounded border border-border bg-canvas focus:outline-none focus:border-terracotta-500 disabled:opacity-50"
                >
                  {isInternetNo ? (
                    <option value="No internet service">No internet service</option>
                  ) : (
                    <>
                      <option value="No">No</option>
                      <option value="Yes">Yes</option>
                    </>
                  )}
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-ink mb-1">Streaming TV</label>
                <select
                  disabled={isInternetNo}
                  value={String(formData.StreamingTV || (isInternetNo ? 'No internet service' : 'No'))}
                  onChange={(e) => handleInputChange('StreamingTV', e.target.value)}
                  className="w-full px-2.5 py-1.5 text-sm rounded border border-border bg-canvas focus:outline-none focus:border-terracotta-500 disabled:opacity-50"
                >
                  {isInternetNo ? (
                    <option value="No internet service">No internet service</option>
                  ) : (
                    <>
                      <option value="No">No</option>
                      <option value="Yes">Yes</option>
                    </>
                  )}
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-ink mb-1">Streaming Movies</label>
                <select
                  disabled={isInternetNo}
                  value={String(formData.StreamingMovies || (isInternetNo ? 'No internet service' : 'No'))}
                  onChange={(e) => handleInputChange('StreamingMovies', e.target.value)}
                  className="w-full px-2.5 py-1.5 text-sm rounded border border-border bg-canvas focus:outline-none focus:border-terracotta-500 disabled:opacity-50"
                >
                  {isInternetNo ? (
                    <option value="No internet service">No internet service</option>
                  ) : (
                    <>
                      <option value="No">No</option>
                      <option value="Yes">Yes</option>
                    </>
                  )}
                </select>
              </div>
            </div>
          </div>

          <div className="bg-panel rounded-lg border border-border p-5 space-y-4">
            <div className="border-b border-border pb-2.5">
              <h3 className="text-sm font-semibold text-ink uppercase tracking-wide">
                3. Demographics & Billing
              </h3>
            </div>
            <div className="space-y-3.5">
              <div>
                <label className="block text-xs font-medium text-ink mb-1">Senior Citizen (65+)</label>
                <select
                  value={String(formData.SeniorCitizen || '0')}
                  onChange={(e) => handleInputChange('SeniorCitizen', e.target.value)}
                  className="w-full px-3 py-1.5 text-sm rounded border border-border bg-canvas focus:outline-none focus:border-terracotta-500"
                >
                  <option value="0">No</option>
                  <option value="1">Yes</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-ink mb-1">Has Partner</label>
                <select
                  value={String(formData.Partner || 'No')}
                  onChange={(e) => handleInputChange('Partner', e.target.value)}
                  className="w-full px-3 py-1.5 text-sm rounded border border-border bg-canvas focus:outline-none focus:border-terracotta-500"
                >
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-ink mb-1">Has Dependents</label>
                <select
                  value={String(formData.Dependents || 'No')}
                  onChange={(e) => handleInputChange('Dependents', e.target.value)}
                  className="w-full px-3 py-1.5 text-sm rounded border border-border bg-canvas focus:outline-none focus:border-terracotta-500"
                >
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-ink mb-1">
                  Monthly Charges <span className="text-ink-muted">(USD)</span>
                </label>
                <div className="relative">
                  <span className="absolute left-3 top-1.5 text-ink-muted text-sm">$</span>
                  <input
                    type="number"
                    step="0.05"
                    min="0"
                    max="1000"
                    value={Number(formData.MonthlyCharges ?? 0)}
                    onChange={(e) => handleInputChange('MonthlyCharges', parseFloat(e.target.value) || 0)}
                    className="w-full pl-7 pr-3 py-1.5 text-sm rounded border border-border bg-canvas focus:outline-none focus:border-terracotta-500 font-mono"
                  />
                </div>
                {fieldErrors.MonthlyCharges && (
                  <p className="text-xs text-red-600 mt-1">{fieldErrors.MonthlyCharges}</p>
                )}
              </div>

              <div>
                <label className="block text-xs font-medium text-ink mb-1">
                  Total Charges to Date <span className="text-ink-muted">(USD)</span>
                </label>
                <div className="relative">
                  <span className="absolute left-3 top-1.5 text-ink-muted text-sm">$</span>
                  <input
                    type="number"
                    step="0.05"
                    min="0"
                    max="100000"
                    value={Number(formData.TotalCharges ?? 0)}
                    onChange={(e) => handleInputChange('TotalCharges', parseFloat(e.target.value) || 0)}
                    className="w-full pl-7 pr-3 py-1.5 text-sm rounded border border-border bg-canvas focus:outline-none focus:border-terracotta-500 font-mono"
                  />
                </div>
                {fieldErrors.TotalCharges && (
                  <p className="text-xs text-red-600 mt-1">{fieldErrors.TotalCharges}</p>
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="flex items-center justify-between">
          <button
            type="button"
            onClick={() => {
              setFormData(PRESETS.average)
              setResult(null)
              setError(null)
              setFieldErrors({})
            }}
            className="inline-flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink transition-colors px-3 py-2 rounded border border-border bg-panel"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset Form
          </button>

          <button
            type="submit"
            disabled={loading}
            className="inline-flex items-center gap-2 px-6 py-2.5 rounded text-sm font-semibold bg-terracotta-600 text-white hover:bg-terracotta-700 transition-colors shadow-sm disabled:opacity-50"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Calculating Risk...
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                Assess Customer Risk
              </>
            )}
          </button>
        </div>
      </form>

      {error && (
        <div className="p-4 rounded-lg bg-red-50 border border-red-200 text-red-800 text-sm flex items-start gap-2.5">
          <AlertCircle className="w-4 h-4 text-red-600 mt-0.5 shrink-0" />
          <div>
            <p className="font-semibold">Assessment Request Failed</p>
            <p className="text-xs mt-1 text-red-700 leading-relaxed">{error}</p>
          </div>
        </div>
      )}

      {result ? (
        <div className="bg-panel rounded-lg border border-border overflow-hidden shadow-sm p-6 space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-border pb-4">
            <div>
              <div className="flex items-center gap-2.5">
                <h3 className="text-base font-bold text-ink">Risk Assessment & Attribution Report</h3>
                <RiskBadge tier={result.risk_tier} />
              </div>
              <p className="text-xs text-ink-muted mt-1 font-mono">
                ID: {result.customer_id || 'Unassigned'} | Model: {result.model_version.split('-')[1]}
              </p>
            </div>
            {result.flagged ? (
              <span className="px-2.5 py-1 text-xs font-semibold bg-red-100 text-red-800 rounded border border-red-300 self-start sm:self-auto">
                RETENTION ACTION FLAG
              </span>
            ) : (
              <span className="px-2.5 py-1 text-xs font-semibold bg-emerald-100 text-emerald-800 rounded border border-emerald-300 self-start sm:self-auto">
                STABLE ACCOUNT
              </span>
            )}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-5 space-y-4">
              <div className="bg-canvas rounded-lg p-5 border border-border space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <span className="text-xs uppercase font-medium text-ink-muted tracking-wider block">
                      Churn Probability
                    </span>
                    <span className="text-3xl font-extrabold font-mono tracking-tight text-ink mt-1 block">
                      {(result.churn_probability * 100).toFixed(1)}%
                    </span>
                    <span className="text-[11px] font-mono text-ink-muted">
                      Cutoff: {(result.decision_threshold * 100).toFixed(1)}%
                    </span>
                  </div>

                  <div className="border-l border-border pl-4">
                    <span className="text-xs uppercase font-medium text-ink-muted tracking-wider flex items-center gap-1">
                      <DollarSign className="w-3.5 h-3.5 text-terracotta-600" />
                      Annual Revenue at Risk
                    </span>
                    <span className="text-2xl font-extrabold font-mono tracking-tight text-terracotta-700 mt-1 block">
                      $
                      {result.expected_revenue_at_risk.toLocaleString(undefined, {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2,
                      })}
                    </span>
                    <span className="text-[11px] text-ink-muted">
                      Monthly × 12 × Churn Prob
                    </span>
                  </div>
                </div>

                <div>
                  <div className="w-full h-3 bg-border rounded-full overflow-hidden flex">
                    <div
                      className="h-full bg-emerald-500"
                      style={{ width: `${result.tier_cutoffs.medium * 100}%` }}
                    />
                    <div
                      className="h-full bg-amber-500"
                      style={{
                        width: `${(result.tier_cutoffs.high - result.tier_cutoffs.medium) * 100}%`,
                      }}
                    />
                    <div
                      className="h-full bg-red-500"
                      style={{ width: `${(1 - result.tier_cutoffs.high) * 100}%` }}
                    />
                  </div>
                  <div className="relative w-full h-2">
                    <div
                      className="absolute -top-1 w-2.5 h-2.5 bg-ink border-2 border-white rounded-full -ml-1 transition-all"
                      style={{ left: `${Math.min(result.churn_probability * 100, 100)}%` }}
                    />
                  </div>
                </div>

                {result.tier_context && (
                  <p className="text-xs text-ink-muted pt-3 border-t border-border/60 leading-relaxed">
                    <strong>Holdout Benchmark:</strong> Accounts in the {result.tier_context.tier} tier
                    exhibited an empirical churn rate of{' '}
                    <strong>{(result.tier_context.observed_churn_rate * 100).toFixed(1)}%</strong> across{' '}
                    {result.tier_context.customers.toLocaleString()} test customers.
                  </p>
                )}
              </div>

              {result.flagged && (
                <div className="p-4 bg-terracotta-50 border border-terracotta-200 rounded-lg text-xs space-y-2">
                  <div className="flex items-center gap-1.5 text-terracotta-700 font-semibold">
                    <CheckCircle className="w-3.5 h-3.5" />
                    Recommended Retention Interventions
                  </div>
                  <ul className="list-disc list-inside text-ink space-y-1 pl-1">
                    {formData.Contract === 'Month-to-month' && (
                      <li>Propose discounted 1-year contract extension (+15-20% retention lift).</li>
                    )}
                    {formData.TechSupport === 'No' && formData.InternetService !== 'No' && (
                      <li>Offer complimentary 3-month Premium Tech Support trial.</li>
                    )}
                    {formData.PaymentMethod === 'Electronic check' && (
                      <li>Incentivize automatic billing setup with a one-time bill credit.</li>
                    )}
                    {Number(formData.tenure) < 6 && (
                      <li>Schedule proactive customer onboarding check-in call.</li>
                    )}
                  </ul>
                </div>
              )}
            </div>

            <div className="lg:col-span-7 bg-canvas rounded-lg p-5 border border-border flex flex-col justify-between space-y-4">
              <div>
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <h4 className="text-xs font-semibold text-ink uppercase tracking-wider">
                      Local Feature Attribution (Top 3 Risk-Increasing & Top 2 Risk-Reducing)
                    </h4>
                    <p className="text-[11px] text-ink-muted mt-0.5">
                      Linear coefficient attribution relative to the empirical training median profile.
                    </p>
                  </div>
                </div>

                {chartDrivers.length === 0 ? (
                  <p className="text-xs text-ink-muted italic p-4 bg-panel rounded border border-border">
                    This customer record matches the baseline median profile with no material probability deviation.
                  </p>
                ) : (
                  <div className="space-y-4">
                    <div className="h-44 w-full bg-panel rounded border border-border p-2">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={chartDrivers}
                          layout="vertical"
                          margin={{ top: 5, right: 25, left: 95, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#E3DDD4" horizontal={false} />
                          <XAxis
                            type="number"
                            fontSize={11}
                            stroke="#8D99A6"
                            tickFormatter={(v) => `${v > 0 ? '+' : ''}${v}%`}
                          />
                          <YAxis
                            type="category"
                            dataKey="name"
                            fontSize={11}
                            stroke="#1B242D"
                            width={90}
                          />
                          <Tooltip
                            formatter={(val: unknown) => [
                              `${Number(val) > 0 ? '+' : ''}${Number(val).toFixed(1)}%`,
                              'Probability Shift',
                            ]}
                            contentStyle={{
                              backgroundColor: '#1B242D',
                              borderRadius: '6px',
                              border: 'none',
                              color: '#F8F6F0',
                              fontSize: '12px',
                            }}
                          />
                          <ReferenceLine x={0} stroke="#1B242D" strokeWidth={1} />
                          <Bar dataKey="effectPct" radius={[0, 3, 3, 0]}>
                            {chartDrivers.map((entry, idx) => (
                              <Cell
                                key={idx}
                                fill={entry.effectPct >= 0 ? '#DC2626' : '#15803D'}
                              />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      {(attributionDrivers.length > 0 ? attributionDrivers : result.factors).map(
                        (factor) => {
                          const isRiskIncrease = factor.effect > 0
                          return (
                            <div
                              key={factor.feature}
                              className="p-2.5 rounded border border-border bg-panel flex items-center justify-between text-xs"
                            >
                              <div className="space-y-0.5 pr-2">
                                <span className="font-semibold text-ink">{factor.label}</span>
                                <div className="text-ink-muted text-[11px]">
                                  <span className="font-mono text-ink font-medium">
                                    {factor.customer_value}
                                  </span>{' '}
                                  <span className="text-ink-faint">(Med: {factor.reference_value})</span>
                                </div>
                              </div>
                              <div
                                className={`flex items-center gap-0.5 font-mono font-bold shrink-0 ${
                                  isRiskIncrease ? 'text-red-700' : 'text-emerald-700'
                                }`}
                              >
                                {isRiskIncrease ? (
                                  <ArrowUpRight className="w-3.5 h-3.5" />
                                ) : (
                                  <ArrowDownRight className="w-3.5 h-3.5" />
                                )}
                                {isRiskIncrease ? '+' : ''}
                                {(factor.effect * 100).toFixed(1)}%
                              </div>
                            </div>
                          )
                        }
                      )}
                    </div>
                  </div>
                )}
              </div>

              <div className="pt-3 border-t border-border flex items-start gap-2 text-[11px] text-ink-muted">
                <Info className="w-3.5 h-3.5 text-ink-muted shrink-0 mt-0.5" />
                <p>
                  Local attribution describes relative probability shift; it does not establish individual causation.
                </p>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-panel rounded-lg border border-border p-10 text-center flex flex-col items-center justify-center">
          <div className="w-10 h-10 rounded-full bg-subtle flex items-center justify-center text-ink-muted mb-2.5">
            <Sparkles className="w-5 h-5 text-terracotta-600" />
          </div>
          <h3 className="text-sm font-semibold text-ink">Ready for Risk Evaluation</h3>
          <p className="text-xs text-ink-muted mt-1 max-w-md">
            Configure the 3-column customer profile above or select a preset, then click Assess Customer Risk to inspect churn probability, annual revenue at risk, and local attribution drivers.
          </p>
        </div>
      )}
    </div>
  )
}
