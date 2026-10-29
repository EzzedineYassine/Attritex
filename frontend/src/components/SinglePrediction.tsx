import React, { useState } from 'react'
import {
  AlertCircle,
  ArrowDownRight,
  ArrowUpRight,
  CheckCircle,
  HelpCircle,
  Info,
  RotateCcw,
  Sparkles,
} from 'lucide-react'
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

export const SinglePrediction: React.FC<SinglePredictionProps> = ({
  modelLoaded,
}) => {
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
        if (!prev.TotalCharges || prev.TotalCharges === 0) {
          next.TotalCharges = Math.round(tenureVal * monthlyVal * 100) / 100
        }
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

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 bg-panel p-4 rounded-lg border border-border">
        <div>
          <h2 className="text-base font-semibold text-ink">Individual Customer Risk Profiler</h2>
          <p className="text-xs text-ink-muted mt-0.5">
            Evaluate churn risk for a specific account and inspect localized explainability factors.
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

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <form onSubmit={handleSubmit} className="lg:col-span-7 space-y-6">
          <div className="bg-panel rounded-lg border border-border p-5 space-y-5">
            <div className="border-b border-border pb-3">
              <h3 className="text-sm font-semibold text-ink uppercase tracking-wide">
                Account & Contract
              </h3>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
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
                  Tenure <span className="text-ink-muted">(months)</span>
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

          <div className="bg-panel rounded-lg border border-border p-5 space-y-5">
            <div className="border-b border-border pb-3">
              <h3 className="text-sm font-semibold text-ink uppercase tracking-wide">
                Telecommunications & Internet Services
              </h3>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div>
                <label className="block text-xs font-medium text-ink mb-1">Phone Service</label>
                <select
                  value={String(formData.PhoneService || 'Yes')}
                  onChange={(e) => handleInputChange('PhoneService', e.target.value)}
                  className="w-full px-3 py-1.5 text-sm rounded border border-border bg-canvas focus:outline-none focus:border-terracotta-500"
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
                  className="w-full px-3 py-1.5 text-sm rounded border border-border bg-canvas focus:outline-none focus:border-terracotta-500 disabled:opacity-50"
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

              <div>
                <label className="block text-xs font-medium text-ink mb-1">Internet Service</label>
                <select
                  value={String(formData.InternetService || 'DSL')}
                  onChange={(e) => handleInputChange('InternetService', e.target.value)}
                  className="w-full px-3 py-1.5 text-sm rounded border border-border bg-canvas focus:outline-none focus:border-terracotta-500"
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
                  className="w-full px-3 py-1.5 text-sm rounded border border-border bg-canvas focus:outline-none focus:border-terracotta-500 disabled:opacity-50"
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
                  className="w-full px-3 py-1.5 text-sm rounded border border-border bg-canvas focus:outline-none focus:border-terracotta-500 disabled:opacity-50"
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
                  className="w-full px-3 py-1.5 text-sm rounded border border-border bg-canvas focus:outline-none focus:border-terracotta-500 disabled:opacity-50"
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
                  className="w-full px-3 py-1.5 text-sm rounded border border-border bg-canvas focus:outline-none focus:border-terracotta-500 disabled:opacity-50"
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
                  className="w-full px-3 py-1.5 text-sm rounded border border-border bg-canvas focus:outline-none focus:border-terracotta-500 disabled:opacity-50"
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
                  className="w-full px-3 py-1.5 text-sm rounded border border-border bg-canvas focus:outline-none focus:border-terracotta-500 disabled:opacity-50"
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

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            <div className="bg-panel rounded-lg border border-border p-5 space-y-4">
              <div className="border-b border-border pb-2">
                <h3 className="text-sm font-semibold text-ink uppercase tracking-wide">
                  Account Demographics
                </h3>
              </div>
              <div className="space-y-3">
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
              </div>
            </div>

            <div className="bg-panel rounded-lg border border-border p-5 space-y-4">
              <div className="border-b border-border pb-2">
                <h3 className="text-sm font-semibold text-ink uppercase tracking-wide">
                  Financial & Billing
                </h3>
              </div>
              <div className="space-y-3">
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
                      max="50000"
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

          <div className="flex items-center justify-between pt-2">
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
              Reset
            </button>

            <button
              type="submit"
              disabled={loading}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded text-sm font-semibold bg-terracotta-600 text-white hover:bg-terracotta-700 transition-colors shadow-sm disabled:opacity-50"
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

        <div className="lg:col-span-5 space-y-6">
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
            <div className="bg-panel rounded-lg border border-border overflow-hidden shadow-sm space-y-5 p-6">
              <div className="flex items-start justify-between border-b border-border pb-4">
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="text-base font-bold text-ink">Risk Assessment Summary</h3>
                    <RiskBadge tier={result.risk_tier} />
                  </div>
                  <p className="text-xs text-ink-muted mt-1 font-mono">
                    ID: {result.customer_id || 'Unassigned'} | Model: {result.model_version.split('-')[1]}
                  </p>
                </div>
                {result.flagged ? (
                  <span className="px-2 py-0.5 text-xs font-semibold bg-red-100 text-red-800 rounded border border-red-300">
                    ACTION FLAG
                  </span>
                ) : (
                  <span className="px-2 py-0.5 text-xs font-semibold bg-emerald-100 text-emerald-800 rounded border border-emerald-300">
                    STABLE
                  </span>
                )}
              </div>

              <div className="bg-canvas rounded-lg p-5 border border-border">
                <div className="flex items-baseline justify-between mb-2">
                  <span className="text-xs uppercase font-medium text-ink-muted tracking-wider">
                    Estimated Churn Probability
                  </span>
                  <span className="text-xs font-mono text-ink-muted">
                    Threshold: {(result.decision_threshold * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex items-baseline gap-3">
                  <span className="text-4xl font-extrabold font-mono tracking-tight text-ink">
                    {(result.churn_probability * 100).toFixed(1)}%
                  </span>
                  <span className="text-xs text-ink-muted">
                    {result.churn_probability >= result.tier_cutoffs.high
                      ? 'Exceeds high risk threshold'
                      : result.churn_probability >= result.tier_cutoffs.medium
                      ? 'Exceeds medium risk tier'
                      : 'Within acceptable baseline'}
                  </span>
                </div>

                <div className="w-full h-3 bg-border rounded-full overflow-hidden mt-4 flex">
                  <div
                    className="h-full bg-emerald-500"
                    style={{ width: `${result.tier_cutoffs.medium * 100}%` }}
                    title={`Low tier: 0 - ${(result.tier_cutoffs.medium * 100).toFixed(1)}%`}
                  />
                  <div
                    className="h-full bg-amber-500"
                    style={{
                      width: `${(result.tier_cutoffs.high - result.tier_cutoffs.medium) * 100}%`,
                    }}
                    title={`Medium tier: ${(result.tier_cutoffs.medium * 100).toFixed(1)}% - ${(result.tier_cutoffs.high * 100).toFixed(1)}%`}
                  />
                  <div
                    className="h-full bg-red-500"
                    style={{ width: `${(1 - result.tier_cutoffs.high) * 100}%` }}
                    title={`High tier: > ${(result.tier_cutoffs.high * 100).toFixed(1)}%`}
                  />
                </div>
                <div className="relative w-full h-2">
                  <div
                    className="absolute -top-1 w-2.5 h-2.5 bg-ink border-2 border-white rounded-full -ml-1 transition-all"
                    style={{ left: `${Math.min(result.churn_probability * 100, 100)}%` }}
                    title={`Customer score: ${(result.churn_probability * 100).toFixed(1)}%`}
                  />
                </div>

                {result.tier_context && (
                  <p className="text-xs text-ink-muted mt-3 pt-3 border-t border-border/60 leading-relaxed">
                    <strong>Holdout Benchmark:</strong> Customers in the {result.tier_context.tier} tier
                    exhibited an empirical churn rate of{' '}
                    <strong>{(result.tier_context.observed_churn_rate * 100).toFixed(1)}%</strong> in the
                    test cohort (accounting for {result.tier_context.customers.toLocaleString()} customers).
                  </p>
                )}
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-xs font-semibold text-ink uppercase tracking-wider">
                    Counterfactual Risk Drivers
                  </h4>
                  <span className="text-xs text-ink-muted flex items-center gap-1" title="Estimated effect compared to training reference values">
                    <HelpCircle className="w-3 h-3" />
                    vs Reference Profile
                  </span>
                </div>
                {result.factors.length === 0 ? (
                  <p className="text-xs text-ink-muted italic p-3 bg-canvas rounded border border-border">
                    This customer's attributes closely mirror the baseline reference profile. No single factor caused an effect greater than 0.5%.
                  </p>
                ) : (
                  <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
                    {result.factors.map((factor) => {
                      const isRiskIncrease = factor.effect > 0
                      return (
                        <div
                          key={factor.feature}
                          className="p-2.5 rounded border border-border bg-canvas flex items-center justify-between text-xs"
                        >
                          <div className="space-y-0.5">
                            <span className="font-semibold text-ink">{factor.label}</span>
                            <div className="text-ink-muted text-[11px]">
                              Value: <span className="font-mono text-ink font-medium">{factor.customer_value}</span> (Ref:{' '}
                              <span className="font-mono">{factor.reference_value}</span>)
                            </div>
                          </div>
                          <div
                            className={`flex items-center gap-1 font-mono font-bold ${
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
                    })}
                  </div>
                )}
              </div>

              {result.flagged && (
                <div className="p-3 bg-terracotta-50 border border-terracotta-200 rounded-lg text-xs space-y-1.5">
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

              <div className="pt-2 border-t border-border">
                <div className="flex items-start gap-2 text-[11px] text-ink-muted leading-relaxed">
                  <Info className="w-3.5 h-3.5 text-ink-muted shrink-0 mt-0.5" />
                  <p>
                    {result.limitations[0] ||
                      'Estimates reflect historical statistical correlation, not direct causation.'}
                  </p>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-panel rounded-lg border border-border p-12 text-center flex flex-col items-center justify-center">
              <div className="w-12 h-12 rounded-full bg-subtle flex items-center justify-center text-ink-muted mb-3">
                <Sparkles className="w-6 h-6 text-terracotta-600" />
              </div>
              <h3 className="text-sm font-semibold text-ink">Ready for Risk Evaluation</h3>
              <p className="text-xs text-ink-muted mt-1 max-w-sm">
                Fill out the customer account details on the left, select a quick preset, or click
                Assess Customer Risk to inspect predictive probability.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
