import React, { useEffect, useState } from 'react'
import { AlertCircle, RefreshCw } from 'lucide-react'
import { fetchHealth, fetchModelEvaluation, fetchModelInfo } from './api'
import { BatchPrediction } from './components/BatchPrediction'
import { Header } from './components/Header'
import { ModelEvaluationView } from './components/ModelEvaluationView'
import { ModelGovernanceView } from './components/ModelGovernanceView'
import { SinglePrediction } from './components/SinglePrediction'
import { HealthStatus, ModelEvaluation, ModelInfo } from './types'

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'scoring' | 'batch' | 'evaluation' | 'governance'>('scoring')
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [evaluation, setEvaluation] = useState<ModelEvaluation | null>(null)
  const [loading, setLoading] = useState(true)
  const [fetchError, setFetchError] = useState<string | null>(null)

  const loadData = async () => {
    setLoading(true)
    setFetchError(null)
    try {
      const healthData = await fetchHealth()
      setHealth(healthData)

      if (healthData.model_loaded) {
        const [infoData, evalData] = await Promise.all([
          fetchModelInfo(),
          fetchModelEvaluation(),
        ])
        setModelInfo(infoData)
        setEvaluation(evalData)
      }
    } catch (err: unknown) {
      if (err instanceof Error) {
        setFetchError(err.message)
      } else {
        setFetchError('Failed to communicate with Attritex prediction service.')
      }
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [])

  return (
    <div className="min-h-screen bg-canvas flex flex-col text-ink selection:bg-terracotta-200">
      <Header
        activeTab={activeTab}
        onTabChange={setActiveTab}
        health={health}
        modelInfo={modelInfo}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {fetchError && (
          <div className="mb-6 p-4 rounded-lg bg-red-50 border border-red-200 text-red-800 text-sm flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-red-600 shrink-0" />
              <span>
                Backend server is unreachable ({fetchError}). Ensure `uv run attritex serve` is running on port 8000.
              </span>
            </div>
            <button
              onClick={loadData}
              className="px-2.5 py-1 text-xs font-semibold rounded bg-red-100 hover:bg-red-200 text-red-900 border border-red-300 flex items-center gap-1 transition-colors"
            >
              <RefreshCw className="w-3 h-3" />
              Retry
            </button>
          </div>
        )}

        {health && !health.model_loaded && (
          <div className="mb-6 p-4 rounded-lg bg-amber-50 border border-amber-200 text-amber-800 text-sm flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-amber-600 shrink-0" />
              <span>
                Model artifacts not found. Run `uv run attritex train` to train and register the classifier.
              </span>
            </div>
            <button
              onClick={loadData}
              className="px-2.5 py-1 text-xs font-semibold rounded bg-amber-100 hover:bg-amber-200 text-amber-900 border border-amber-300 flex items-center gap-1 transition-colors"
            >
              <RefreshCw className="w-3 h-3" />
              Reload State
            </button>
          </div>
        )}

        {loading && !modelInfo ? (
          <div className="p-16 flex flex-col items-center justify-center text-center">
            <div className="w-8 h-8 border-2 border-terracotta-600 border-t-transparent rounded-full animate-spin mb-3" />
            <p className="text-sm font-medium text-ink">Initializing Attritex Intelligence System...</p>
            <p className="text-xs text-ink-muted mt-1">Connecting to FastAPI prediction pipeline</p>
          </div>
        ) : (
          <div>
            {activeTab === 'scoring' && (
              <SinglePrediction
                features={modelInfo?.features || []}
                modelLoaded={Boolean(health?.model_loaded)}
              />
            )}
            {activeTab === 'batch' && <BatchPrediction />}
            {activeTab === 'evaluation' && (
              <ModelEvaluationView evaluation={evaluation} modelInfo={modelInfo} />
            )}
            {activeTab === 'governance' && (
              <ModelGovernanceView modelInfo={modelInfo} />
            )}
          </div>
        )}
      </main>

      <footer className="border-t border-border bg-panel py-6 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between text-xs text-ink-muted gap-3">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-ink">Attritex</span>
            <span>&middot;</span>
            <span>Customer Retention & Risk Analytics</span>
          </div>
          <div className="flex items-center gap-4">
            <span>FastAPI + scikit-learn + React</span>
            <span>&middot;</span>
            <span>Apache-2.0 Public Data Benchmark</span>
          </div>
        </div>
      </footer>
    </div>
  )
}
