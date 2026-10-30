import React from 'react'
import { AlertTriangle, CheckCircle2, ShieldCheck } from 'lucide-react'
import { HealthStatus, ModelInfo } from '../types'
import { Logo } from './Logo'

interface HeaderProps {
  activeTab: 'scoring' | 'batch' | 'evaluation' | 'governance'
  onTabChange: (tab: 'scoring' | 'batch' | 'evaluation' | 'governance') => void
  health: HealthStatus | null
  modelInfo: ModelInfo | null
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  onTabChange,
  health,
  modelInfo,
}) => {
  const isHealthy = health?.status === 'ok' && health?.model_loaded

  return (
    <header className="border-b border-border bg-panel sticky top-0 z-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-6">
            <Logo />

            <nav className="flex items-center space-x-1 sm:space-x-2">
              <button
                onClick={() => onTabChange('scoring')}
                className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'scoring'
                    ? 'bg-ink text-canvas'
                    : 'text-ink-muted hover:text-ink hover:bg-subtle'
                }`}
              >
                Customer Scoring
              </button>
              <button
                onClick={() => onTabChange('batch')}
                className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'batch'
                    ? 'bg-ink text-canvas'
                    : 'text-ink-muted hover:text-ink hover:bg-subtle'
                }`}
              >
                Batch Analysis
              </button>
              <button
                onClick={() => onTabChange('evaluation')}
                className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'evaluation'
                    ? 'bg-ink text-canvas'
                    : 'text-ink-muted hover:text-ink hover:bg-subtle'
                }`}
              >
                Model Evaluation
              </button>
              <button
                onClick={() => onTabChange('governance')}
                className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'governance'
                    ? 'bg-ink text-canvas'
                    : 'text-ink-muted hover:text-ink hover:bg-subtle'
                }`}
              >
                Governance & Data
              </button>
            </nav>
          </div>

          <div className="flex items-center gap-3">
            {modelInfo && (
              <div className="hidden md:flex items-center gap-2 text-xs text-ink-muted bg-subtle px-2.5 py-1 rounded border border-border">
                <ShieldCheck className="w-3.5 h-3.5 text-terracotta-600" />
                <span className="font-mono">{modelInfo.model.label}</span>
                <span className="text-border">|</span>
                <span>{modelInfo.dataset.provenance.rows.toLocaleString()} records</span>
              </div>
            )}

            <div
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium border ${
                isHealthy
                  ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                  : health
                  ? 'bg-amber-50 text-amber-800 border-amber-200'
                  : 'bg-red-50 text-red-800 border-red-200'
              }`}
            >
              {isHealthy ? (
                <>
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                  <span>Model Live</span>
                </>
              ) : health ? (
                <>
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
                  <span>Model Not Loaded</span>
                </>
              ) : (
                <>
                  <AlertTriangle className="w-3.5 h-3.5 text-red-600" />
                  <span>Server Offline</span>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}
