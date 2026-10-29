import React from 'react'

interface MetricCardProps {
  label: string
  value: string | number
  subtext?: string
  confidenceInterval?: { lower: number; upper: number }
  variant?: 'default' | 'highlight' | 'danger'
  tooltip?: string
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  subtext,
  confidenceInterval,
  variant = 'default',
  tooltip,
}) => {
  const borderClass =
    variant === 'highlight'
      ? 'border-terracotta-500/40 bg-terracotta-50/30'
      : variant === 'danger'
      ? 'border-red-300 bg-red-50/30'
      : 'border-border bg-panel'

  return (
    <div className={`p-4 rounded-lg border ${borderClass} flex flex-col justify-between`} title={tooltip}>
      <div className="text-xs font-medium text-ink-muted uppercase tracking-wider mb-1.5 flex items-center justify-between">
        <span>{label}</span>
      </div>
      <div className="flex items-baseline gap-2">
        <span className="text-2xl font-bold tracking-tight text-ink font-mono">{value}</span>
        {confidenceInterval && (
          <span className="text-xs text-ink-muted font-mono">
            [{confidenceInterval.lower.toFixed(3)}, {confidenceInterval.upper.toFixed(3)}]
          </span>
        )}
      </div>
      {subtext && <p className="text-xs text-ink-muted mt-2 leading-relaxed">{subtext}</p>}
    </div>
  )
}
