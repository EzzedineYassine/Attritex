import React from 'react'
import { RiskTier } from '../types'

interface RiskBadgeProps {
  tier: RiskTier
  showLabel?: boolean
  className?: string
}

export const RiskBadge: React.FC<RiskBadgeProps> = ({ tier, showLabel = true, className = '' }) => {
  const configs: Record<RiskTier, { label: string; bg: string; text: string; border: string }> = {
    high: {
      label: 'High Risk',
      bg: 'bg-red-50',
      text: 'text-red-700',
      border: 'border-red-200',
    },
    medium: {
      label: 'Medium Risk',
      bg: 'bg-amber-50',
      text: 'text-amber-800',
      border: 'border-amber-200',
    },
    low: {
      label: 'Low Risk',
      bg: 'bg-emerald-50',
      text: 'text-emerald-800',
      border: 'border-emerald-200',
    },
  }

  const config = configs[tier]

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium border ${config.bg} ${config.text} ${config.border} ${className}`}
    >
      <span
        className={`w-1.5 h-1.5 rounded-full ${
          tier === 'high' ? 'bg-red-600' : tier === 'medium' ? 'bg-amber-600' : 'bg-emerald-600'
        }`}
      />
      {showLabel ? config.label : tier.toUpperCase()}
    </span>
  )
}
