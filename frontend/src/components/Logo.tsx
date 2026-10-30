import React from 'react'

interface LogoProps {
  className?: string
  height?: number
  iconOnly?: boolean
}

export const Logo: React.FC<LogoProps> = ({
  className = '',
  height = 36,
  iconOnly = false,
}) => {
  if (iconOnly) {
    return (
      <div className={`flex items-center ${className}`}>
        <img
          src="/attritex-icon.png"
          alt="Attritex Mark"
          style={{ height: `${height}px`, width: 'auto' }}
          className="object-contain select-none"
        />
      </div>
    )
  }

  return (
    <div className={`flex items-center ${className}`}>
      <img
        src="/attritex-logo.png"
        alt="Attritex — Retention & Revenue Intelligence"
        style={{ height: `${height}px`, width: 'auto' }}
        className="object-contain select-none max-w-full"
      />
    </div>
  )
}
