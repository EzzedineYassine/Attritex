export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: '#F8F6F0',
        panel: '#FFFFFF',
        subtle: '#F2EFEB',
        border: '#E3DDD4',
        ink: {
          DEFAULT: '#1B242D',
          muted: '#5B6875',
          faint: '#8D99A6',
        },
        terracotta: {
          50: '#FDF5F2',
          100: '#FCEBE5',
          200: '#F9D4C7',
          500: '#D65A31',
          600: '#BF4820',
          700: '#9E3715',
        },
        risk: {
          high: '#DC2626',
          highBg: '#FEF2F2',
          highBorder: '#FECACA',
          medium: '#D97706',
          mediumBg: '#FFFBEB',
          mediumBorder: '#FDE68A',
          low: '#15803D',
          lowBg: '#F0FDF4',
          lowBorder: '#BBF7D0',
        }
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      }
    },
  },
  plugins: [],
}
