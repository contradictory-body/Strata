import type { Config } from 'tailwindcss'

export default {
  content:  ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',   // html.dark 切换
  theme: {
    extend: {
      fontFamily: {
        sans:  ['"DM Sans"',          'system-ui', 'sans-serif'],
        serif: ['"DM Serif Display"', 'Georgia',   'serif'],
      },
      colors: {
        // 亮色主题
        accent: {
          DEFAULT: '#C9501F',
          light:   '#E0622C',
          dark:    '#A33F17',
          subtle:  '#FDF0EB',
        },
        canvas: {
          DEFAULT: '#FAFAF8',
          sidebar: '#F3F1EE',
          border:  '#E8E5E0',
        },
        ink: {
          DEFAULT: '#1C1917',
          muted:   '#78716C',
          faint:   '#A8A29E',
        },
        // 深色主题专用（dark: 前缀时使用）
        night: {
          DEFAULT: '#171412',
          sidebar: '#0F0D0C',
          panel:   '#201D1B',
          border:  '#2E2A27',
          hover:   '#252120',
          text:    '#EDE8E1',
          muted:   '#8A817B',
          faint:   '#564E49',
          accent:  '#E0622C',
        },
      },
      keyframes: {
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%':      { opacity: '0' },
        },
        'fade-in': {
          '0%':   { opacity: '0', transform: 'translateY(4px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'slide-up': {
          '0%':   { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'pulse-dot': {
          '0%, 100%': { transform: 'scale(1)',   opacity: '0.4' },
          '50%':      { transform: 'scale(1.5)', opacity: '1'   },
        },
      },
      animation: {
        blink:       'blink 0.9s step-end infinite',
        'fade-in':   'fade-in 0.2s ease-out',
        'slide-up':  'slide-up 0.25s ease-out',
        'pulse-dot': 'pulse-dot 1.4s ease-in-out infinite',
      },
    },
  },
} satisfies Config
