import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // REST API → 后端
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // WebSocket → 后端
      '/ws': {
        target:  'ws://localhost:8000',
        ws:      true,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    rollupOptions: {
      output: {
        // 按模块分包，减小主 bundle 体积
        manualChunks: {
          vendor: ['react', 'react-dom', 'react-router-dom'],
          state:  ['zustand'],
          http:   ['axios'],
        },
      },
    },
  },
})
