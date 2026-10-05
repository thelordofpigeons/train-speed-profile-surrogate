import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [react()],
    server: {
        proxy: {
            '/runs': 'http://localhost:8000',
            '/simulate': 'http://localhost:8000',
            '/jobs': 'http://localhost:8000',
        }
    },
    build: {
        outDir: '../static',
        emptyOutDir: true
    }
})
