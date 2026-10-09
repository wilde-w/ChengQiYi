import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'node:path'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      // 走代理而不是在客户端拼绝对地址：SSE 需要同源，
      // 且这样前后端都能用相对路径，免去 CORS 与 base url 配置漂移。
      '/api': {
        // 目标是 **127.0.0.1 而非 localhost**。本机的 localhost 优先解析到
        // IPv6 的 ::1，而后端 uvicorn 绑在 IPv4 上——Node 会先撞一次
        // ECONNREFUSED 再回退，表现为每个请求都白等一截。用字面量 IP
        // 把解析这一步整个跳过。
        target: process.env.VITE_API_BASE || 'http://127.0.0.1:8000',
        changeOrigin: true,
        // SSE 必须关掉代理缓冲，否则事件会攒在缓冲区里成批到达，
        // 「逐字输出」就退化成了「一次性出现」。
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            if (String(proxyRes.headers['content-type']).includes('text/event-stream')) {
              proxyRes.headers['cache-control'] = 'no-cache, no-transform'
              delete proxyRes.headers['content-encoding']
            }
          })
        },
      },
    },
  },
})
