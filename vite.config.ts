import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const projectRoot = path.dirname(fileURLToPath(import.meta.url))

function contentTypeFor(filePath: string) {
  const extension = path.extname(filePath).toLowerCase()
  if (extension === '.png') return 'image/png'
  if (extension === '.jpg' || extension === '.jpeg') return 'image/jpeg'
  if (extension === '.webp') return 'image/webp'
  if (extension === '.gif') return 'image/gif'
  if (extension === '.svg') return 'image/svg+xml'
  return 'application/octet-stream'
}

function isInsideProject(filePath: string) {
  const relative = path.relative(projectRoot, filePath)
  return relative === '' || (!relative.startsWith('..') && !path.isAbsolute(relative))
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    {
      name: 'paperlens-media',
      configureServer(server) {
        server.middlewares.use('/paperlens-media', (request, response) => {
          const url = new URL(request.url ?? '', 'http://127.0.0.1')
          const rawPath = url.searchParams.get('path') ?? ''
          const filePath = path.resolve(rawPath)
          if (!rawPath || !isInsideProject(filePath)) {
            response.statusCode = 403
            response.end('Forbidden')
            return
          }
          fs.stat(filePath, (statError, stat) => {
            if (statError || !stat.isFile()) {
              response.statusCode = 404
              response.end('Not found')
              return
            }
            response.setHeader('Content-Type', contentTypeFor(filePath))
            response.setHeader('Content-Length', String(stat.size))
            response.setHeader('Cache-Control', 'no-store')
            fs.createReadStream(filePath).pipe(response)
          })
        })
      },
    },
  ],
})
