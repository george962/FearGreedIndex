import { defineConfig } from 'vite';
import fs from 'node:fs';
import path from 'node:path';

// Relative URLs allow the same bundle to work at GitHub Pages' /FearGreedIndex/ path.
// Read-only local dev middleware serves the exact generated Python artifacts.
const DATA_FILES = new Set([
  'analysis.json', 'historical_decisions.json', 'version.json',
  'v3_challenger.json', 'v3_challenger_history.csv',
  'historical_decisions.csv', 'analogs.csv', 'event_study.csv',
]);
export default defineConfig({
  base: './',
  build: { outDir: 'dist', emptyOutDir: true, sourcemap: false, target: 'es2020' },
  server: { fs: { strict: true } },
  plugins: [{
    name: 'serve-generated-fgi-data',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const pathname = (req.url || '').split('?')[0].replace(/^\//, '');
        if (!DATA_FILES.has(pathname)) return next();
        const file = path.resolve(process.cwd(), '../site', pathname);
        if (!fs.existsSync(file)) return next();
        res.setHeader('Content-Type', pathname.endsWith('.json') ? 'application/json; charset=utf-8' : 'text/csv; charset=utf-8');
        res.setHeader('Cache-Control', 'no-store');
        fs.createReadStream(file).pipe(res);
      });
    },
  }],
});
