#!/usr/bin/env node
// axe-core CLI runner — preview server를 띄우고 주요 페이지를 자동 검사.
// 사용: npm run audit:a11y

import { spawn, execSync } from 'node:child_process';
import { writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import http from 'node:http';

const PORT = 4322;
const BASE = `http://localhost:${PORT}`;

const PAGES = [
  '/', '/posts/', '/notes/', '/reviews/', '/projects/',
  '/about/', '/contact/', '/404', '/status/',
  '/design', '/design/changelog', '/design/voice',
];

async function waitForServer(url, timeout = 20000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    try {
      await new Promise((resolve, reject) => {
        const req = http.get(url, res => { res.resume(); resolve(); });
        req.on('error', reject);
        req.setTimeout(500, () => { req.destroy(); reject(new Error('timeout')); });
      });
      return true;
    } catch {
      await new Promise(r => setTimeout(r, 500));
    }
  }
  return false;
}

mkdirSync('scripts/axe-reports', { recursive: true });

console.log('Starting astro preview...');
const server = spawn('npx', ['astro', 'preview', '--port', String(PORT)], {
  stdio: ['ignore', 'pipe', 'pipe'],
});

let serverOut = '';
server.stdout.on('data', d => serverOut += d.toString());
server.stderr.on('data', d => serverOut += d.toString());

try {
  const ready = await waitForServer(BASE);
  if (!ready) {
    console.error('Server failed to start:\n', serverOut);
    process.exit(1);
  }
  console.log('Server up.\n');

  const results = [];
  for (const path of PAGES) {
    const url = BASE + path;
    process.stdout.write(`  ${path.padEnd(20)} `);
    try {
      const out = execSync(`npx @axe-core/cli "${url}" --exit --tags wcag2a,wcag2aa,wcag21a,wcag21aa,wcag22aa 2>&1 || true`, {
        encoding: 'utf8',
        maxBuffer: 4 * 1024 * 1024,
      });
      const violations = (out.match(/Violations found:\s*(\d+)/) ?? [])[1] ?? '?';
      const passes = (out.match(/(\d+)\s+passes?/i) ?? [])[1] ?? '?';
      console.log(`violations:${violations}  passes:${passes}`);
      results.push({ path, violations, passes, raw: out });
      writeFileSync(`scripts/axe-reports/${path.replace(/[\/]/g, '_') || 'root'}.txt`, out);
    } catch (e) {
      console.log('ERR', e.message);
    }
  }

  const summary = {
    at: new Date().toISOString(),
    pages: results.map(({ path, violations, passes }) => ({ path, violations, passes })),
    totalViolations: results.reduce((a, r) => a + (parseInt(r.violations) || 0), 0),
  };
  writeFileSync('scripts/axe-latest.json', JSON.stringify(summary, null, 2));
  console.log(`\nTotal violations: ${summary.totalViolations}`);
  console.log(`Reports → scripts/axe-reports/*.txt`);
  console.log(`Summary → scripts/axe-latest.json`);
} finally {
  server.kill('SIGTERM');
}
