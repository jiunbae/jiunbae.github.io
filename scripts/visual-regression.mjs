#!/usr/bin/env node
// Playwright visual regression — 페이지 × {light, dark} × {mobile, desktop} 스냅샷.
// baseline 비교: 첫 실행 시 scripts/visual-baseline/에 저장. 이후엔 scripts/visual-actual/와 diff.
// 사용:
//   npm run visual:baseline   # baseline 생성
//   npm run visual:check      # 비교 (baseline vs actual)
//
// 의존: playwright (이미 설치됨)
// PNG diff은 사용자 측에서 시각 비교 — 본 스크립트는 픽셀 수준 hash만 비교.

import { spawn } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync, existsSync, readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { join } from 'node:path';
import http from 'node:http';
import { chromium } from 'playwright';

const MODE = process.argv[2] || 'check';  // 'baseline' or 'check'
const PORT = 4322;
const BASE = `http://localhost:${PORT}`;

const PAGES = [
  { path: '/', name: 'home' },
  { path: '/posts/', name: 'posts' },
  { path: '/notes/', name: 'notes' },
  { path: '/reviews/', name: 'reviews' },
  { path: '/projects/', name: 'projects' },
  { path: '/about/', name: 'about' },
  { path: '/contact/', name: 'contact' },
  { path: '/404', name: '404' },
  { path: '/status/', name: 'status' },
  { path: '/design', name: 'design' },
  { path: '/design/changelog', name: 'changelog' },
  { path: '/design/voice', name: 'voice' },
  { path: '/playground/', name: 'playground' },
  { path: '/tools/', name: 'tools' },
  { path: '/privacy/', name: 'privacy' },
];

const VIEWPORTS = {
  mobile: { width: 390, height: 844 },
  desktop: { width: 1280, height: 800 },
};

const THEMES = ['light', 'dark'];

const OUT_DIR = MODE === 'baseline' ? 'scripts/visual-baseline' : 'scripts/visual-actual';

async function waitForServer(url, timeout = 30000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    try {
      await new Promise((resolve, reject) => {
        const req = http.get(url, res => { res.resume(); resolve(); });
        req.on('error', reject);
        req.setTimeout(500, () => { req.destroy(); reject(new Error('t')); });
      });
      return true;
    } catch { await new Promise(r => setTimeout(r, 500)); }
  }
  return false;
}

mkdirSync(OUT_DIR, { recursive: true });

console.log(`Mode: ${MODE} → ${OUT_DIR}`);
console.log('Starting astro preview...');
const server = spawn('npx', ['astro', 'preview', '--port', String(PORT)], {
  stdio: ['ignore', 'pipe', 'pipe'],
});
let serverOut = '';
server.stdout.on('data', d => serverOut += d.toString());
server.stderr.on('data', d => serverOut += d.toString());

const browser = await chromium.launch();
try {
  const ok = await waitForServer(BASE);
  if (!ok) { console.error('Server failed:\n', serverOut); process.exit(1); }
  console.log('Server up.\n');

  const results = [];
  for (const theme of THEMES) {
    for (const [vp, size] of Object.entries(VIEWPORTS)) {
      const ctx = await browser.newContext({
        viewport: size,
        colorScheme: theme,
        deviceScaleFactor: 1,
      });
      const page = await ctx.newPage();
      for (const { path, name } of PAGES) {
        const file = `${name}.${theme}.${vp}.png`;
        process.stdout.write(`  ${file.padEnd(40)} `);
        try {
          await page.goto(BASE + path, { waitUntil: 'networkidle', timeout: 15000 });
          // Set theme via localStorage if site has manual toggle
          await page.evaluate(t => {
            document.documentElement.setAttribute('data-theme', t);
          }, theme);
          await page.waitForTimeout(200);
          const buf = await page.screenshot({ fullPage: true });
          writeFileSync(join(OUT_DIR, file), buf);
          const hash = createHash('sha256').update(buf).digest('hex').slice(0, 12);
          console.log(`saved (${(buf.length/1024).toFixed(0)} kB, hash ${hash})`);
          results.push({ file, hash, bytes: buf.length });
        } catch (e) {
          console.log('ERR', e.message);
          results.push({ file, error: e.message });
        }
      }
      await ctx.close();
    }
  }

  // Save manifest
  const manifest = { mode: MODE, at: new Date().toISOString(), results };
  writeFileSync(join(OUT_DIR, '_manifest.json'), JSON.stringify(manifest, null, 2));

  if (MODE === 'check' && existsSync('scripts/visual-baseline/_manifest.json')) {
    const baseline = JSON.parse(readFileSync('scripts/visual-baseline/_manifest.json', 'utf8'));
    const baseMap = new Map(baseline.results.filter(r => r.hash).map(r => [r.file, r.hash]));
    const diffs = [];
    for (const r of results) {
      if (r.hash && baseMap.has(r.file) && baseMap.get(r.file) !== r.hash) {
        diffs.push({ file: r.file, baseline: baseMap.get(r.file), actual: r.hash });
      }
    }
    console.log(`\n=== Diff vs baseline ===`);
    console.log(`Total snapshots: ${results.length}`);
    console.log(`Changed: ${diffs.length}`);
    if (diffs.length) {
      console.log('\nChanged files:');
      for (const d of diffs) console.log(`  ${d.file}  ${d.baseline} → ${d.actual}`);
    } else {
      console.log('No pixel changes detected.');
    }
    writeFileSync('scripts/visual-diff.json', JSON.stringify({ diffs }, null, 2));
  }

  console.log(`\nSnapshots → ${OUT_DIR}/`);
} finally {
  await browser.close();
  server.kill('SIGTERM');
}
