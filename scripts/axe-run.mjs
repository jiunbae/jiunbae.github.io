#!/usr/bin/env node
// axe-core via Playwright — chromedriver-free, runs against `astro preview`.
// 결과: scripts/axe-latest.json (violations 합산) + per-page txt
// 사용: npm run audit:a11y

import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import http from 'node:http';
import { chromium } from 'playwright';
import { AxeBuilder } from '@axe-core/playwright';

const PORT = 4322;
const BASE = `http://localhost:${PORT}`;

const PAGES = [
  '/', '/posts/', '/notes/', '/reviews/', '/projects/',
  '/about/', '/contact/', '/404', '/status/',
  '/design', '/design/changelog', '/design/voice',
  '/playground/', '/tools/', '/privacy/',
];

const TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa', 'best-practice'];

async function waitForServer(url, timeout = 30000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    try {
      await new Promise((resolve, reject) => {
        const req = http.get(url, res => { res.resume(); resolve(); });
        req.on('error', reject);
        req.setTimeout(500, () => { req.destroy(); reject(new Error('timeout')); });
      });
      return true;
    } catch { await new Promise(r => setTimeout(r, 500)); }
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

const browser = await chromium.launch();
try {
  const ok = await waitForServer(BASE);
  if (!ok) { console.error('Server failed:\n', serverOut); process.exit(1); }
  console.log('Server up.\n');

  const ctx = await browser.newContext();
  const page = await ctx.newPage();

  const results = [];
  let totalViolations = 0;
  for (const path of PAGES) {
    process.stdout.write(`  ${path.padEnd(22)} `);
    try {
      await page.goto(BASE + path, { waitUntil: 'networkidle', timeout: 15000 });
      const r = await new AxeBuilder({ page }).withTags(TAGS).analyze();
      const violations = r.violations;
      const byImpact = { critical: 0, serious: 0, moderate: 0, minor: 0 };
      for (const v of violations) byImpact[v.impact] = (byImpact[v.impact] ?? 0) + v.nodes.length;
      const total = Object.values(byImpact).reduce((a,b) => a+b, 0);
      totalViolations += total;
      console.log(`v:${total}  c:${byImpact.critical}  s:${byImpact.serious}  m:${byImpact.moderate}  mi:${byImpact.minor}`);
      results.push({ path, total, byImpact, violations: violations.map(v => ({
        id: v.id, impact: v.impact, help: v.help, helpUrl: v.helpUrl,
        nodes: v.nodes.length, sample: v.nodes[0]?.target,
      })) });
      writeFileSync(
        `scripts/axe-reports/${(path.replace(/[\/]/g, '_') || 'root')}.json`,
        JSON.stringify({ path, byImpact, violations: r.violations }, null, 2),
      );
    } catch (e) {
      console.log('ERR', e.message);
      results.push({ path, error: e.message });
    }
  }

  const summary = {
    at: new Date().toISOString(),
    totalViolations,
    perPage: results,
  };
  writeFileSync('scripts/axe-latest.json', JSON.stringify(summary, null, 2));
  console.log(`\nTotal violations: ${totalViolations}`);
  console.log(`Summary → scripts/axe-latest.json`);
  console.log(`Reports → scripts/axe-reports/*.json`);
} finally {
  await browser.close();
  server.kill('SIGTERM');
}
