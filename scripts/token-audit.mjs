#!/usr/bin/env node
// Token adoption audit — raw value 인벤토리와 토큰 사용률을 측정.
// 결과를 stdout(요약) + scripts/audit-history.jsonl(누적 기록)로 출력.

import { readdirSync, readFileSync, statSync, appendFileSync, existsSync, writeFileSync } from 'node:fs';
import { join, extname, relative } from 'node:path';
import { execSync } from 'node:child_process';

const ROOT = process.cwd();
const TARGETS = ['src/pages', 'src/components', 'src/layouts', 'src/views', 'src/styles'];
const EXTS = new Set(['.astro', '.scss', '.tsx', '.css']);

const PATTERNS = {
  'raw-spacing': /(?:padding|margin|gap):\s*\d+(?:\.\d+)?(?:px|rem)/g,
  'raw-radius': /border-radius:\s*\d+(?:\.\d+)?(?:px|rem)\s*(?:;|\s)/g,
  'raw-motion': /transition:[^;{}]*\d+(?:\.\d+)?(?:s|ms)\b[^;{}]*/g,
  'raw-color-hex': /:\s*#[0-9a-fA-F]{3,6}\b/g,
  'raw-shadow-rgba': /box-shadow:[^;]*rgba\(/g,
};
const TOKEN_USES = /var\(--[a-z0-9-_]+\)/gi;

function walk(dir) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const p = join(dir, entry);
    const st = statSync(p);
    if (st.isDirectory()) out.push(...walk(p));
    else if (EXTS.has(extname(p))) out.push(p);
  }
  return out;
}

const totals = Object.fromEntries(Object.keys(PATTERNS).map(k => [k, 0]));
let tokenUses = 0, filesScanned = 0;
const perFile = [];

for (const target of TARGETS) {
  const dir = join(ROOT, target);
  if (!existsSync(dir)) continue;
  for (const file of walk(dir)) {
    const rel = relative(ROOT, file);
    if (rel.includes('node_modules')) continue;
    const text = readFileSync(file, 'utf8');
    const counts = {};
    let total = 0;
    for (const [k, p] of Object.entries(PATTERNS)) {
      const c = (text.match(p) ?? []).length;
      counts[k] = c;
      totals[k] += c;
      total += c;
    }
    const tk = (text.match(TOKEN_USES) ?? []).length;
    tokenUses += tk;
    filesScanned++;
    if (total > 0) perFile.push({ file: rel, total, ...counts, tokens: tk });
  }
}

const rawTotal = Object.values(totals).reduce((a,b)=>a+b,0);
const adoptionPct = tokenUses + rawTotal === 0
  ? 100
  : ((tokenUses / (tokenUses + rawTotal)) * 100).toFixed(2);

const now = new Date().toISOString();
let commit = 'unknown';
try { commit = execSync('git rev-parse --short HEAD', { encoding: 'utf8' }).trim(); } catch {}

const summary = {
  at: now,
  commit,
  filesScanned,
  tokenUses,
  raw: totals,
  rawTotal,
  adoptionPct: parseFloat(adoptionPct),
};

console.log('=== TOKEN ADOPTION ===');
console.log(`commit ${commit} · ${filesScanned} files · ${tokenUses} token uses · ${rawTotal} raw values · ${adoptionPct}% adoption\n`);
console.log('Raw breakdown:');
for (const [k,v] of Object.entries(totals)) console.log(`  ${k.padEnd(20)} ${v}`);

console.log('\nTop offenders:');
perFile.sort((a,b) => b.total - a.total);
for (const r of perFile.slice(0, 15)) {
  console.log(`  ${r.total.toString().padStart(3)} ${r.file}`);
}

// Append to history
const historyPath = join(ROOT, 'scripts/audit-history.jsonl');
appendFileSync(historyPath, JSON.stringify(summary) + '\n');

// Latest snapshot
writeFileSync(join(ROOT, 'scripts/audit-latest.json'), JSON.stringify(summary, null, 2));

console.log(`\nLogged → scripts/audit-history.jsonl`);
