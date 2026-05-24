#!/usr/bin/env node
// Score tracker — 4명 cold reviewer 점수 누적.
// 사용: node scripts/score-tracker.mjs add R15 9.6 9.5 9.8 9.2 "note"
//      node scripts/score-tracker.mjs show

import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { execSync } from 'node:child_process';

const FILE = join(process.cwd(), 'scripts/scores.json');

function load() {
  if (!existsSync(FILE)) return { dimensions: ['A','B','C','D'], names: { A: 'DS Adoption (Atomic)', B: 'Nielsen + RUI', C: 'WCAG 2.2 AA + ARIA', D: 'Gestalt + IA + Grid' }, rounds: [] };
  return JSON.parse(readFileSync(FILE, 'utf8'));
}

function save(data) {
  writeFileSync(FILE, JSON.stringify(data, null, 2));
}

function cmd() { try { return execSync('git rev-parse --short HEAD', { encoding: 'utf8' }).trim(); } catch { return 'unknown'; } }

const [,, action, ...args] = process.argv;

if (action === 'add') {
  const [label, a, b, c, d, ...note] = args;
  const data = load();
  const round = {
    label,
    at: new Date().toISOString(),
    commit: cmd(),
    A: parseFloat(a), B: parseFloat(b), C: parseFloat(c), D: parseFloat(d),
    avg: parseFloat(((parseFloat(a)+parseFloat(b)+parseFloat(c)+parseFloat(d))/4).toFixed(3)),
    note: note.join(' '),
  };
  data.rounds.push(round);
  save(data);
  console.log(`Added ${label}: A=${a} B=${b} C=${c} D=${d} avg=${round.avg}`);
} else if (action === 'show' || !action) {
  const data = load();
  console.log('=== SCORE HISTORY ===');
  console.log('Round'.padEnd(8) + 'A'.padEnd(7) + 'B'.padEnd(7) + 'C'.padEnd(7) + 'D'.padEnd(7) + 'Avg'.padEnd(8) + 'Δavg'.padEnd(8) + 'Note');
  let prev = null;
  for (const r of data.rounds) {
    const delta = prev ? (r.avg - prev.avg).toFixed(2) : '—';
    const sign = prev && r.avg > prev.avg ? '+' : '';
    console.log(
      r.label.padEnd(8) +
      r.A.toFixed(1).padEnd(7) + r.B.toFixed(1).padEnd(7) + r.C.toFixed(1).padEnd(7) + r.D.toFixed(1).padEnd(7) +
      r.avg.toFixed(3).padEnd(8) + (sign + delta).padEnd(8) +
      (r.note ?? '')
    );
    prev = r;
  }
} else {
  console.log('Usage:');
  console.log('  node scripts/score-tracker.mjs add <label> <A> <B> <C> <D> [note...]');
  console.log('  node scripts/score-tracker.mjs show');
}
