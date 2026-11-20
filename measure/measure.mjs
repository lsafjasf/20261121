import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const base = process.env.BASE_URL || 'http://127.0.0.1:8080';
const outDir = path.join(root, 'artifacts');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const post = (action, payload = {}) => fetch(`${base}/api/control`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ action, payload }),
}).then((r) => r.json());

async function resetTimeline() {
  await post('unclear');
  await post('setDelay', { delayMs: 0 });
  await post('pause');
  await post('seek', { positionMs: 0 });
}

async function makeContext(browser, name, route = '/viewer', cid = null) {
  const context = await browser.newContext({ viewport: { width: 1280, height: 800 } });
  const page = await context.newPage();
  const target = cid ? `${route}?cid=${encodeURIComponent(cid)}` : route;
  await page.goto(`${base}${target}`);
  await page.waitForFunction(
    () => window.__subState && window.__subState.source === 'live',
    null, { timeout: 8000 },
  );
  const samples = [];
  const sourceHistory = [];
  const timer = setInterval(async () => {
    const localMs = Date.now();
    const sub = await page.evaluate(() => window.__subState).catch(() => null);
    if (sub) {
      samples.push({ localMs, ...sub });
      const last = sourceHistory.at(-1);
      if (!last || last.source !== sub.source) {
        sourceHistory.push({ localMs, source: sub.source });
      }
    }
  }, 50);
  return { name, context, page, samples, timer, sourceHistory };
}

function percentile(values, p) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))];
}

function summarize(values) {
  const steady = values.filter((v) => v <= 100);
  return {
    n: values.length,
    mean: Math.round(values.reduce((a, b) => a + b, 0) / Math.max(1, values.length)),
    p50: Math.round(percentile(values, 50)),
    p95: Math.round(percentile(values, 95)),
    max: Math.round(Math.max(...values)),
    steadyP95: steady.length ? Math.round(percentile(steady, 95)) : null,
    steadyMax: steady.length ? Math.round(Math.max(...steady)) : null,
    transientSpikes: values.filter((v) => v > 100).length,
  };
}

function pairSamples(ctxX, ctxY, sinceMs, field = 'effectiveMs') {
  const pairs = [];
  const ys = ctxY.samples.filter((s) => s.localMs >= sinceMs - 50);
  for (const sx of ctxX.samples) {
    if (sx.localMs < sinceMs) continue;
    let best = null;
    let bestGap = Infinity;
    for (const sy of ys) {
      const gap = Math.abs(sy.localMs - sx.localMs);
      if (gap < bestGap) {
        bestGap = gap;
        best = sy;
      }
    }
    if (best && bestGap <= 50) pairs.push([sx, best, bestGap]);
  }
  const diffs = pairs.map(([x, y]) => Math.abs(x[field] - y[field]));
  const shown = (s) => (s.cleared ? '' : s.cueA);
  const contentMismatch = pairs.filter(([x, y]) => shown(x) !== shown(y)).length;
  const pairingGaps = pairs.map(([, , gap]) => gap);
  return {
    ...summarize(diffs),
    pairs: pairs.length,
    contentMismatch,
    maxPairingGapMs: Math.max(...pairingGaps, 0),
  };
}

const events = [];
const mark = (phase, note) => events.push({ t: Date.now(), phase, note });

const browser = await chromium.launch();
const viewerA = await makeContext(browser, 'viewer-A', '/viewer', 'viewer-A');
const director = await makeContext(browser, 'director', '/director');
await resetTimeline();
mark('setup', 'two contexts live, timeline reset');

const report = { startedAt: new Date().toISOString(), base, phases: {} };
const measure = async (phase, ctxX, ctxY, waitMs, field = 'effectiveMs') => {
  const since = Date.now();
  await sleep(waitMs);
  const stats = pairSamples(ctxX, ctxY, since, field);
  stats.note = events.at(-1)?.note || phase;
  report.phases[phase] = stats;
};

await post('play');
mark('play', 'play from 0');
await measure('baselinePlay', viewerA, director, 4000);

await post('pause');
mark('pause', 'paused around 4s');
await measure('duringPause', viewerA, director, 3000);
await post('play');
mark('resume', 'resumed');
await sleep(1500);

await post('pause');
await post('seek', { positionMs: 18000 });
mark('skip', 'seek to 18000 (cue #4)');
await post('play');
await measure('afterSkip', viewerA, director, 3000);

await post('setDelay', { delayMs: 2000 });
mark('delay', 'manual delay 2000ms');
await measure('manualDelay', viewerA, director, 3000);

await post('clear', { holdMs: 3000 });
mark('clear', 'clear 3000ms');
await measure('duringClear', viewerA, director, 3500);
await measure('afterClear', viewerA, director, 1500);

await post('setDelay', { delayMs: 0 });
await fetch(`${base}/api/disconnect?cid=viewer-A`, { method: 'POST' });
mark('offline', 'viewer-A network cut');
const offlineSince = Date.now();
await sleep(4500);
{
  const stats = pairSamples(viewerA, director, offlineSince);
  stats.sourceSeen = [...new Set(
    viewerA.samples.filter((s) => s.localMs >= offlineSince).map((s) => s.source),
  )];
  stats.sourceTransitions = viewerA.sourceHistory
    .filter((h) => h.localMs >= offlineSince - 200);
  report.phases.offlineExtrapolation = stats;
}
mark('online', 'viewer-A EventSource auto-reconnected');
await measure('afterReconnect', viewerA, director, 3000);

const viewerB = await makeContext(browser, 'viewer-B', '/viewer', 'viewer-B');
mark('lateJoin', 'viewer-B opened late');
const lateSince = Date.now();
await sleep(3500);
{
  const stats = pairSamples(viewerA, viewerB, lateSince);
  const ys = viewerB.samples;
  const pairs = [];
  for (const sx of viewerA.samples.filter((s) => s.localMs >= lateSince)) {
    let best = null;
    let bestGap = Infinity;
    for (const sy of ys) {
      const gap = Math.abs(sy.localMs - sx.localMs);
      if (gap < bestGap) {
        bestGap = gap;
        best = sy;
      }
    }
    if (best && bestGap <= 50) pairs.push([sx, best]);
  }
  stats.sameContent = pairs.every(([x, y]) =>
    (x.cleared ? '' : x.cueA) === (y.cleared ? '' : y.cueA)
    && (x.cleared ? '' : x.cueB) === (y.cleared ? '' : y.cueB)
    && x.cleared === y.cleared);
  stats.note = 'late joiner vs connected viewer';
  report.phases.lateJoin = stats;
}

await post('pause');
const viewerC = await makeContext(browser, 'viewer-C', '/viewer', 'viewer-C');
mark('lateJoinPaused', 'viewer-C joined while paused');
await measure('lateJoinWhilePaused', viewerA, viewerC, 1500);

report.events = events;
report.clockOffsets = {
  viewerA: viewerA.samples.at(-1)?.clockOffset,
  viewerB: viewerB.samples.at(-1)?.clockOffset,
  viewerC: viewerC.samples.at(-1)?.clockOffset,
  director: director.samples.at(-1)?.clockOffset,
};

for (const ctx of [viewerA, viewerB, viewerC, director]) {
  clearInterval(ctx.timer);
  await ctx.context.close().catch(() => {});
}
await browser.close();

fs.writeFileSync(path.join(outDir, 'measure-report.json'), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report.phases, null, 2));
console.log('clock offsets (ms):', JSON.stringify(report.clockOffsets));
console.log(`full report: ${path.join(outDir, 'measure-report.json')}`);
