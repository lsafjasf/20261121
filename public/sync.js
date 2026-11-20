export const state = {
  snapshot: null,
  clockOffset: 0,
  source: 'connecting',
  lastEventAt: 0,
  lastHeartbeatAt: 0,
};

export async function estimateClockOffset() {
  const samples = [];
  for (let i = 0; i < 4; i += 1) {
    const t1 = Date.now();
    try {
      const res = await fetch(`/api/time?t=${t1}`, { cache: 'no-store' });
      const t4 = Date.now();
      const data = await res.json();
      const rtt = t4 - t1;
      const offset = data.serverMs - (t1 + rtt / 2);
      samples.push({ rtt, offset });
    } catch {
      // retry next iteration
    }
  }
  if (samples.length) {
    samples.sort((a, b) => a.rtt - b.rtt);
    state.clockOffset = samples[0].offset;
  }
  return state.clockOffset;
}

export function serverNow() {
  return Date.now() + state.clockOffset;
}

export function connect(onUpdate, cid = null) {
  const url = cid ? `/api/events?cid=${encodeURIComponent(cid)}` : '/api/events';
  const es = new EventSource(url);
  es.onmessage = (event) => {
    state.snapshot = JSON.parse(event.data);
    state.lastEventAt = serverNow();
    state.lastHeartbeatAt = state.lastEventAt;
    state.source = 'live';
    onUpdate(state.snapshot);
  };
  es.addEventListener('heartbeat', () => {
    state.lastHeartbeatAt = serverNow();
    state.source = 'live';
  });
  es.onerror = () => {
    state.source = 'reconnecting';
  };
  es.onopen = () => {
    state.source = 'live';
    state.lastHeartbeatAt = serverNow();
  };
  setInterval(() => {
    if (state.source === 'live' && state.lastHeartbeatAt
      && serverNow() - state.lastHeartbeatAt > 3500) {
      state.source = 'reconnecting';
      onUpdate(state.snapshot);
    }
  }, 500);
  return es;
}

export function mediaPosition(now = serverNow()) {
  const s = state.snapshot;
  if (!s) return 0;
  const pos = s.playing && s.anchorServerMs !== null
    ? s.anchorPos + Math.max(0, now - s.anchorServerMs)
    : s.anchorPos;
  return Math.max(0, pos);
}

export function effectivePosition(now = serverNow()) {
  const s = state.snapshot;
  const delay = s ? s.delayMs : 0;
  return Math.max(0, mediaPosition(now) - delay);
}

export function isCleared(now = serverNow()) {
  const s = state.snapshot;
  if (!s || !s.clearUntil) return false;
  return now < s.clearUntil;
}

export function activeCue(cues, pos) {
  return cues.find((cue) => pos >= cue.start && pos < cue.end) || null;
}

export function fmtTime(ms) {
  const value = Math.max(0, Math.round(ms));
  const h = Math.floor(value / 3600000);
  const m = Math.floor((value % 3600000) / 60000);
  const s = Math.floor((value % 60000) / 1000);
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(h)}:${pad(m)}:${pad(s)}`;
}

export async function postControl(action, payload = {}) {
  const res = await fetch('/api/control', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, payload }),
  });
  if (!res.ok) throw new Error(`control failed: ${res.status}`);
  state.snapshot = await res.json();
  state.lastEventAt = serverNow();
  return state.snapshot;
}
