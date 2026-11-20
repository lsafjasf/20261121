import {
  state, connect, estimateClockOffset, serverNow,
  effectivePosition, isCleared, activeCue, fmtTime,
} from './sync.js';

const el = {
  status: document.getElementById('status'),
  cueA: document.getElementById('cue-a'),
  cueB: document.getElementById('cue-b'),
  cleared: document.getElementById('cleared'),
  clock: document.getElementById('clock'),
  delay: document.getElementById('delay'),
  lag: document.getElementById('lag'),
};

function render() {
  const now = serverNow();
  const pos = effectivePosition(now);
  const snap = state.snapshot;
  const cleared = isCleared(now);

  if (snap) {
    const cueA = activeCue(snap.cuesA, pos);
    const cueB = activeCue(snap.cuesB, pos);
    el.cueA.textContent = cleared ? '' : (cueA?.text ?? '');
    el.cueB.textContent = cleared ? '' : (cueB?.text ?? '');
    el.cueA.hidden = cleared || !cueA;
    el.cueB.hidden = cleared || !cueB;
    el.cleared.hidden = !cleared;
    el.clock.textContent = fmtTime(pos);
    el.delay.textContent = snap.delayMs ? `人工推迟 ${(snap.delayMs / 1000).toFixed(1)}s` : '';
  }

  const lag = state.lastEventAt ? now - state.lastEventAt : null;
  el.lag.textContent = lag === null ? '' : `事件距今 ${lag}ms`;
  el.status.textContent = state.source === 'live' ? '已同步'
    : state.source === 'reconnecting' ? '断开，按导播基准本地续播…' : '连接中…';
  el.status.className = `status-pill status-${state.source}`;

  window.__subState = {
    now,
    positionMs: pos,
    effectiveMs: pos,
    cleared,
    cueA: el.cueA.textContent,
    cueB: el.cueB.textContent,
    source: state.source,
    clockOffset: state.clockOffset,
  };
  requestAnimationFrame(render);
}

await estimateClockOffset();
const cid = new URLSearchParams(location.search).get('cid');
connect(() => {}, cid);
requestAnimationFrame(render);
