import {
  state, connect, estimateClockOffset, serverNow,
  mediaPosition, effectivePosition, isCleared, activeCue, fmtTime, postControl,
} from './sync.js';

const el = {
  status: document.getElementById('status'),
  cueA: document.getElementById('cue-a'),
  cueB: document.getElementById('cue-b'),
  clock: document.getElementById('clock'),
  effective: document.getElementById('effective'),
  seq: document.getElementById('seq'),
  delayValue: document.getElementById('delay-value'),
  clearValue: document.getElementById('clear-value'),
  delayInput: document.getElementById('delay-input'),
  seekInput: document.getElementById('seek-input'),
};

const bind = (id, action, payload = {}) => {
  document.getElementById(id).addEventListener('click', () => postControl(action, payload));
};

bind('play', 'play');
bind('pause', 'pause');
bind('next', 'nextCue');
bind('prev', 'prevCue');
bind('clear-hold', 'clear', { holdMs: 3000 });
bind('clear-indef', 'clear');
bind('clear-off', 'unclear');
bind('delay-reset', 'setDelay', { delayMs: 0 });

document.querySelectorAll('[data-delay]').forEach((button) => {
  button.addEventListener('click', () => postControl('nudgeDelay', {
    delayMs: Number(button.dataset.delay),
  }));
});

document.getElementById('delay-set').addEventListener('click', () => {
  postControl('setDelay', { delayMs: Number(el.delayInput.value || 0) });
});
document.getElementById('seek-go').addEventListener('click', () => {
  postControl('seek', { positionMs: Number(el.seekInput.value || 0) });
});

function render() {
  const now = serverNow();
  const snap = state.snapshot;
  if (snap) {
    const rawPos = mediaPosition(now);
    const effPos = effectivePosition(now);
    const cueA = activeCue(snap.cuesA, rawPos);
    const cueB = activeCue(snap.cuesB, rawPos);
    el.cueA.textContent = cueA?.text ?? '';
    el.cueB.textContent = cueB?.text ?? '';
    el.cueA.hidden = !cueA;
    el.cueB.hidden = !cueB;
    el.clock.textContent = `时间轴: ${fmtTime(rawPos)}`;
    el.effective.textContent = `观众端: ${fmtTime(effPos)}${isCleared(now) ? ' · 清屏中' : ''}`;
    el.seq.textContent = `seq #${snap.seq} · ${snap.playing ? '播放中' : '已暂停'}`;
    el.delayValue.textContent = `当前推迟 ${(snap.delayMs / 1000).toFixed(1)}s`;
    el.clearValue.textContent = isCleared(now) ? '清屏生效中' : '';
    window.__subState = {
      now,
      positionMs: rawPos,
      effectiveMs: effPos,
      cleared: isCleared(now),
      cueA: el.cueA.textContent,
      cueB: el.cueB.textContent,
      source: state.source,
      clockOffset: state.clockOffset,
      seq: snap.seq,
    };
  }
  el.status.textContent = state.source === 'live' ? '已连接'
    : state.source === 'reconnecting' ? '重连中…' : '连接中…';
  el.status.className = `status-pill status-${state.source}`;
  requestAnimationFrame(render);
}

await estimateClockOffset();
connect(() => {});
requestAnimationFrame(render);
