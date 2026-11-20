import { LoopAudioEngine } from "./audio-engine.js";
import {
  DEFAULT_LADDER,
  buildTimeline,
  formatClock,
  formatPulse,
  roundAtTime,
} from "./timeline.js";
import { TRACKS, LADDER_PRESETS } from "./tracks.js";
import { runTimingChecks } from "./verification.js";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const engine = new LoopAudioEngine();

const state = {
  activeTrack: null,
  duration: 0,
  loopIn: 0,
  loopOut: 1,
  cursorSource: 0,
  snapHalfBeat: true,
  countInBars: 1,
  ladder: [...DEFAULT_LADDER],
  peaks: [],
  playing: false,
  preparing: false,
  drag: null,
  dragMoved: false,
  lastActiveRow: -1,
  toastTimer: null,
  loadToken: 0,
};

function makeTrackLadder(track) {
  return Array.from({ length: 10 }, (_, index) => {
    const value = track.bpm - 20 + index * 4;
    return Math.round(value / 2) * 2;
  });
}

function snapTime(time) {
  const track = state.activeTrack;
  if (!track || !state.snapHalfBeat) return Math.max(0, Math.min(state.duration, time));
  const halfBeat = 30 / track.bpm;
  return Math.round(time / halfBeat) * halfBeat;
}

function sourceBeat(time) {
  return (time * state.activeTrack.bpm) / 60;
}

function trackDurationFallback(track) {
  return (track.bars * 4 * 60) / track.bpm;
}

function showToast(message, duration = 1800) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("visible");
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => toast.classList.remove("visible"), duration);
}

function setEngineStatus(label, mode = "") {
  const status = $("#engine-status");
  status.className = `engine-status ${mode}`.trim();
  status.querySelector("span").textContent = label;
}

function drawMiniWave(canvas, track, index) {
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(80, rect.width || 180);
  const height = Math.max(25, rect.height || 37);
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  const context = canvas.getContext("2d");
  context.scale(dpr, dpr);
  const count = Math.floor(width / 3);
  let seed = 87 + index * 131;
  const random = () => {
    seed = (seed * 9301 + 49297) % 233280;
    return seed / 233280;
  };
  context.fillStyle = track.accent;
  for (let bar = 0; bar < count; bar += 1) {
    const pulse = 0.3 + 0.7 * Math.abs(Math.sin((bar / count) * Math.PI * 5 + index));
    const amp = (0.15 + random() * 0.75) * pulse;
    const barHeight = Math.max(1, amp * height * 0.82);
    context.globalAlpha = 0.28 + random() * 0.48;
    context.fillRect(bar * 3, (height - barHeight) / 2, 1.5, barHeight);
  }
}

function renderTrackList() {
  const container = $("#track-list");
  container.replaceChildren();
  TRACKS.forEach((track, index) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "track-card";
    card.dataset.trackId = track.id;
    card.style.setProperty("--accent", track.accent);
    card.innerHTML = `
      <div class="track-preview">
        <canvas aria-hidden="true"></canvas>
        <span class="style-badge">${track.style}</span>
      </div>
      <div class="track-card-body">
        <div class="track-card-title"><strong>${track.title}</strong><span>${track.bpm} BPM</span></div>
        <p>${track.subtitle}</p>
        <div class="track-card-meta"><span><b>${track.key}</b></span><span><b>${track.bars} BARS</b></span><span><b>CC0</b></span></div>
      </div>`;
    card.addEventListener("click", () => loadTrack(index));
    container.append(card);
    const canvas = card.querySelector("canvas");
    requestAnimationFrame(() => drawMiniWave(canvas, track, index));
  });
}

function markActiveTrack() {
  $$(".track-card").forEach((card) => {
    card.classList.toggle("active", card.dataset.trackId === state.activeTrack?.id);
  });
}

function computePeaks(buffer, bins = 1800) {
  const channel = buffer.getChannelData(0);
  const peaks = new Float32Array(bins);
  const step = Math.max(1, Math.floor(channel.length / bins));
  for (let bin = 0; bin < bins; bin += 1) {
    const start = bin * step;
    const end = Math.min(channel.length, start + step);
    let maximum = 0;
    for (let index = start; index < end; index += 1) {
      maximum = Math.max(maximum, Math.abs(channel[index]));
    }
    peaks[bin] = maximum;
  }
  return peaks;
}

async function loadTrack(index) {
  const track = TRACKS[index];
  if (!track) return;
  const token = ++state.loadToken;
  engine.stop();
  state.playing = false;
  state.preparing = false;
  updatePlayButton();
  $("#countin-flag").classList.add("hidden");
  $("#wave-empty").classList.remove("hidden");
  setEngineStatus("LOADING SOURCE", "loading");
  try {
    const loaded = await engine.loadTrack(track, track.file);
    if (token !== state.loadToken) return;
    state.activeTrack = track;
    state.duration = loaded.buffer.duration;
    state.ladder = makeTrackLadder(track);
    const beat = 60 / track.bpm;
    state.loopIn = Math.max(0, track.defaultBeats[0] * beat);
    state.loopOut = Math.min(state.duration, track.defaultBeats[1] * beat);
    state.cursorSource = state.loopIn;
    state.peaks = computePeaks(loaded.buffer);
    engine.setLoop(state.loopIn, state.loopOut);
    markActiveTrack();
    renderTrackDetails();
    renderLoopControls();
    renderLadder();
    renderPhraseMap();
    renderVerification();
    drawWaveform(0);
    $("#wave-empty").classList.add("hidden");
    setEngineStatus("PREPARING LADDER", "loading");
    state.preparing = true;
    updatePlayButton();
    await engine.prepare(state.ladder);
    if (token !== state.loadToken) return;
    state.preparing = false;
    setEngineStatus("ENGINE ARMED");
    updatePlayButton();
  } catch (error) {
    console.error(error);
    setEngineStatus("AUDIO ERROR", "error");
    showToast(`无法载入素材：${error.message}`, 4000);
  }
}

function renderTrackDetails() {
  const track = state.activeTrack;
  if (!track) return;
  $("#track-title").textContent = track.title;
  $("#track-subtitle").textContent = track.subtitle;
  $("#track-style").textContent = track.style;
  $("#fact-bpm").textContent = `${track.bpm} BPM`;
  $("#fact-meter").textContent = track.meter;
  $("#fact-key").textContent = track.key;
  document.documentElement.style.setProperty("--track-accent", track.accent);
}

function currentSourceTime() {
  if (!state.playing || !engine.timeline) return state.loopIn;
  const now = engine.now();
  if (now < engine.timeline.loopStart) return state.loopIn;
  const round = roundAtTime(engine.timeline, now);
  if (!round) return state.loopOut;
  const fraction = Math.max(0, Math.min(1, (now - round.start) / round.duration));
  return state.loopIn + fraction * (state.loopOut - state.loopIn);
}

function drawWaveform(playheadSource = null) {
  const canvas = $("#waveform");
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const dpr = window.devicePixelRatio || 1;
  const pixelWidth = Math.floor(rect.width * dpr);
  const pixelHeight = Math.floor(rect.height * dpr);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const context = canvas.getContext("2d");
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  const width = rect.width;
  const height = rect.height;
  const mid = height * 0.5;
  context.clearRect(0, 0, width, height);

  const track = state.activeTrack;
  if (!track || !state.duration) return;
  const xFor = (time) => (time / state.duration) * width;
  const inX = xFor(state.loopIn);
  const outX = xFor(state.loopOut);
  const beat = 60 / track.bpm;
  const totalBeats = state.duration / beat;

  context.lineWidth = 1;
  for (let index = 0; index <= totalBeats; index += 0.5) {
    const x = Math.round(xFor(index * beat)) + 0.5;
    const isBar = index % 4 === 0;
    const isBeat = index % 1 === 0;
    context.strokeStyle = isBar
      ? "rgba(228, 237, 208, 0.14)"
      : isBeat
        ? "rgba(228, 237, 208, 0.065)"
        : "rgba(228, 237, 208, 0.025)";
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, height);
    context.stroke();
    if (isBar && index < totalBeats) {
      context.fillStyle = "rgba(143, 148, 132, 0.42)";
      context.font = '7px "Ubuntu Mono", monospace';
      context.fillText(String(Math.floor(index / 4) + 1).padStart(2, "0"), x + 4, 11);
    }
  }

  context.fillStyle = "rgba(215, 255, 105, 0.032)";
  context.fillRect(inX, 0, outX - inX, height);
  context.strokeStyle = "rgba(215, 255, 105, 0.13)";
  context.beginPath();
  context.moveTo(0, mid + 0.5);
  context.lineTo(width, mid + 0.5);
  context.stroke();

  const peaks = state.peaks;
  if (peaks.length) {
    const level = Math.round(width);
    for (let x = 0; x < level; x += 1) {
      const peakIndex = Math.min(peaks.length - 1, Math.floor((x / width) * peaks.length));
      const amplitude = Math.pow(peaks[peakIndex], 0.78) * height * 0.39;
      const active = x >= inX && x <= outX;
      context.fillStyle = active ? "rgba(215, 255, 105, 0.91)" : "rgba(139, 146, 127, 0.34)";
      context.fillRect(x, mid - amplitude, 0.75, Math.max(1, amplitude * 2));
    }
  }

  context.strokeStyle = "rgba(215, 255, 105, 0.35)";
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(inX, 0);
  context.lineTo(inX, height);
  context.moveTo(outX, 0);
  context.lineTo(outX, height);
  context.stroke();

  const label = (x, text, alignRight = false) => {
    const boxWidth = 29;
    const boxX = alignRight ? x - boxWidth - 3 : x + 3;
    context.fillStyle = "#d7ff69";
    context.beginPath();
    context.roundRect(boxX, 13, boxWidth, 15, 2);
    context.fill();
    context.fillStyle = "#12130f";
    context.font = '8px "Ubuntu Mono", monospace';
    context.fillText(text, boxX + 7, 24);
  };
  label(inX, "IN");
  label(outX, "OUT", outX > width - 38);

  if (playheadSource !== null) {
    const x = Math.max(0, Math.min(width, xFor(playheadSource)));
    context.strokeStyle = "#ffad5a";
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(x + 0.5, 0);
    context.lineTo(x + 0.5, height);
    context.stroke();
    context.fillStyle = "#ffad5a";
    context.beginPath();
    context.moveTo(x - 4, 0);
    context.lineTo(x + 5, 0);
    context.lineTo(x + 0.5, 7);
    context.closePath();
    context.fill();
  }
}

function renderLoopControls() {
  const track = state.activeTrack;
  if (!track) return;
  const beat = 60 / track.bpm;
  const loopBeats = (state.loopOut - state.loopIn) / beat;
  const beatLabel = (time) => {
    const value = sourceBeat(time);
    return `BEAT ${(Math.floor(value) + 1).toString().padStart(2, "0")}.${Math.floor((value % 1) * 10)}`;
  };
  $("#in-time").textContent = formatClock(state.loopIn);
  $("#in-beat").textContent = beatLabel(state.loopIn);
  $("#out-time").textContent = formatClock(state.loopOut);
  $("#out-beat").textContent = beatLabel(state.loopOut);
  $("#loop-length").textContent = `${loopBeats.toFixed(2)} BEATS`;
  $("#loop-seconds").textContent = `${(state.loopOut - state.loopIn).toFixed(3)} SEC @ ${track.bpm}`;
  $("#countin-bars").textContent = state.countInBars === 1 ? "1 BAR" : `${state.countInBars} BARS`;
  $("#countin-flag").classList.toggle("hidden", state.countInBars === 0 || !state.playing);
  $$("#countin-control button").forEach((button) => {
    button.classList.toggle("active", Number(button.dataset.countin) === state.countInBars);
  });
}

function renderPhraseMap() {
  const track = state.activeTrack;
  if (!track) return;
  const container = $("#phrase-map");
  container.replaceChildren();
  const beat = 60 / track.bpm;
  const firstBar = sourceBeat(state.loopIn) / 4;
  const lastBar = sourceBeat(state.loopOut) / 4;
  for (let bar = 0; bar < track.bars; bar += 1) {
    const item = document.createElement("div");
    item.className = "phrase-bar";
    item.dataset.bar = bar;
    if (lastBar > bar && firstBar < bar + 1) item.classList.add("in-loop");
    item.style.setProperty("--beat", beat);
    if (bar % 2 === 0) item.innerHTML = `<span>${String(bar + 1).padStart(2, "0")}</span>`;
    container.append(item);
  }
}

function updatePhraseActive(sourceTime) {
  const bar = Math.floor(sourceBeat(sourceTime) / 4);
  $$(".phrase-bar").forEach((item) => {
    item.classList.toggle("active", Number(item.dataset.bar) === bar);
  });
}

function updateLoopFromPointer(event) {
  if (!state.drag || !state.activeTrack) return;
  const canvas = $("#waveform");
  const rect = canvas.getBoundingClientRect();
  const time = ((event.clientX - rect.left) / rect.width) * state.duration;
  const halfBeat = 30 / state.activeTrack.bpm;
  const minimum = Math.max(halfBeat, 0.02);
  const loopLength = state.loopOut - state.loopIn;
  if (state.drag.kind === "in") {
    state.loopIn = Math.min(snapTime(time), state.loopOut - minimum);
  } else if (state.drag.kind === "out") {
    state.loopOut = Math.max(snapTime(time), state.loopIn + minimum);
  } else if (state.drag.kind === "region") {
    const shiftedIn = Math.max(0, Math.min(state.duration - loopLength, state.loopIn + time - state.drag.origin));
    state.loopIn = shiftedIn;
    state.loopOut = shiftedIn + loopLength;
  }
  state.loopIn = Math.max(0, state.loopIn);
  state.loopOut = Math.min(state.duration, state.loopOut);
  renderLoopControls();
  renderPhraseMap();
  drawWaveform(state.playing ? currentSourceTime() : null);
}

function commitLoopChange() {
  if (!state.activeTrack) return;
  engine.setLoop(state.loopIn, state.loopOut);
  renderVerification();
  state.preparing = true;
  updatePlayButton();
  engine
    .prepare(state.ladder)
    .then(() => {
      state.preparing = false;
      setEngineStatus("ENGINE ARMED");
      updatePlayButton();
    })
    .catch((error) => {
      state.preparing = false;
      setEngineStatus("AUDIO ERROR", "error");
      showToast(error.message, 3500);
    });
  if (state.playing) stopPlayback();
  showToast("循环区已更新 · 音高锁定");
}

function renderLadder() {
  const container = $("#ladder-list");
  container.replaceChildren();
  const minimum = Math.min(...state.ladder);
  const maximum = Math.max(...state.ladder);
  state.ladder.forEach((bpm, index) => {
    const row = document.createElement("div");
    row.className = "ladder-row";
    row.dataset.index = index;
    const width = 16 + ((bpm - minimum) / Math.max(1, maximum - minimum)) * 84;
    row.innerHTML = `
      <span class="round-number">${String(index + 1).padStart(2, "0")}</span>
      <input class="bpm-input" type="number" min="20" max="300" step="1" value="${bpm}" aria-label="第 ${index + 1} 轮速度" />
      <span class="bpm-unit">BPM</span>
      <span class="ladder-meter"><i style="width:${width}%"></i></span>
      <button class="remove-step" title="移除这一轮" aria-label="移除第 ${index + 1} 轮">×</button>`;
    const input = row.querySelector("input");
    input.addEventListener("change", () => updateLadderValue(index, input.value));
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") input.blur();
    });
    row.querySelector(".remove-step").addEventListener("click", () => {
      if (state.ladder.length <= 2) return showToast("至少保留两轮速度");
      state.ladder.splice(index, 1);
      applyLadderChange({ render: true });
    });
    container.append(row);
  });
  $("#ladder-count").textContent = state.ladder.length;
  const markers = $("#progress-markers");
  markers.replaceChildren();
  state.ladder.forEach(() => {
    const marker = document.createElement("i");
    marker.style.width = `${100 / state.ladder.length}%`;
    markers.append(marker);
  });
  if (!state.playing) {
    $("#current-bpm").textContent = state.ladder[0];
    $("#next-bpm").textContent = state.ladder[1] ?? state.ladder[0];
    $("#round-badge").textContent = `ROUND 01 / ${state.ladder.length}`;
    $("#cycle-badge").textContent = "CYCLE 01";
  }
  renderNextChange();
}

function updateLadderValue(index, rawValue) {
  const value = Math.max(20, Math.min(300, Math.round(Number(rawValue) || state.ladder[index])));
  if (value === state.ladder[index]) return;
  state.ladder[index] = value;
  const input = document.querySelector(`.ladder-row[data-index="${index}"] .bpm-input`);
  if (input) input.value = value;
  const minimum = Math.min(...state.ladder);
  const maximum = Math.max(...state.ladder);
  $$(".ladder-row").forEach((row, rowIndex) => {
    const width = 16 + ((state.ladder[rowIndex] - minimum) / Math.max(1, maximum - minimum)) * 84;
    row.querySelector(".ladder-meter i").style.width = `${width}%`;
  });
  applyLadderChange();
}

function applyLadderChange({ render = false } = {}) {
  if (render) renderLadder();
  renderVerification();
  renderNextChange();
  if (!state.playing) return;
  const applying = async () => {
    state.preparing = true;
    updatePlayButton();
    try {
      await engine.prepare(state.ladder);
      engine.retime(state.ladder);
      showToast("热改速度将在下一轮边界提交", 2200);
    } catch (error) {
      showToast(`速度提交失败：${error.message}`, 3500);
    } finally {
      state.preparing = false;
      updatePlayButton();
    }
  };
  applying();
}

function renderNextChange() {
  if (!state.activeTrack) return;
  let nextRoundIndex = 0;
  let note = "启动后第 1 轮";
  if (state.playing && engine.timeline) {
    const now = engine.now();
    if (now >= engine.timeline.loopStart) {
      const current = roundAtTime(engine.timeline, now);
      if (current) {
        nextRoundIndex = (current.index + 1) % state.ladder.length;
        note = `第 ${nextRoundIndex + 1} 轮开始时`;
      }
    }
  }
  $("#next-change-bpm").textContent = `${state.ladder[nextRoundIndex]} BPM`;
  $("#next-change-note").textContent = note;
}

function renderVerification() {
  if (!state.activeTrack) return;
  const checks = runTimingChecks({
    sourceBpm: state.activeTrack.bpm,
    sourceIn: state.loopIn,
    sourceOut: state.loopOut,
    ladder: state.ladder,
  });
  const passCount = checks.filter((check) => check.pass).length;
  $("#verify-pass").textContent = passCount === checks.length ? "PASS" : "FAIL";
  $("#verify-summary").textContent =
    passCount === checks.length ? "10TH PULSE · SHORT LOOP · HOT TEMPO" : `${checks.length - passCount} CHECK FAILED`;
  const list = $("#check-list");
  list.replaceChildren();
  checks.forEach((check) => {
    const row = document.createElement("div");
    row.className = `check-row${check.pass ? "" : " fail"}`;
    row.innerHTML = `
      <span class="check-status"><svg><use href="#i-${check.pass ? "check" : "close"}"></use></svg></span>
      <span class="check-copy"><b>${check.label}</b><small>${check.detail}</small></span>
      <span class="check-value">${check.value}</span>`;
    list.append(row);
  });
}

function updatePlayButton() {
  const button = $("#btn-play");
  button.classList.toggle("playing", state.playing);
  button.classList.toggle("preparing", state.preparing);
  button.disabled = state.preparing;
  button.title = state.preparing ? "正在预渲染变速音频" : state.playing ? "停止 (Space)" : "播放 (Space)";
}

async function togglePlayback() {
  if (state.preparing) return;
  if (state.playing) {
    stopPlayback();
    return;
  }
  if (!state.activeTrack) return;
  state.preparing = true;
  updatePlayButton();
  setEngineStatus("PREPARING AUDIO", "loading");
  try {
    await engine.play({
      ladder: state.ladder,
      countInBars: state.countInBars,
      beatsPerBar: 4,
    });
    state.playing = true;
    state.preparing = false;
    setEngineStatus("PLAYING · PITCH LOCK");
    renderLoopControls();
    updatePlayButton();
  } catch (error) {
    state.preparing = false;
    setEngineStatus("AUDIO ERROR", "error");
    showToast(`播放失败：${error.message}`, 3500);
    updatePlayButton();
  }
}

function stopPlayback() {
  engine.stop();
  state.playing = false;
  state.preparing = false;
  setEngineStatus("ENGINE ARMED");
  $("#countin-flag").classList.add("hidden");
  updatePlayButton();
  $("#progress-fill").style.width = "0%";
  $("#position-pulse").textContent = "001:1:00";
  $("#position-time").textContent = "00:00.000";
  $("#round-badge").textContent = `ROUND 01 / ${state.ladder.length}`;
  $("#cycle-badge").textContent = "CYCLE 01";
  $("#current-bpm").textContent = state.ladder[0];
  $("#next-bpm").textContent = state.ladder[1] ?? state.ladder[0];
  $$(".ladder-row").forEach((row) => row.classList.remove("active", "complete"));
  drawWaveform(null);
  renderNextChange();
}

function updateLadderHighlight(activeIndex, completedRound = -1) {
  if (state.lastActiveRow === activeIndex && completedRound < 0) return;
  state.lastActiveRow = activeIndex;
  $$(".ladder-row").forEach((row, index) => {
    row.classList.toggle("active", index === activeIndex);
    row.classList.toggle("complete", completedRound >= 0 && index <= completedRound && index !== activeIndex);
  });
}

function updateTransportFromClock() {
  if (!state.playing || !engine.timeline) return;
  const now = engine.now();
  const timeline = engine.timeline;
  const countInDuration = timeline.loopStart - timeline.anchorTime;
  if (now >= timeline.rounds.at(-1).end) {
    stopPlayback();
    return;
  }
  if (now < timeline.loopStart) {
    const elapsed = Math.max(0, now - timeline.anchorTime);
    const countBeat = Math.min(timeline.countInBeats, Math.floor((elapsed * timeline.firstBpm) / 60) + 1);
    $("#round-badge").textContent = "COUNT-IN";
    $("#cycle-badge").textContent = `${countInDuration.toFixed(2)} SEC`;
    $("#position-pulse").textContent = `CNT ${countBeat}:${String(Math.floor(elapsed * 100) % 100).padStart(2, "0")}`;
    $("#position-time").textContent = formatClock(elapsed);
    $("#progress-fill").style.width = "0%";
    $("#current-bpm").textContent = timeline.firstBpm;
    $("#next-bpm").textContent = state.ladder[1] ?? state.ladder[0];
    $("#countin-flag").classList.remove("hidden");
    updatePhraseActive(state.loopIn);
    drawWaveform(state.loopIn);
    return;
  }
  const current = roundAtTime(timeline, now);
  if (!current) return;
  const fraction = Math.max(0, Math.min(1, (now - current.start) / current.duration));
  const rowIndex = current.index % state.ladder.length;
  const cycle = Math.floor(current.index / state.ladder.length) + 1;
  const next = timeline.rounds[current.index + 1];
  const sourceTime = state.loopIn + fraction * (state.loopOut - state.loopIn);
  const pulse = current.pulseStart + fraction * timeline.loopBeats;
  $("#round-badge").textContent = `ROUND ${String(rowIndex + 1).padStart(2, "0")} / ${state.ladder.length}`;
  $("#cycle-badge").textContent = `CYCLE ${String(cycle).padStart(2, "0")}`;
  $("#position-pulse").textContent = formatPulse(pulse);
  $("#position-time").textContent = formatClock(now - timeline.anchorTime);
  $("#progress-fill").style.width = `${fraction * 100}%`;
  $("#current-bpm").textContent = current.bpm;
  $("#next-bpm").textContent = next?.bpm ?? state.ladder[0];
  $("#countin-flag").classList.add("hidden");
  updateLadderHighlight(rowIndex, current.index % state.ladder.length === 0 ? -1 : rowIndex - 1);
  updatePhraseActive(sourceTime);
  renderNextChange();
  drawWaveform(sourceTime);
}

function adjustPoint(kind, halfBeatAmount) {
  if (!state.activeTrack) return;
  const halfBeat = 30 / state.activeTrack.bpm;
  const minimum = Math.max(halfBeat, 0.02);
  if (kind === "in") {
    state.loopIn = Math.max(0, Math.min(state.loopOut - minimum, snapTime(state.loopIn + halfBeatAmount * halfBeat)));
  } else {
    state.loopOut = Math.min(
      state.duration,
      Math.max(state.loopIn + minimum, snapTime(state.loopOut + halfBeatAmount * halfBeat)),
    );
  }
  renderLoopControls();
  renderPhraseMap();
  commitLoopChange();
}

function setPointAtPlayhead(kind) {
  if (!state.activeTrack) return;
  const time = currentSourceTime();
  const minimum = Math.max(30 / state.activeTrack.bpm, 0.02);
  if (kind === "in") {
    state.loopIn = Math.max(0, Math.min(state.loopOut - minimum, snapTime(time)));
  } else {
    state.loopOut = Math.min(state.duration, Math.max(state.loopIn + minimum, snapTime(time)));
  }
  renderLoopControls();
  renderPhraseMap();
  commitLoopChange();
  showToast(`${kind === "in" ? "入点" : "出点"}已设于 ${formatClock(time)}`);
}

function openModal(selector) {
  const modal = $(selector);
  modal.classList.add("open");
  modal.setAttribute("aria-hidden", "false");
}

function closeModals() {
  $$(".modal-backdrop").forEach((modal) => {
    modal.classList.remove("open");
    modal.setAttribute("aria-hidden", "true");
  });
}

function bindWaveformInteraction() {
  const canvas = $("#waveform");
  canvas.addEventListener("pointerdown", (event) => {
    if (!state.activeTrack) return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const inX = (state.loopIn / state.duration) * rect.width;
    const outX = (state.loopOut / state.duration) * rect.width;
    const threshold = 13;
    let kind = "seek";
    if (Math.abs(x - inX) <= threshold) kind = "in";
    else if (Math.abs(x - outX) <= threshold) kind = "out";
    state.drag = {
      kind,
      origin: ((event.clientX - rect.left) / rect.width) * state.duration,
      pointerId: event.pointerId,
    };
    state.dragMoved = false;
    canvas.setPointerCapture(event.pointerId);
    if (kind === "seek") {
      state.cursorSource = snapTime(state.drag.origin);
      drawWaveform(state.cursorSource);
    }
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!state.drag) return;
    state.dragMoved = true;
    if (state.drag.kind === "seek") {
      const rect = canvas.getBoundingClientRect();
      state.cursorSource = snapTime(((event.clientX - rect.left) / rect.width) * state.duration);
      drawWaveform(state.cursorSource);
      return;
    }
    updateLoopFromPointer(event);
  });

  const release = (event) => {
    if (!state.drag) return;
    const drag = state.drag;
    state.drag = null;
    try {
      canvas.releasePointerCapture(event.pointerId);
    } catch {
      // Pointer capture may already be released.
    }
    if (drag.kind !== "seek" && state.dragMoved) commitLoopChange();
  };
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);
}

function setCountIn(bars) {
  state.countInBars = bars;
  renderLoopControls();
  showToast(bars === 0 ? "计数已关闭" : `${bars} 小节计数将在播放前响起`);
}

function bindControls() {
  $("#btn-play").addEventListener("click", togglePlayback);
  $("#btn-stop").addEventListener("click", () => {
    stopPlayback();
    showToast("已停止 · 边界锚点重置");
  });
  $("#btn-rewind").addEventListener("click", () => {
    stopPlayback();
    state.cursorSource = state.loopIn;
    drawWaveform(state.cursorSource);
    showToast("回到入点");
  });

  $("#btn-add-step").addEventListener("click", () => {
    if (state.ladder.length >= 16) return showToast("最多 16 轮");
    state.ladder.push(Math.min(300, state.ladder.at(-1) + 2));
    applyLadderChange({ render: true });
  });

  $("#btn-ladder-preset").addEventListener("click", () => {
    const base = state.activeTrack.bpm - 20;
    state.ladder = Array.from({ length: 10 }, (_, index) => base + index * 4);
    applyLadderChange({ render: true });
    showToast("已应用 10 轮渐快阶梯");
  });

  $$("#countin-control button").forEach((button) => {
    button.addEventListener("click", () => setCountIn(Number(button.dataset.countin)));
  });

  $$("[data-nudge]").forEach((button) => {
    button.addEventListener("click", () => {
      adjustPoint(button.dataset.nudge, Number(button.dataset.amount));
    });
  });

  $("#btn-play-region").addEventListener("click", async () => {
    if (!state.activeTrack || state.preparing) return;
    if (state.playing) stopPlayback();
    try {
      await engine.audition(state.ladder[0]);
      showToast(`预听 ${state.ladder[0]} BPM · 音高锁定`, 1400);
    } catch (error) {
      showToast(`预听失败：${error.message}`, 3000);
    }
  });

  $("#click-toggle").addEventListener("change", (event) => {
    engine.setClickEnabled(event.target.checked);
  });

  $("#master-volume").addEventListener("input", (event) => {
    const value = Number(event.target.value);
    engine.setMasterVolume(value);
    $("#volume-label").textContent = `${value < 0 ? "−" : ""}${Math.abs(value)} dB`;
  });

  $("#btn-verify").addEventListener("click", () => {
    renderVerification();
    openModal("#verify-modal");
  });
  $("#verify-strip").addEventListener("click", () => {
    renderVerification();
    openModal("#verify-modal");
  });
  $("#btn-shortcuts").addEventListener("click", () => openModal("#shortcuts-modal"));
  $("#btn-help").addEventListener("click", () => openModal("#shortcuts-modal"));
  $$("[data-close-modal]").forEach((button) => button.addEventListener("click", closeModals));
  $$(".modal-backdrop").forEach((modal) => {
    modal.addEventListener("pointerdown", (event) => {
      if (event.target === modal) closeModals();
    });
  });
}

function bindKeyboard() {
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeModals();
      return;
    }
    if (event.target instanceof HTMLInputElement) return;
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    if (event.code === "Space") {
      event.preventDefault();
      togglePlayback();
    } else if (event.key.toLowerCase() === "i") {
      setPointAtPlayhead("in");
    } else if (event.key.toLowerCase() === "o") {
      setPointAtPlayhead("out");
    } else if (event.key.toLowerCase() === "r") {
      stopPlayback();
      state.cursorSource = state.loopIn;
      drawWaveform(state.cursorSource);
    } else if (event.key === "[") {
      adjustPoint("in", -1);
    } else if (event.key === "]") {
      adjustPoint("out", 1);
    } else if (event.key === "ArrowUp" && state.playing && engine.timeline) {
      event.preventDefault();
      const current = roundAtTime(engine.timeline, engine.now());
      if (current) {
        const index = (current.index + 1) % state.ladder.length;
        updateLadderValue(index, state.ladder[index] + 2);
        showToast(`下一轮 +2 BPM → ${state.ladder[index]} BPM`);
      }
    }
  });
}

function startAnimation() {
  let lastUpdate = 0;
  const frame = (timestamp) => {
    if (state.playing && timestamp - lastUpdate > 24) {
      updateTransportFromClock();
      lastUpdate = timestamp;
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

function init() {
  renderTrackList();
  bindControls();
  bindKeyboard();
  bindWaveformInteraction();
  startAnimation();
  const resizeObserver = new ResizeObserver(() => {
    drawWaveform(state.playing ? currentSourceTime() : state.cursorSource);
    $$(".track-preview canvas").forEach((canvas, index) => drawMiniWave(canvas, TRACKS[index], index));
  });
  resizeObserver.observe($("#wave-canvas-wrap"));
  loadTrack(0);
}

init();
