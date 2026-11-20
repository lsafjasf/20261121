"use strict";

const state = {
  report: null,
  reelIndex: 0,
  activeTrack: "main",   // main | hi
  playing: false,
};

const el = {
  reelList: document.getElementById("reel-list"),
  reelTitle: document.getElementById("reel-title"),
  reelStats: document.getElementById("reel-stats"),
  playlistMeta: document.getElementById("playlist-meta"),
  btnMain: document.getElementById("btn-main"),
  btnHi: document.getElementById("btn-hi"),
  btnPlay: document.getElementById("btn-play"),
  seek: document.getElementById("seek"),
  posText: document.getElementById("pos-text"),
  audioMain: document.getElementById("audio-main"),
  audioHi: document.getElementById("audio-hi"),
  waveMain: document.getElementById("wave-main"),
  waveHi: document.getElementById("wave-hi"),
  markers: document.getElementById("issue-markers"),
  cards: document.getElementById("issue-cards"),
};

function currentReel() {
  return state.report.results[state.reelIndex];
}

function audioFor(track) {
  return track === "main" ? el.audioMain : el.audioHi;
}

async function boot() {
  const resp = await fetch("/api/report");
  if (!resp.ok) {
    document.body.innerHTML = "<p style='padding:24px'>报告加载失败，请先运行审片。</p>";
    return;
  }
  state.report = await resp.json();
  renderPlaylist();
  selectReel(0);
  bindControls();
}

function renderPlaylist() {
  const pl = state.report.playlist;
  el.playlistMeta.textContent =
    `${pl.name} · 参考语言 ${pl.reference.expected_lang} · ${pl.reels.length} 卷`;
  el.reelList.innerHTML = "";
  state.report.results.forEach((reel, idx) => {
    const li = document.createElement("li");
    const count = reel.issues.length;
    li.className = `reel-item ${count ? "fault" : "clean"}${idx === 0 ? " active" : ""}`;
    li.innerHTML =
      `<span><span class="dot"></span>${reel.reel_id}</span>` +
      `<span class="badge ${count ? "" : "zero"}">${count ? count + " 问题" : "正常"}</span>`;
    li.addEventListener("click", () => selectReel(idx));
    el.reelList.appendChild(li);
  });
}

function selectReel(idx) {
  // 切换卷前停止播放，避免跨卷串音
  pauseBoth();
  state.reelIndex = idx;
  state.activeTrack = "main";
  const reel = currentReel();
  document.querySelectorAll(".reel-item").forEach((node, i) =>
    node.classList.toggle("active", i === idx));
  el.reelTitle.textContent = `卷 ${reel.reel_id}`;
  el.reelStats.textContent =
    `时长 ${reel.duration_sec}s · 时延 ${reel.delay_ms} ms · ` +
    `相关 ${reel.best_corr} · 检测语言 ${reel.detected_lang || "未识别"}` +
    `（参考 ${reel.expected_lang}）`;

  const meta = state.report.playlist.reels[idx];
  el.audioMain.src = `/audio/${meta.main}`;
  el.audioHi.src = `/audio/${meta.hearing}`;
  el.seek.value = 0;
  el.posText.textContent = "0.00s";
  drawWaveform(el.waveMain, reel.waveform_main, "#4ea1ff");
  drawWaveform(el.waveHi, reel.waveform_hi, "#46c98d");
  drawMarkers(reel);
  renderCards(reel);
  syncTrackButtons();
}

function drawWaveform(canvas, data, color) {
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 700;
  const height = Number(canvas.getAttribute("height"));
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);
  const bins = data.min.length;
  const mid = height / 2;
  const step = width / bins;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < bins; i++) {
    const x = i * step;
    const yMin = mid + data.min[i] * (mid - 4);
    const yMax = mid + data.max[i] * (mid - 4);
    ctx.moveTo(x, yMin);
    ctx.lineTo(x, yMax);
  }
  ctx.stroke();
}

function drawMarkers(reel) {
  el.markers.innerHTML = "";
  reel.issues.forEach((issue) => {
    const pct = (issue.start_sec / reel.duration_sec) * 100;
    const mark = document.createElement("div");
    mark.className = "marker";
    mark.style.left = `${pct}%`;
    mark.title = `${issue.category} @ ${issue.start_sec}s`;
    el.markers.appendChild(mark);
  });
}

function bindControls() {
  el.btnMain.addEventListener("click", () => switchTrack("main"));
  el.btnHi.addEventListener("click", () => switchTrack("hi"));
  el.btnPlay.addEventListener("click", togglePlay);
  el.seek.addEventListener("input", onSeek);
  [el.audioMain, el.audioHi].forEach((audio) => {
    audio.addEventListener("timeupdate", onTimeUpdate);
    audio.addEventListener("ended", pauseBoth);
  });
}

function switchTrack(track) {
  if (track === state.activeTrack) return;
  const wasPlaying = state.playing;
  const oldAudio = audioFor(state.activeTrack);
  const position = oldAudio.currentTime;   // 保留播放位置
  pauseBoth();
  state.activeTrack = track;
  const nextAudio = audioFor(track);
  nextAudio.currentTime = position;
  syncTrackButtons();
  if (wasPlaying) {
    nextAudio.play().then(() => {
      state.playing = true;
      el.btnPlay.textContent = "暂停";
    }).catch(() => {});
  }
}

function syncTrackButtons() {
  el.btnMain.classList.toggle("active", state.activeTrack === "main");
  el.btnHi.classList.toggle("active", state.activeTrack === "hi");
}

function togglePlay() {
  const audio = audioFor(state.activeTrack);
  if (state.playing) {
    pauseBoth();
  } else {
    audio.play().then(() => {
      state.playing = true;
      el.btnPlay.textContent = "暂停";
    }).catch(() => {});
  }
}

function pauseBoth() {
  el.audioMain.pause();
  el.audioHi.pause();
  state.playing = false;
  el.btnPlay.textContent = "播放";
}

function onSeek() {
  const reel = currentReel();
  const t = (Number(el.seek.value) / 1000) * reel.duration_sec;
  el.audioMain.currentTime = t;
  el.audioHi.currentTime = t;
  el.posText.textContent = `${t.toFixed(2)}s`;
}

function onTimeUpdate() {
  const audio = audioFor(state.activeTrack);
  const reel = currentReel();
  if (!reel) return;
  el.posText.textContent = `${audio.currentTime.toFixed(2)}s`;
  el.seek.value = (audio.currentTime / reel.duration_sec) * 1000;
}

function renderCards(reel) {
  el.cards.innerHTML = "";
  if (!reel.issues.length) {
    el.cards.innerHTML = "<p style='color:var(--muted)'>本卷未发现问题。</p>";
    return;
  }
  reel.issues.forEach((issue) => {
    const card = document.createElement("div");
    card.className = `card ${issue.category === "LANG_MISMATCH" ? "lang"
      : issue.category === "DELAY" ? "delay" : ""}`;
    const ev = JSON.stringify(issue.evidence, null, 0);
    card.innerHTML =
      `<h4><span class="tag ${issue.category.startsWith("DUP") ? "dup"
        : issue.category === "DELAY" ? "delay" : "lang"}">${issue.category}</span>` +
      `<span class="time">${issue.start_sec.toFixed(2)}s – ${issue.end_sec.toFixed(2)}s</span></h4>` +
      `<p>${issue.detail}</p>` +
      `<div class="ev">证据数据：${ev}</div>` +
      `<div class="clips">
         <label style="font-size:12px;color:var(--muted)">助听片段</label>
         <audio controls preload="none" src="/${issue.clip_hi}"></audio>
         <label style="font-size:12px;color:var(--muted)">主声片段</label>
         <audio controls preload="none" src="/${issue.clip_main}"></audio>
       </div>`;
    el.cards.appendChild(card);
  });
}

window.addEventListener("resize", () => {
  const reel = currentReel();
  if (!reel) return;
  drawWaveform(el.waveMain, reel.waveform_main, "#4ea1ff");
  drawWaveform(el.waveHi, reel.waveform_hi, "#46c98d");
});

boot();
