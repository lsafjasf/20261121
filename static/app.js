(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));
  const frameFormatter = new Intl.NumberFormat("en-US");
  const state = {
    sources: [],
    source: null,
    selection: { start: 0, end: 1 },
    view: { start: 0, end: 1 },
    freq: { min: 10000, max: 96000 },
    params: {
      mode: "heterodyne",
      center_hz: 70000,
      bandwidth_hz: 16000,
      sideband: "USB",
      factor: 10,
    },
    monitor: null,
    waveform: null,
    monitorDirty: true,
    cursorFrame: 0,
    specToken: 0,
    analysisToken: 0,
    selectionDrag: null,
    playing: false,
    exports: [],
    lastSelfTest: null,
  };

  const dom = {
    sourceList: $("source-list"),
    sourceCount: $("source-count"),
    sourceTitle: $("source-title"),
    crumbKind: $("crumb-kind"),
    ratePill: $("rate-pill"),
    durationPill: $("duration-pill"),
    channelPill: $("channel-pill"),
    sideRate: $("side-rate"),
    specCanvas: $("spectrogram-canvas"),
    waveCanvas: $("waveform-canvas"),
    freqAxis: $("frequency-axis"),
    timeAxis: $("time-axis"),
    plotArea: $("plot-area"),
    plotLoading: $("plot-loading"),
    plotEmpty: $("plot-empty"),
    viewRange: $("view-range"),
    selectionOverlay: $("selection-overlay"),
    selectionLabel: $("selection-label"),
    bandOverlay: $("band-overlay"),
    playhead: $("playhead"),
    playheadFlag: $("playhead-flag"),
    hoverCursor: $("hover-cursor"),
    startInput: $("start-input"),
    endInput: $("end-input"),
    startTime: $("start-time"),
    endTime: $("end-time"),
    cursorSource: $("cursor-source"),
    cursorMonitor: $("cursor-monitor"),
    sourceTime: $("source-time"),
    monitorTime: $("monitor-time"),
    renderIndicator: $("render-indicator"),
    renderButton: $("render-button"),
    playButton: $("play-button"),
    audio: $("monitor-audio"),
    modeTag: $("mode-tag"),
    heterodyneControls: $("heterodyne-controls"),
    expansionControls: $("expansion-controls"),
    centerRange: $("center-range"),
    centerValue: $("center-value"),
    bandwidthSelect: $("bandwidth-select"),
    sidebandSwitch: $("sideband-switch"),
    bandLow: $("band-low"),
    bandCenter: $("band-center"),
    bandHigh: $("band-high"),
    bandFill: $("band-fill"),
    bandCenterMark: $("band-center-mark"),
    bandCaption: $("band-caption"),
    heterodyneNote: $("heterodyne-note"),
    factorRange: $("factor-range"),
    factorValue: $("factor-value"),
    factorGrid: $("factor-grid"),
    ratioSourceDuration: $("ratio-source-duration"),
    ratioMonitorDuration: $("ratio-monitor-duration"),
    mappingFormula: $("mapping-formula"),
    mapStart: $("map-start"),
    mapEnd: $("map-end"),
    mapDuration: $("map-duration"),
    mapFrequency: $("map-frequency"),
    monitorReady: $("monitor-ready"),
    monitorFile: $("monitor-file"),
    monitorFrames: $("monitor-frames"),
    monitorCanvas: $("monitor-canvas"),
    monitorPlayhead: $("monitor-playhead"),
    monitorEnd: $("monitor-end"),
    exportHistory: $("export-history"),
    exportCount: $("export-count"),
    toastStack: $("toast-stack"),
    busyOverlay: $("busy-overlay"),
    busyTitle: $("busy-title"),
    busyDetail: $("busy-detail"),
    uploadCard: $("upload-card"),
    fileInput: $("file-input"),
    selftestModal: $("selftest-modal"),
    selftestBody: $("selftest-body"),
    selftestSource: $("selftest-source"),
    exportModal: $("export-modal"),
    exportBody: $("export-body"),
    exportFolder: $("export-folder"),
  };

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function formatTime(seconds, places = 3) {
    if (!Number.isFinite(seconds)) return "—";
    const safe = Math.max(0, seconds);
    const minutes = Math.floor(safe / 60);
    const remainder = safe - minutes * 60;
    return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(places).padStart(places + 3, "0")}`;
  }

  function formatSeconds(seconds) {
    if (!Number.isFinite(seconds)) return "—";
    return seconds < 1 ? `${(seconds * 1000).toFixed(1)} ms` : `${seconds.toFixed(3)} s`;
  }

  function formatKhz(hz) {
    return `${(hz / 1000).toFixed(hz % 1000 === 0 ? 0 : 1)} kHz`;
  }

  function formatFrames(frames) {
    return frameFormatter.format(Math.round(frames));
  }

  function sampleLabel(frame) {
    return `S${formatFrames(frame)}`;
  }

  function debounce(fn, delay) {
    let timer = null;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), delay);
    };
  }

  function postJson(url, payload) {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(async (response) => {
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      return data;
    });
  }

  function toast(message, kind = "info", duration = 3400) {
    const node = document.createElement("div");
    node.className = `toast ${kind === "error" ? "error" : ""}`;
    node.textContent = message;
    dom.toastStack.appendChild(node);
    setTimeout(() => node.remove(), duration);
  }

  function setBusy(active, title = "正在处理", detail = "请稍候") {
    dom.busyOverlay.hidden = !active;
    if (active) {
      dom.busyTitle.textContent = title;
      dom.busyDetail.textContent = detail;
    }
  }

  const sourceIcon = `
    <svg viewBox="0 0 24 24"><path d="M3 12h3l2-6 4 12 3-8 2 4h4"/></svg>`;

  const colorStops = [
    [0.0, [4, 9, 14]],
    [0.18, [8, 27, 37]],
    [0.37, [14, 58, 65]],
    [0.55, [24, 112, 104]],
    [0.72, [106, 155, 100]],
    [0.87, [225, 184, 87]],
    [1.0, [255, 244, 205]],
  ];
  const colorLut = new Uint8Array(256 * 3);
  for (let index = 0; index < 256; index += 1) {
    const value = index / 255;
    let right = 1;
    while (right < colorStops.length && colorStops[right][0] < value) right += 1;
    const left = Math.max(0, right - 1);
    const [aPos, aColor] = colorStops[left];
    const [bPos, bColor] = colorStops[Math.min(right, colorStops.length - 1)];
    const t = bPos === aPos ? 0 : (value - aPos) / (bPos - aPos);
    colorLut[index * 3] = Math.round(aColor[0] + (bColor[0] - aColor[0]) * t);
    colorLut[index * 3 + 1] = Math.round(aColor[1] + (bColor[1] - aColor[1]) * t);
    colorLut[index * 3 + 2] = Math.round(aColor[2] + (bColor[2] - aColor[2]) * t);
  }

  function syncCanvasSize(canvas) {
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
    const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    return { width, height, ratio };
  }

  function renderSourceList() {
    dom.sourceCount.textContent = state.sources.length;
    dom.sourceList.innerHTML = state.sources
      .map((source) => {
        const active = state.source && source.id === state.source.id ? " active" : "";
        const tag = source.kind === "calibration" ? "CAL" : source.kind === "demo" ? "DEMO" : "WAV";
        return `
          <button class="source-item${active}" data-source-id="${escapeHtml(source.id)}">
            <span class="source-icon">${sourceIcon}</span>
            <span class="source-copy">
              <b>${escapeHtml(source.name)}</b>
              <small><span>${tag}</span><span>${(source.rate / 1000).toFixed(1)} kHz</span><span>${source.duration.toFixed(2)} s</span></small>
            </span>
          </button>`;
      })
      .join("");
  }

  function updateSourceChrome() {
    const source = state.source;
    if (!source) return;
    dom.sourceTitle.textContent = source.name;
    dom.crumbKind.textContent =
      source.kind === "calibration" ? "CALIBRATION" : source.kind === "demo" ? "FIELD DEMO" : "LOCAL FILE";
    dom.ratePill.textContent = `${(source.rate / 1000).toFixed(1)} kHz`;
    dom.durationPill.textContent = `${source.duration.toFixed(3)} s`;
    dom.channelPill.textContent = `${source.channels} ch`;
    dom.sideRate.textContent = `${(source.rate / 1000).toFixed(1)} kHz`;
    dom.ratePill.classList.toggle("accent", source.rate >= 190000);
  }

  function resetMonitorView() {
    state.monitor = null;
    state.monitorDirty = true;
    dom.audio.pause();
    dom.audio.removeAttribute("src");
    dom.audio.load();
    dom.monitorReady.textContent = "STALE";
    dom.monitorReady.classList.remove("ready");
    dom.monitorFile.textContent = "—";
    dom.monitorFrames.textContent = "—";
    dom.monitorEnd.textContent = "0";
    dom.monitorPlayhead.style.left = "0%";
    dom.renderIndicator.classList.remove("ready", "dirty");
    dom.renderIndicator.querySelector("b").textContent = "未生成监听";
    clearMonitorWave();
  }

  async function selectSource(sourceId, options = {}) {
    const source = state.sources.find((item) => item.id === sourceId);
    if (!source) return;
    if (state.source && state.source.id === sourceId && !options.force) return;

    dom.audio.pause();
    state.playing = false;
    dom.playButton.classList.remove("playing");
    state.source = source;
    state.analysisToken += 1;
    state.specToken += 1;
    state.selection = {
      start: Math.round(source.default_selection.start_frame),
      end: Math.round(source.default_selection.end_frame),
    };
    state.view = { start: 0, end: source.frames };
    state.cursorFrame = state.selection.start;
    state.freq.max = Math.min(96000, source.rate / 2);
    state.freq.min = Math.min(10000, Math.max(0, state.freq.max - 10000));
    if (state.freq.max <= 48000) state.freq.min = 0;

    state.params.center_hz = clamp(
      state.params.center_hz,
      1000,
      Math.max(1500, source.rate / 2 - 500),
    );
    state.params.factor = clamp(Number(state.params.factor) || 10, 1, 50);

    resetMonitorView();
    renderSourceList();
    updateSourceChrome();
    updateFrequencyPreset();
    updateControlsUI();
    updateSelectionUI();
    updateMappingUI();
    drawAxes();
    fetchAnalysis();
    if (options.autoRender !== false) {
      setTimeout(() => renderMonitor({ quiet: true }), 90);
    }
  }

  function updateFrequencyPreset() {
    const select = $("frequency-preset");
    const matched = [...select.options].find((option) => {
      const [low, high] = option.value.split(",").map(Number);
      return Math.abs(low - state.freq.min) < 2 && Math.abs(high - state.freq.max) < 2;
    });
    if (matched) select.value = matched.value;
    else select.value = `${state.freq.min},${state.freq.max}`;
  }

  function fetchAnalysis() {
    if (!state.source) return;
    const token = ++state.analysisToken;
    const source = state.source;
    fetchSpectrogram(token);
    fetchWaveform(token);
    dom.viewRange.textContent =
      `${(state.view.start / source.rate).toFixed(2)}–${(state.view.end / source.rate).toFixed(2)} s`;
  }

  async function fetchSpectrogram(token) {
    if (!state.source) return;
    const source = state.source;
    const width = Math.max(700, Math.round(dom.plotArea.clientWidth * (window.devicePixelRatio || 1) * 0.82));
    const height = Math.max(260, Math.round(dom.plotArea.clientHeight * (window.devicePixelRatio || 1) * 0.82));
    const query = new URLSearchParams({
      source: source.id,
      start: String(Math.round(state.view.start)),
      end: String(Math.round(state.view.end)),
      fmin: String(state.freq.min),
      fmax: String(state.freq.max),
      width: String(Math.min(width, 2200)),
      height: String(Math.min(height, 760)),
    });
    dom.plotLoading.classList.add("visible");
    try {
      const response = await fetch(`/api/spectrogram?${query}`);
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || "频谱计算失败");
      }
      const raw = await response.arrayBuffer();
      if (token !== state.analysisToken) return;
      const imageWidth = Number(response.headers.get("X-Spec-Width"));
      const imageHeight = Number(response.headers.get("X-Spec-Height"));
      const bytes = new Uint8Array(raw);
      const imageData = new ImageData(imageWidth, imageHeight);
      for (let index = 0; index < bytes.length; index += 1) {
        const value = bytes[index];
        const offset = index * 4;
        imageData.data[offset] = colorLut[value * 3];
        imageData.data[offset + 1] = colorLut[value * 3 + 1];
        imageData.data[offset + 2] = colorLut[value * 3 + 2];
        imageData.data[offset + 3] = 255;
      }
      dom.specCanvas.width = imageWidth;
      dom.specCanvas.height = imageHeight;
      const context = dom.specCanvas.getContext("2d", { alpha: false });
      context.putImageData(imageData, 0, 0);
      dom.specCanvas.classList.add("ready");
      dom.plotEmpty.classList.add("hidden");
    } catch (error) {
      if (token === state.analysisToken) toast(error.message, "error");
    } finally {
      if (token === state.analysisToken) dom.plotLoading.classList.remove("visible");
    }
  }

  async function fetchWaveform(token) {
    if (!state.source) return;
    const source = state.source;
    const query = new URLSearchParams({
      source: source.id,
      start: String(Math.round(state.view.start)),
      end: String(Math.round(state.view.end)),
      width: "1400",
    });
    try {
      const response = await fetch(`/api/waveform?${query}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "波形读取失败");
      if (token !== state.analysisToken) return;
      state.waveform = data;
      drawWaveform(data);
    } catch (error) {
      if (token === state.analysisToken) console.warn(error);
    }
  }

  function drawWaveform(data) {
    const { width, height, ratio } = syncCanvasSize(dom.waveCanvas);
    const context = dom.waveCanvas.getContext("2d");
    context.clearRect(0, 0, width, height);
    const minimums = data.min;
    const maximums = data.max;
    const count = Math.min(minimums.length, maximums.length);
    if (!count) return;
    const center = height / 2;
    const radius = height * 0.39;
    context.beginPath();
    for (let x = 0; x < width; x += 1) {
      const index = Math.min(count - 1, Math.floor((x / width) * count));
      const upper = center - clamp(maximums[index], -1, 1) * radius;
      if (x === 0) context.moveTo(x, upper);
      else context.lineTo(x, upper);
    }
    for (let x = width - 1; x >= 0; x -= 1) {
      const index = Math.min(count - 1, Math.floor((x / width) * count));
      const lower = center - clamp(minimums[index], -1, 1) * radius;
      context.lineTo(x, lower);
    }
    context.closePath();
    context.fillStyle = "rgba(136, 224, 193, 0.30)";
    context.fill();
    context.strokeStyle = "rgba(183, 244, 220, 0.52)";
    context.lineWidth = Math.max(1, ratio * 0.55);
    context.stroke();
    context.beginPath();
    context.moveTo(0, center);
    context.lineTo(width, center);
    context.strokeStyle = "rgba(143,188,190,.12)";
    context.lineWidth = 1;
    context.stroke();
  }

  function clearMonitorWave() {
    const { width, height } = syncCanvasSize(dom.monitorCanvas);
    const context = dom.monitorCanvas.getContext("2d");
    context.clearRect(0, 0, width, height);
    context.beginPath();
    context.moveTo(0, height / 2);
    context.lineTo(width, height / 2);
    context.strokeStyle = "rgba(143,188,190,.13)";
    context.stroke();
  }

  function drawMonitorWave() {
    const peaks = state.monitor && state.monitor.wave_peaks;
    if (!peaks || !peaks.min.length) {
      clearMonitorWave();
      return;
    }
    const { width, height, ratio } = syncCanvasSize(dom.monitorCanvas);
    const context = dom.monitorCanvas.getContext("2d");
    context.clearRect(0, 0, width, height);
    const count = peaks.min.length;
    const center = height / 2;
    const radius = height * 0.4;
    context.beginPath();
    for (let x = 0; x < width; x += 1) {
      const index = Math.min(count - 1, Math.floor((x / width) * count));
      context.lineTo(x, center - clamp(peaks.max[index], -1, 1) * radius);
    }
    for (let x = width - 1; x >= 0; x -= 1) {
      const index = Math.min(count - 1, Math.floor((x / width) * count));
      context.lineTo(x, center - clamp(peaks.min[index], -1, 1) * radius);
    }
    context.closePath();
    context.fillStyle = "rgba(115,199,216,.30)";
    context.fill();
    context.strokeStyle = "rgba(136,224,193,.8)";
    context.lineWidth = Math.max(1, ratio * 0.48);
    context.stroke();
  }

  function drawAxes() {
    const source = state.source;
    if (!source) return;
    const frequencyTicks = 6;
    dom.freqAxis.innerHTML = "";
    for (let index = 0; index < frequencyTicks; index += 1) {
      const ratio = index / (frequencyTicks - 1);
      const frequency = state.freq.max - ratio * (state.freq.max - state.freq.min);
      const label = document.createElement("span");
      label.style.top = `${ratio * 100}%`;
      label.textContent = `${(frequency / 1000).toFixed(frequency < 10000 ? 1 : 0)}k`;
      dom.freqAxis.appendChild(label);
    }
    const duration = (state.view.end - state.view.start) / source.rate;
    const decimals = duration < 0.4 ? 3 : duration < 3 ? 2 : 1;
    dom.timeAxis.innerHTML = "";
    const timeTicks = window.innerWidth < 1280 ? 4 : 5;
    for (let index = 0; index < timeTicks; index += 1) {
      const ratio = index / (timeTicks - 1);
      const seconds = (state.view.start + ratio * (state.view.end - state.view.start)) / source.rate;
      const label = document.createElement("span");
      label.style.left = `${ratio * 100}%`;
      label.style.transform =
        index === 0 ? "none" : index === timeTicks - 1 ? "translateX(-100%)" : "translateX(-50%)";
      label.textContent = `${seconds.toFixed(decimals)} s`;
      dom.timeAxis.appendChild(label);
    }
    dom.viewRange.textContent = `${(state.view.start / source.rate).toFixed(3)}–${(
      state.view.end / source.rate
    ).toFixed(3)} s`;
  }

  function setDirty() {
    state.monitorDirty = true;
    state.monitor = state.monitor
      ? { ...state.monitor, stale: true }
      : null;
    dom.renderIndicator.classList.remove("ready");
    dom.renderIndicator.classList.add("dirty");
    dom.renderIndicator.querySelector("b").textContent = state.monitor
      ? "参数已更改"
      : "未生成监听";
    if (state.monitor) {
      dom.monitorReady.textContent = "STALE";
      dom.monitorReady.classList.remove("ready");
    }
  }

  function updateSelectionUI() {
    const source = state.source;
    if (!source) return;
    const start = Math.round(state.selection.start);
    const end = Math.round(state.selection.end);
    dom.startInput.value = String(start);
    dom.endInput.value = String(end);
    dom.startTime.textContent = formatTime(start / source.rate);
    dom.endTime.textContent = formatTime(end / source.rate);
    dom.selectionLabel.textContent = `${sampleLabel(start)} → ${sampleLabel(end)} · ${formatSeconds(
      (end - start) / source.rate,
    )}`;
    dom.ratioSourceDuration.textContent = formatSeconds((end - start) / source.rate);
    dom.ratioMonitorDuration.textContent = formatSeconds(
      ((end - start) / source.rate) * Number(state.params.factor),
    );
    updateSelectionOverlay();
    updateBandOverlay();
    updateCursorUI();
    updateMappingUI();
  }

  function updateSelectionOverlay() {
    if (!state.source) return;
    const span = state.view.end - state.view.start;
    if (span <= 0) return;
    const startRatio = (state.selection.start - state.view.start) / span;
    const endRatio = (state.selection.end - state.view.start) / span;
    const intersects = endRatio > 0 && startRatio < 1;
    dom.selectionOverlay.classList.toggle("visible", intersects);
    if (!intersects) return;
    const left = clamp(startRatio * 100, 0, 100);
    const right = clamp(endRatio * 100, 0, 100);
    dom.selectionOverlay.style.left = `${left}%`;
    dom.selectionOverlay.style.width = `${Math.max(0.05, right - left)}%`;
    dom.selectionLabel?.classList.toggle("outside", left <= 0.5);
  }

  function updateBandOverlay() {
    if (!state.source) return;
    if (state.params.mode !== "heterodyne") {
      dom.bandOverlay.classList.remove("visible");
      return;
    }
    const center = Number(state.params.center_hz);
    const bandwidth = Number(state.params.bandwidth_hz);
    const low = state.params.sideband === "USB" ? center : center - bandwidth;
    const high = state.params.sideband === "USB" ? center + bandwidth : center;
    const span = state.freq.max - state.freq.min;
    if (span <= 0 || high < state.freq.min || low > state.freq.max) {
      dom.bandOverlay.classList.remove("visible");
      return;
    }
    const top = ((state.freq.max - Math.min(high, state.freq.max)) / span) * 100;
    const bottom = ((state.freq.max - Math.max(low, state.freq.min)) / span) * 100;
    dom.bandOverlay.classList.add("visible");
    dom.bandOverlay.style.top = `${clamp(top, 0, 100)}%`;
    dom.bandOverlay.style.height = `${Math.max(1, clamp(bottom - top, 0, 100))}%`;
  }

  function updateControlsUI() {
    const source = state.source;
    if (!source) return;
    const isHeterodyne = state.params.mode === "heterodyne";
    dom.heterodyneControls.hidden = !isHeterodyne;
    dom.expansionControls.hidden = isHeterodyne;
    dom.modeTag.textContent = isHeterodyne ? "HET" : "TEXP";
    document.querySelectorAll(".mode-tab").forEach((button) => {
      button.classList.toggle("active", button.dataset.mode === state.params.mode);
    });

    const safeMaximum = Math.max(2000, source.rate / 2 - 500);
    dom.centerRange.min = "500";
    dom.centerRange.max = String(Math.round(safeMaximum));
    dom.centerRange.step = "250";
    dom.centerRange.value = String(state.params.center_hz);
    dom.centerValue.textContent = `${(state.params.center_hz / 1000).toFixed(1)} kHz`;
    dom.bandwidthSelect.value = String(state.params.bandwidth_hz);
    dom.sidebandSwitch.querySelectorAll("button").forEach((button) => {
      button.classList.toggle("active", button.dataset.sideband === state.params.sideband);
    });
    document.querySelectorAll(".quick-frequencies button").forEach((button) => {
      button.classList.toggle(
        "active",
        Number(button.dataset.frequency) === Number(state.params.center_hz),
      );
    });

    const center = Number(state.params.center_hz);
    const bandwidth = Number(state.params.bandwidth_hz);
    const low = state.params.sideband === "USB" ? center : center - bandwidth;
    const high = state.params.sideband === "USB" ? center + bandwidth : center;
    dom.bandLow.textContent = `${((center - bandwidth) / 1000).toFixed(0)}`;
    dom.bandCenter.textContent = `${(center / 1000).toFixed(0)}`;
    dom.bandHigh.textContent = `${((center + bandwidth) / 1000).toFixed(0)}`;
    dom.bandCaption.textContent = `${(low / 1000).toFixed(1)}–${(high / 1000).toFixed(1)} kHz`;
    if (state.params.sideband === "USB") {
      dom.bandFill.style.left = "50%";
      dom.bandFill.style.right = "0";
    } else {
      dom.bandFill.style.left = "0";
      dom.bandFill.style.right = "50%";
    }
    const upper = Math.min(high, source.rate / 2);
    dom.heterodyneNote.textContent = `听到的 0–${(
      Math.min(bandwidth, 24000) / 1000
    ).toFixed(0)} kHz 对应源信号 ${(low / 1000).toFixed(1)}–${(upper / 1000).toFixed(1)} kHz；反向回算不丢失时间位置。`;

    dom.factorRange.value = String(state.params.factor);
    dom.factorValue.textContent = `${Number(state.params.factor).toFixed(1)}×`;
    dom.factorGrid.querySelectorAll("button").forEach((button) => {
      button.classList.toggle("active", Number(button.dataset.factor) === Number(state.params.factor));
    });
    updateBandOverlay();
    updateSelectionUI();
  }

  function monitorSecondsForCursor() {
    if (!state.source) return null;
    const frame = state.cursorFrame;
    if (frame < state.selection.start || frame > state.selection.end) return null;
    const factor = state.params.mode === "time_expansion" ? Number(state.params.factor) : 1;
    return ((frame - state.selection.start) / state.source.rate) * factor;
  }

  function updateCursorUI() {
    const source = state.source;
    if (!source) return;
    const frame = clamp(Math.round(state.cursorFrame), 0, source.frames - 1);
    state.cursorFrame = frame;
    const monitorSeconds = monitorSecondsForCursor();
    dom.cursorSource.textContent = sampleLabel(frame);
    dom.cursorMonitor.textContent =
      monitorSeconds === null ? "区外" : `M ${formatTime(monitorSeconds)}`;
    dom.sourceTime.textContent = formatTime(frame / source.rate);
    if (state.monitor && dom.audio.currentTime >= 0) {
      dom.monitorTime.textContent = formatTime(dom.audio.currentTime);
    } else {
      dom.monitorTime.textContent = monitorSeconds === null ? "—" : formatTime(monitorSeconds);
    }
    const span = state.view.end - state.view.start;
    const ratio = (frame - state.view.start) / span;
    if (ratio >= 0 && ratio <= 1) {
      dom.playhead.classList.add("visible");
      dom.playhead.style.left = `${ratio * 100}%`;
      dom.playheadFlag.textContent = `S${formatFrames(frame)}`;
    } else {
      dom.playhead.classList.remove("visible");
    }
    if (state.monitor && state.monitor.monitor) {
      const monitorRatio = dom.audio.currentTime / state.monitor.monitor.duration;
      dom.monitorPlayhead.style.left = `${clamp(monitorRatio, 0, 1) * 100}%`;
    } else {
      dom.monitorPlayhead.style.left = "0%";
    }
  }

  function updateMappingUI() {
    const source = state.source;
    if (!source) return;
    const factor = state.params.mode === "time_expansion" ? Number(state.params.factor) : 1;
    const start = Math.round(state.selection.start);
    const end = Math.round(state.selection.end);
    const selectionDuration = (end - start) / source.rate;
    const predictedDuration = selectionDuration * factor;
    const duration = state.monitor ? state.monitor.monitor.duration : predictedDuration;
    dom.mappingFormula.textContent = `source = round(${start} + t × ${source.rate} / ${factor.toFixed(
      factor % 1 ? 1 : 0,
    )})`;
    dom.mapStart.textContent = sampleLabel(start);
    dom.mapEnd.textContent = sampleLabel(end);
    dom.mapDuration.textContent = formatSeconds(duration);
    if (state.params.mode === "heterodyne") {
      const sign = state.params.sideband === "USB" ? "+" : "−";
      dom.mapFrequency.textContent = `源频 = ${(state.params.center_hz / 1000).toFixed(1)}k ${sign} 监听频`;
    } else {
      dom.mapFrequency.textContent = `源频 = 监听频 × ${Number(state.params.factor).toFixed(0)}`;
    }
    dom.ratioSourceDuration.textContent = formatSeconds(selectionDuration);
    dom.ratioMonitorDuration.textContent = formatSeconds(predictedDuration);
    dom.monitorReady.textContent = state.monitor
      ? state.monitorDirty
        ? "STALE"
        : "READY"
      : "STALE";
    dom.monitorReady.classList.toggle("ready", Boolean(state.monitor && !state.monitorDirty));
    if (state.monitor) {
      dom.monitorFile.textContent = `${state.params.mode === "heterodyne" ? "het" : `slow${Number(
        factor,
      )}x`}_monitor.wav`;
      dom.monitorFrames.textContent = formatFrames(state.monitor.monitor.frames);
      dom.monitorEnd.textContent = formatTime(state.monitor.monitor.duration, 2);
    }
    if (state.monitorDirty) {
      dom.renderIndicator.classList.remove("ready");
      dom.renderIndicator.classList.add("dirty");
      dom.renderIndicator.querySelector("b").textContent = state.monitor
        ? "参数已更改"
        : "未生成监听";
    } else {
      dom.renderIndicator.classList.remove("dirty");
      dom.renderIndicator.classList.add("ready");
      dom.renderIndicator.querySelector("b").textContent = "监听已就绪";
    }
    updateCursorUI();
  }

  async function renderMonitor(options = {}) {
    if (!state.source || state.rendering) return null;
    const sourceId = state.source.id;
    state.rendering = true;
    dom.renderButton.disabled = true;
    dom.playButton.disabled = true;
    dom.renderIndicator.classList.remove("ready");
    dom.renderIndicator.classList.add("dirty");
    dom.renderIndicator.querySelector("b").textContent = "正在生成";
    if (!options.quiet) {
      setBusy(
        true,
        "渲染监听 WAV",
        state.params.mode === "heterodyne" ? "正在做边带选择与下变频…" : "正在按时间比例减速…",
      );
    }
    try {
      const data = await postJson("/api/render", {
        source_id: sourceId,
        start_frame: Math.round(state.selection.start),
        end_frame: Math.round(state.selection.end),
        params: { ...state.params },
      });
      if (!state.source || state.source.id !== sourceId) return null;
      state.monitor = data;
      state.monitorDirty = false;
      dom.audio.src = `${data.url}?v=${data.render_id}`;
      dom.audio.load();
      dom.monitorFile.textContent =
        state.params.mode === "heterodyne"
          ? "heterodyne_monitor.wav"
          : `slow${Number(state.params.factor)}x_monitor.wav`;
      dom.monitorFrames.textContent = formatFrames(data.monitor.frames);
      dom.monitorEnd.textContent = formatTime(data.monitor.duration, 2);
      dom.audio.currentTime = 0;
      drawMonitorWave();
      updateMappingUI();
      updateCursorUI();
      if (!options.quiet) {
        toast(`监听已生成 · ${formatTime(data.monitor.duration)} · ${formatFrames(data.monitor.frames)} 帧`);
      }
      if (options.autoplay) {
        dom.audio.addEventListener(
          "canplay",
          () => {
            if (options.autoplay) {
              dom.audio.play().catch((error) => toast(`播放失败：${error.message}`, "error"));
            }
          },
          { once: true },
        );
      }
      return data;
    } catch (error) {
      toast(error.message, "error");
      state.monitorDirty = true;
      return null;
    } finally {
      state.rendering = false;
      dom.renderButton.disabled = false;
      dom.playButton.disabled = false;
      if (!options.quiet) setBusy(false);
      updateMappingUI();
    }
  }

  async function togglePlayback() {
    if (!state.source) return;
    if (!dom.audio.paused) {
      dom.audio.pause();
      return;
    }
    if (!state.monitor || state.monitorDirty) {
      await renderMonitor({ quiet: true, autoplay: true });
      return;
    }
    if (state.monitor && dom.audio.currentTime >= state.monitor.monitor.duration - 0.01) {
      dom.audio.currentTime = 0;
    }
    dom.audio.play().catch((error) => toast(`播放失败：${error.message}`, "error"));
  }

  function updatePlaybackPosition() {
    if (!state.monitor || !state.source) return;
    const data = state.monitor;
    const factor = Number(data.time_factor);
    const seconds = clamp(dom.audio.currentTime, 0, data.monitor.duration);
    const sourceFrame =
      data.selection.start_frame + (seconds * data.source_rate) / Math.max(factor, 1e-9);
    state.cursorFrame = clamp(
      Math.round(sourceFrame),
      data.selection.start_frame,
      data.selection.end_frame,
    );
    updateCursorUI();
  }

  function playbackAnimation() {
    if (!state.playing) return;
    updatePlaybackPosition();
    requestAnimationFrame(playbackAnimation);
  }

  function setPlayingUi(playing) {
    state.playing = playing;
    dom.playButton.classList.toggle("playing", playing);
    dom.playButton.setAttribute("aria-label", playing ? "暂停" : "播放");
  }

  function seekMonitorFromEvent(event) {
    if (!state.monitor) return;
    const rect = $("monitor-wave").getBoundingClientRect();
    const ratio = clamp((event.clientX - rect.left) / rect.width, 0, 1);
    dom.audio.currentTime = ratio * state.monitor.monitor.duration;
    updatePlaybackPosition();
  }

  function setParameter(name, value, options = {}) {
    state.params[name] = value;
    if (options.pause !== false) {
      dom.audio.pause();
      setPlayingUi(false);
    }
    setDirty();
    updateControlsUI();
    if (options.toast) toast(options.toast);
  }

  function normalizeSelectionInputs() {
    if (!state.source) return;
    let start = Number.parseInt(dom.startInput.value.replaceAll(",", ""), 10);
    let end = Number.parseInt(dom.endInput.value.replaceAll(",", ""), 10);
    if (!Number.isFinite(start)) start = state.selection.start;
    if (!Number.isFinite(end)) end = state.selection.end;
    start = clamp(Math.round(start), 0, state.source.frames - 2);
    end = clamp(Math.round(end), start + 1, state.source.frames);
    state.selection = { start, end };
    state.cursorFrame = clamp(state.cursorFrame, start, end);
    setDirty();
    updateSelectionUI();
  }

  function nudgeSelection(edge, deltaFrames) {
    if (!state.source) return;
    const frame = Math.round(state.cursorFrame);
    if (edge === "start") {
      state.selection.start = clamp(frame, 0, state.selection.end - 1);
    } else {
      state.selection.end = clamp(frame, state.selection.start + 1, state.source.frames);
    }
    setDirty();
    updateSelectionUI();
  }

  function copyMapping() {
    if (!state.source) return;
    const factor = state.params.mode === "time_expansion" ? Number(state.params.factor) : 1;
    const text = [
      `source = round(${Math.round(state.selection.start)} + monitor_seconds * ${state.source.rate} / ${factor})`,
      `monitor_seconds = (source - ${Math.round(state.selection.start)}) * ${factor} / ${state.source.rate}`,
      `selection = [${Math.round(state.selection.start)}, ${Math.round(state.selection.end)})`,
      `rate = ${state.source.rate} Hz`,
      `mode = ${state.params.mode}`,
    ].join("\n");
    navigator.clipboard
      .writeText(text)
      .then(() => toast("映射公式已复制"))
      .catch(() => toast("浏览器未允许剪贴板访问", "error"));
  }

  function clientXToFrame(clientX) {
    if (!state.source) return 0;
    const rect = dom.plotArea.getBoundingClientRect();
    const ratio = clamp((clientX - rect.left) / Math.max(1, rect.width), 0, 1);
    return state.view.start + ratio * (state.view.end - state.view.start);
  }

  function placeCursorAt(clientX, options = {}) {
    if (!state.source) return;
    const frame = clamp(Math.round(clientXToFrame(clientX)), 0, state.source.frames - 1);
    state.cursorFrame = frame;
    if (
      options.seek !== false &&
      state.monitor &&
      !state.monitorDirty &&
      frame >= state.selection.start &&
      frame <= state.selection.end
    ) {
      const factor = Number(state.monitor.time_factor);
      dom.audio.currentTime = clamp(
        ((frame - state.selection.start) / state.source.rate) * factor,
        0,
        state.monitor.monitor.duration,
      );
    }
    updateCursorUI();
  }

  function setupPlotInteraction() {
    const overlay = dom.selectionOverlay;
    dom.plotArea.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || !state.source) return;
      event.preventDefault();
      const handle = event.target.closest(".selection-handle");
      const rect = dom.plotArea.getBoundingClientRect();
      const startX = event.clientX;
      const startFrame = clamp(clientXToFrame(startX), 0, state.source.frames);
      state.selectionDrag = {
        kind: handle ? handle.dataset.handle : "new",
        startX,
        startFrame,
        originStart: state.selection.start,
        originEnd: state.selection.end,
        moved: false,
        rect,
      };
      dom.plotArea.setPointerCapture(event.pointerId);
      if (!handle) placeCursorAt(startX, { seek: true });
    });

    dom.plotArea.addEventListener("pointermove", (event) => {
      if (!state.source) return;
      const rect = dom.plotArea.getBoundingClientRect();
      const hoverX = clamp(event.clientX - rect.left, 0, rect.width);
      dom.hoverCursor.classList.add("visible");
      dom.hoverCursor.style.left = `${hoverX}px`;
      const drag = state.selectionDrag;
      if (!drag) return;
      const delta = Math.abs(event.clientX - drag.startX);
      if (delta > 3) drag.moved = true;
      if (!drag.moved || drag.kind === "new") {
        if (!drag.moved) return;
        const currentFrame = clamp(clientXToFrame(event.clientX), 0, state.source.frames);
        state.selection.start = Math.min(startFrameSafe(drag.startFrame), currentFrame);
        state.selection.end = Math.max(startFrameSafe(drag.startFrame), currentFrame);
      } else if (drag.kind === "left") {
        state.selection.start = clamp(
          Math.round(clientXToFrame(event.clientX)),
          0,
          state.selection.end - 1,
        );
      } else if (drag.kind === "right") {
        state.selection.end = clamp(
          Math.round(clientXToFrame(event.clientX)),
          state.selection.start + 1,
          state.source.frames,
        );
      }
      const start = Math.round(state.selection.start);
      const end = Math.round(state.selection.end);
      if (end - start < 1) return;
      dom.startInput.value = String(start);
      dom.endInput.value = String(end);
      dom.startTime.textContent = formatTime(start / state.source.rate);
      dom.endTime.textContent = formatTime(end / state.source.rate);
      dom.selectionLabel.textContent = `${sampleLabel(start)} → ${sampleLabel(end)} · ${formatSeconds(
        (end - start) / state.source.rate,
      )}`;
      updateSelectionOverlay();
      updateMappingUI();
    });

    const finishDrag = () => {
      const drag = state.selectionDrag;
      if (!drag) return;
      state.selectionDrag = null;
      if (!drag.moved) return;
      if (state.selection.end - state.selection.start < 1) {
        state.selection.start = Math.max(0, state.selection.end - 1);
      }
      state.selection.start = Math.round(state.selection.start);
      state.selection.end = Math.round(state.selection.end);
      setDirty();
      updateSelectionUI();
    };
    dom.plotArea.addEventListener("pointerup", finishDrag);
    dom.plotArea.addEventListener("pointercancel", finishDrag);
    dom.plotArea.addEventListener("pointerleave", () => {
      if (!state.selectionDrag) dom.hoverCursor.classList.remove("visible");
    });
    dom.plotArea.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        const factor = event.deltaY < 0 ? 0.72 : 1.38;
        zoomView(factor, clientXToFrame(event.clientX));
      },
      { passive: false },
    );
    dom.plotArea.addEventListener("dblclick", () => {
      state.view = { start: 0, end: state.source.frames };
      refreshView();
    });
    dom.specCanvas.style.pointerEvents = "none";
    dom.waveCanvas.style.pointerEvents = "none";
  }

  function startFrameSafe(value) {
    return clamp(value, 0, state.source ? state.source.frames : value);
  }

  const scheduledViewRefresh = debounce(() => fetchAnalysis(), 130);

  function zoomView(multiplier, anchorFrame) {
    if (!state.source) return;
    const span = state.view.end - state.view.start;
    const minimumSpan = Math.min(state.source.frames, Math.max(256, state.source.rate * 0.002));
    const nextSpan = clamp(span * multiplier, minimumSpan, state.source.frames);
    const anchorRatio = span > 0 ? (anchorFrame - state.view.start) / span : 0.5;
    let start = anchorFrame - nextSpan * anchorRatio;
    let end = start + nextSpan;
    if (start < 0) {
      start = 0;
      end = nextSpan;
    }
    if (end > state.source.frames) {
      end = state.source.frames;
      start = end - nextSpan;
    }
    state.view = { start, end };
    drawAxes();
    updateSelectionOverlay();
    updateBandOverlay();
    updateCursorUI();
    scheduledViewRefresh();
  }

  function refreshView() {
    if (!state.source) return;
    drawAxes();
    updateSelectionOverlay();
    updateBandOverlay();
    updateCursorUI();
    fetchAnalysis();
  }

  function fitSelection() {
    if (!state.source) return;
    const duration = state.selection.end - state.selection.start;
    const margin = Math.max(duration * 0.08, state.source.rate * 0.002);
    state.view = {
      start: clamp(state.selection.start - margin, 0, state.source.frames),
      end: clamp(state.selection.end + margin, 0, state.source.frames),
    };
    if (state.view.end - state.view.start < 64) {
      state.view.end = Math.min(state.source.frames, state.view.start + 64);
    }
    refreshView();
  }

  const resizeHandler = debounce(() => {
    if (state.waveform) drawWaveform(state.waveform);
    drawMonitorWave();
    drawAxes();
    updateSelectionOverlay();
    updateCursorUI();
  }, 120);

  function openModal(element) {
    element.hidden = false;
    document.body.style.overflow = "hidden";
  }

  function closeModal(element) {
    element.hidden = true;
    if (dom.selftestModal.hidden && dom.exportModal.hidden) {
      document.body.style.overflow = "";
    }
  }

  function renderSelfTest(result) {
    dom.selftestSource.textContent = `${result.source_name} · 192 kHz`;
    const checks = result.checks || [];
    const sweep = result.sweep || {};
    const frequencies = sweep.frequencies_hz || [];
    const bars = frequencies
      .map((frequency, index) => {
        const height = 30 + (index / Math.max(1, frequencies.length - 1)) * 70;
        return `<i style="height:${height}%" title="${(frequency / 1000).toFixed(0)} kHz"></i>`;
      })
      .join("");
    dom.selftestBody.innerHTML = `
      <div class="selftest-summary ${result.passed ? "pass" : ""}">
        <span class="summary-ring">${result.passed ? "✓" : "!"}</span>
        <span class="summary-copy">
          <b>${escapeHtml(result.summary)}</b>
          <small>${sweep.tone_count || 0} 阶多频扫频 · 中心 ${(sweep.center_hz / 1000).toFixed(0)} kHz · 两个端点探针</small>
        </span>
      </div>
      <div class="selftest-sweep">
        <div class="sweep-head">
          <span>7 阶参考频标</span>
          <span>${(sweep.duration_s || 0).toFixed(3)} s</span>
        </div>
        <div class="sweep-track">${bars}</div>
        <div class="sweep-labels">
          <span>${frequencies.length ? (frequencies[0] / 1000).toFixed(0) : "—"} kHz</span>
          <span>${sweep.center_hz ? (sweep.center_hz / 1000).toFixed(0) : "—"} kHz</span>
          <span>${frequencies.length ? (frequencies[frequencies.length - 1] / 1000).toFixed(0) : "—"} kHz</span>
        </div>
      </div>
      <div class="check-list">
        ${checks
          .map(
            (check) => `
          <div class="check-row ${check.pass ? "" : "fail"}">
            <span class="check-mark">${check.pass ? "✓" : "×"}</span>
            <span class="check-copy"><b>${escapeHtml(check.label)}</b><small>${escapeHtml(check.detail)}</small></span>
            <span class="check-value">${escapeHtml(check.value)}</span>
          </div>`,
          )
          .join("")}
      </div>`;
  }

  async function runSelfTest() {
    openModal(dom.selftestModal);
    dom.selftestBody.innerHTML =
      '<div class="selftest-loading"><span class="busy-spinner"></span> 正在运行 7 阶扫频与端点探针…</div>';
    try {
      const response = await fetch(
        `/api/selftest?source_id=${encodeURIComponent(state.calibrationId || "")}`,
      );
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "自检失败");
      state.lastSelfTest = data.result;
      renderSelfTest(data.result);
    } catch (error) {
      dom.selftestBody.innerHTML = `<div class="selftest-loading">自检失败：${escapeHtml(error.message)}</div>`;
    }
  }

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes)) return "—";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function downloadUrl(url) {
    return `${url}${url.includes("?") ? "&" : "?"}download=1`;
  }

  function renderExportResult(exportData) {
    const files = exportData.files;
    const urls = exportData.urls;
    const mapping = exportData.mapping;
    const relation = mapping.source_frame_from_monitor_seconds;
    const inverse = mapping.monitor_seconds_from_source_frame;
    dom.exportFolder.textContent = `exports/${exportData.folder_name || exportData.export_id}/`;
    dom.exportBody.innerHTML = `
      <div class="export-success">
        <span class="success-mark">✓</span>
        <div><b>原始片段与监听 WAV 已锁定</b><small>${escapeHtml(exportData.export_id)}</small></div>
      </div>
      <div class="export-files">
        <div class="export-file">
          <div class="file-kind"><span>RAW CLIP</span><b>${escapeHtml(files.raw_clip.subtype || "FLOAT")} / ${(exportData.source.sample_rate / 1000).toFixed(1)}k</b></div>
          <strong>${escapeHtml(files.raw_clip.name)}</strong>
          <small>${formatBytes(files.raw_clip.bytes)} · ${formatFrames(exportData.selection.frames)} 帧 · 原采样率</small>
          <a class="download-link" href="${downloadUrl(urls.raw)}" download>↓ 下载原始片段</a>
        </div>
        <div class="export-file">
          <div class="file-kind"><span>MONITOR</span><b>PCM16 / 48k</b></div>
          <strong>${escapeHtml(files.monitor_clip.name)}</strong>
          <small>${formatBytes(files.monitor_clip.bytes)} · ${formatFrames(exportData.processing.monitor_frames)} 帧 · ${(exportData.processing.monitor_duration_seconds).toFixed(3)} s</small>
          <a class="download-link" href="${downloadUrl(urls.monitor)}" download>↓ 下载监听 WAV</a>
        </div>
      </div>
      <div class="export-map">
        <div class="export-map-head">
          <span>源样本 ↔ 监听时间对应</span>
          <span>${escapeHtml(exportData.processing.mode)} · ${exportData.mapping.time_factor}×</span>
        </div>
        <div class="endpoint-rail">
          <span>S${formatFrames(mapping.start.source_frame)} @ 0.000 s</span>
          <span>S${formatFrames(mapping.end_exclusive.source_frame)} @ ${mapping.end_exclusive.monitor_seconds.toFixed(3)} s</span>
        </div>
        <div class="map-relations">
          <span>回算</span><code>${escapeHtml(relation)}</code>
          <span>正算</span><code>${escapeHtml(inverse)}</code>
          <span>频率</span><code>${escapeHtml(mapping.frequency.source_hz)}</code>
        </div>
      </div>
      <div class="export-extra">
        <a href="${downloadUrl(urls.mapping)}" download>↓ mapping.csv</a>
        <a href="${downloadUrl(urls.manifest)}" download>↓ manifest.json</a>
        <a href="${urls.manifest}" target="_blank" rel="noopener">查看清单</a>
      </div>
      <details class="export-details">
        <summary>校验与文件清单</summary>
        <pre>${escapeHtml(JSON.stringify(files, null, 2))}</pre>
      </details>`;
  }

  async function runExport() {
    openModal(dom.exportModal);
    dom.exportBody.innerHTML =
      '<div class="selftest-loading"><span class="busy-spinner"></span> 正在写入原始片段、监听 WAV 与映射表…</div>';
    dom.exportFolder.textContent = "exports/";
    try {
      const data = await postJson("/api/export", {
        source_id: state.source.id,
        start_frame: Math.round(state.selection.start),
        end_frame: Math.round(state.selection.end),
        params: { ...state.params },
      });
      state.exports.unshift(data.export);
      renderExportHistory();
      renderExportResult(data.export);
    } catch (error) {
      dom.exportBody.innerHTML = `
        <div class="selftest-loading" style="flex-direction:column">
          导出失败：${escapeHtml(error.message)}
          <button class="button ghost" id="retry-export">重试</button>
        </div>`;
      $("retry-export")?.addEventListener("click", runExport);
    }
  }

  function renderExportHistory() {
    dom.exportCount.textContent = state.exports.length;
    if (!state.exports.length) {
      dom.exportHistory.innerHTML =
        '<p class="empty-copy">尚无导出。每次导出会同时写入原始片段、监听 WAV 与映射清单。</p>';
      return;
    }
    dom.exportHistory.innerHTML = state.exports
      .slice(0, 8)
      .map((item, index) => {
        const stamp = item.created_utc
          ? new Date(item.created_utc).toLocaleTimeString("zh-CN", {
              hour: "2-digit",
              minute: "2-digit",
            })
          : "—";
        return `
          <button class="export-item" data-export-id="${escapeHtml(item.export_id)}">
            <span class="export-item-index">${String(index + 1).padStart(2, "0")}</span>
            <span class="export-item-copy">
              <b>${escapeHtml(item.processing.mode)} · ${item.mapping.time_factor}×</b>
              <small>${stamp} · ${formatFrames(item.selection.frames)} src</small>
            </span>
            <span>›</span>
          </button>`;
      })
      .join("");
  }

  async function uploadSource(file) {
    if (!file) return;
    if (file.size > 512 * 1024 * 1024) {
      toast("文件超过 512 MB 限制", "error");
      return;
    }
    setBusy(true, "载入本机声源", `${file.name} · ${formatBytes(file.size)}`);
    try {
      const form = new FormData();
      form.append("file", file, file.name);
      const response = await fetch("/api/upload", { method: "POST", body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "上传失败");
      state.sources.push(data.source);
      renderSourceList();
      await selectSource(data.source.id, { force: true, autoRender: true });
      toast(`已载入 ${data.source.name} · ${(data.source.rate / 1000).toFixed(1)} kHz`);
    } catch (error) {
      toast(error.message, "error");
    } finally {
      setBusy(false);
      dom.fileInput.value = "";
    }
  }

  function bindEvents() {
    dom.sourceList.addEventListener("click", (event) => {
      const item = event.target.closest("[data-source-id]");
      if (item) selectSource(item.dataset.sourceId, { autoRender: true });
    });

    document.querySelectorAll(".mode-tab").forEach((button) => {
      button.addEventListener("click", () => {
        if (button.dataset.mode === state.params.mode) return;
        setParameter("mode", button.dataset.mode);
      });
    });

    dom.centerRange.addEventListener("input", () => {
      setParameter("center_hz", Number(dom.centerRange.value), { pause: true });
    });
    dom.bandwidthSelect.addEventListener("change", () => {
      setParameter("bandwidth_hz", Number(dom.bandwidthSelect.value));
    });
    dom.sidebandSwitch.addEventListener("click", (event) => {
      const button = event.target.closest("[data-sideband]");
      if (button) setParameter("sideband", button.dataset.sideband);
    });
    document.querySelectorAll(".quick-frequencies button").forEach((button) => {
      button.addEventListener("click", () => {
        setParameter("center_hz", Number(button.dataset.frequency));
      });
    });

    dom.factorGrid.addEventListener("click", (event) => {
      const button = event.target.closest("[data-factor]");
      if (button) setParameter("factor", Number(button.dataset.factor));
    });
    dom.factorRange.addEventListener("input", () => {
      setParameter("factor", Number(dom.factorRange.value), { pause: true });
    });

    [dom.startInput, dom.endInput].forEach((input) => {
      input.addEventListener("change", normalizeSelectionInputs);
      input.addEventListener("blur", normalizeSelectionInputs);
      input.addEventListener("keydown", (event) => {
        if (event.key === "Enter") input.blur();
      });
      input.addEventListener("focus", () => input.select());
    });
    $("set-start").addEventListener("click", () => nudgeSelection("start"));
    $("set-end").addEventListener("click", () => nudgeSelection("end"));

    $("zoom-in").addEventListener("click", () => {
      zoomView(0.68, (state.view.start + state.view.end) / 2);
    });
    $("zoom-out").addEventListener("click", () => {
      zoomView(1.55, (state.view.start + state.view.end) / 2);
    });
    $("fit-selection").addEventListener("click", fitSelection);
    $("full-view").addEventListener("click", () => {
      state.view = { start: 0, end: state.source.frames };
      refreshView();
    });
    $("frequency-preset").addEventListener("change", (event) => {
      const [minimum, maximum] = event.target.value.split(",").map(Number);
      state.freq = {
        min: Math.min(minimum, state.source.rate / 2 - 1000),
        max: Math.min(maximum, state.source.rate / 2),
      };
      if (state.freq.max - state.freq.min < 1000) state.freq.min = 0;
      drawAxes();
      updateBandOverlay();
      fetchAnalysis();
    });

    dom.renderButton.addEventListener("click", () => renderMonitor({ quiet: false }));
    dom.playButton.addEventListener("click", togglePlayback);
    $("copy-mapping").addEventListener("click", copyMapping);
    $("export-button").addEventListener("click", runExport);
    $("selftest-button").addEventListener("click", runSelfTest);
    $("rerun-selftest").addEventListener("click", runSelfTest);
    $("load-calibration").addEventListener("click", async () => {
      closeModal(dom.selftestModal);
      await selectSource(state.calibrationId, { force: true, autoRender: true });
      toast("已载入 192 kHz 多频扫频校准信号");
    });

    dom.audio.addEventListener("play", () => {
      setPlayingUi(true);
      requestAnimationFrame(playbackAnimation);
    });
    dom.audio.addEventListener("pause", () => {
      setPlayingUi(false);
      updatePlaybackPosition();
    });
    dom.audio.addEventListener("timeupdate", updatePlaybackPosition);
    dom.audio.addEventListener("seeked", updatePlaybackPosition);
    dom.audio.addEventListener("ended", () => {
      setPlayingUi(false);
      updatePlaybackPosition();
    });
    dom.audio.addEventListener("error", () => {
      if (dom.audio.getAttribute("src")) toast("监听音频加载失败", "error");
    });

    $("monitor-wave").addEventListener("click", seekMonitorFromEvent);
    dom.exportHistory.addEventListener("click", (event) => {
      const item = event.target.closest("[data-export-id]");
      if (!item) return;
      const exportData = state.exports.find((entry) => entry.export_id === item.dataset.exportId);
      if (!exportData) return;
      openModal(dom.exportModal);
      renderExportResult(exportData);
    });

    dom.fileInput.addEventListener("change", () => uploadSource(dom.fileInput.files[0]));
    ["dragenter", "dragover"].forEach((name) => {
      document.addEventListener(name, (event) => {
        event.preventDefault();
        if (event.dataTransfer?.types?.includes("Files")) dom.uploadCard.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach((name) => {
      document.addEventListener(name, (event) => {
        event.preventDefault();
        if (name === "drop" && event.dataTransfer?.files?.[0]) {
          uploadSource(event.dataTransfer.files[0]);
        }
        dom.uploadCard.classList.remove("dragover");
      });
    });

    document.querySelectorAll("[data-close-modal]").forEach((element) => {
      element.addEventListener("click", () => {
        closeModal(dom.selftestModal);
        closeModal(dom.exportModal);
      });
    });

    document.addEventListener("keydown", (event) => {
      const tag = event.target.tagName;
      const typing = tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA";
      if (event.key === "Escape") {
        closeModal(dom.selftestModal);
        closeModal(dom.exportModal);
      }
      if (typing) return;
      if (event.code === "Space") {
        event.preventDefault();
        togglePlayback();
      }
      if (event.key.toLowerCase() === "r") renderMonitor({ quiet: false });
    });

    window.addEventListener("resize", resizeHandler);
    dom.monitorCanvas.addEventListener("dblclick", () => {
      if (state.monitor) {
        dom.audio.currentTime = 0;
        updatePlaybackPosition();
      }
    });
  }

  async function bootstrap() {
    try {
      const response = await fetch("/api/bootstrap");
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "服务初始化失败");
      state.sources = data.sources || [];
      state.calibrationId = data.calibration_source_id;
      state.exports = data.exports || [];
      state.params = {
        ...state.params,
        mode: data.defaults.mode,
        center_hz: data.defaults.center_hz,
        bandwidth_hz: data.defaults.bandwidth_hz,
        sideband: data.defaults.sideband,
        factor: data.defaults.factor,
      };
      state.freq = { min: data.defaults.f_min, max: data.defaults.f_max };
      renderSourceList();
      renderExportHistory();
      await selectSource(data.default_source_id, { autoRender: true });
    } catch (error) {
      dom.sourceTitle.textContent = "无法启动筛查台";
      toast(error.message, "error", 10000);
      console.error(error);
    }
  }

  setupPlotInteraction();
  bindEvents();
  bootstrap();
})();
