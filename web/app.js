const state = {
  report: null,
  selected: 0,
  audioFile: null,
  manifestFile: null,
  audioUrl: "",
  objectUrl: "",
  animation: 0,
};

const $ = (selector) => document.querySelector(selector);
const audio = $("#audio-player");

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatTime(seconds) {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const minutes = Math.floor(safe / 60);
  const remainder = safe - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(3).padStart(6, "0")}`;
}

function formatHz(value) {
  return `${(value / 1000).toFixed(3)} kHz`;
}

function signedMs(value) {
  if (value === null || value === undefined) return "—";
  const sign = value > 0 ? "+" : value < 0 ? "−" : "";
  return `${sign}${Math.abs(value).toFixed(1)} ms`;
}

function showToast(message, error = false) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.toggle("error", error);
  toast.classList.add("visible");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("visible"), 3600);
}

function setBusy(busy) {
  $("#progress").hidden = !busy;
  $("#analyze-button").disabled = busy || !state.audioFile;
  $("#sample-button").disabled = busy;
  $("#analyze-button").textContent = busy ? "正在分析…" : "开始离线分析";
}

function setAudioSource(url) {
  if (state.objectUrl && state.objectUrl !== url) URL.revokeObjectURL(state.objectUrl);
  state.audioUrl = url;
  audio.src = url;
  audio.load();
  $("#play-button").textContent = "▶";
}

function selectDetection(index) {
  state.selected = index;
  document.querySelectorAll("#result-body tr").forEach((row) => {
    row.classList.toggle("selected", Number(row.dataset.index) === index);
  });
  renderEvidence();
  drawTimeline();
}

function renderHeader(report, uploaded = false) {
  const duration = report.source.duration_seconds;
  $("#report-title").textContent = uploaded
    ? `录音分析 / ${report.source.name}`
    : "馆藏样例 / 混响录音";
  $("#report-subtitle").textContent =
    `${report.protocol.name} · ${report.source.sample_rate.toLocaleString()} Hz · ` +
    `${duration.toFixed(2)} 秒 · 重复标记检测`;
  $("#engine-state").textContent = report.protocol.name;
  if (report.generated_at) {
    const stamp = new Date(report.generated_at);
    $("#report-stamp").textContent = stamp.toLocaleString("zh-CN", {
      month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
    });
  } else {
    $("#report-stamp").textContent = "LOCAL SESSION";
  }
  $("#footer-meta").textContent =
    `OFFLINE / PCM ${report.source.sample_rate.toLocaleString()} Hz / ${report.protocol.name}`;
  $("#audio-duration").textContent = formatTime(duration);
  $("#audio-time").textContent = formatTime(0);
  $("#scrub-fill").style.width = "0%";
}

function renderMetrics(report) {
  const summary = report.summary;
  $("#metric-hits").textContent = String(summary.hits).padStart(2, "0");
  $("#metric-candidates").textContent = String(summary.candidates).padStart(2, "0");
  $("#metric-fp").textContent =
    summary.false_positives === null || summary.false_positives === undefined
      ? "—"
      : String(summary.false_positives).padStart(2, "0");
  $("#metric-fp").nextElementSibling.textContent =
    summary.false_positives === null || summary.false_positives === undefined
      ? "缺少真值"
      : "无真值对应";
  $("#metric-offset").textContent =
    summary.timing_mae_ms === null || summary.timing_mae_ms === undefined
      ? "—"
      : `${summary.timing_mae_ms.toFixed(1)} ms`;
  $("#metric-offset").nextElementSibling.textContent =
    summary.timing_mae_ms === null ? "缺少真值" : "平均绝对误差";
}

function renderRegistry(report) {
  const hitsByAsset = new Map();
  report.detections.forEach((item) => {
    if (item.kind === "hit") {
      hitsByAsset.set(item.asset_id, (hitsByAsset.get(item.asset_id) || 0) + 1);
    }
  });
  const assets = report.protocol.known_assets;
  $("#registry-count").textContent = String(assets.length).padStart(2, "0");
  $("#registry-list").innerHTML = assets.map((asset) => `
    <div class="registry-item">
      <i></i>
      <div><b>${escapeHtml(asset.label)}</b><small>${escapeHtml(asset.title)}</small></div>
      <em>${hitsByAsset.get(asset.id) || 0}×</em>
    </div>
  `).join("");
}

function renderDamage(report) {
  const damage = report.damage_chain;
  if (!damage) {
    $("#damage-strip").innerHTML =
      '<span class="chain-item"><b>无真值清单</b> · 仍可识别标记，但不计算时标偏差</span>';
    return;
  }
  const speed = Number(damage.speed_factor);
  const parts = [
    ["源采样", `${(damage.downsample_from_hz / 1000).toFixed(1)} kHz`],
    ["降采样", `${(damage.output_rate_hz / 1000).toFixed(3)} kHz`],
    ["混响", `${Math.round(Number(damage.reverb_wet) * 100)}%`],
    ["变速", `${speed.toFixed(3)}×`],
    ["信噪比", `${Number(damage.snr_db).toFixed(0)} dB`],
  ];
  $("#damage-strip").innerHTML =
    '<span class="chain-item"><b>损伤链</b></span>' +
    parts.map(([label, value]) =>
      `<span class="chain-arrow">→</span><span class="chain-item">${label} <b>${escapeHtml(value)}</b></span>`
    ).join("");
}

function renderResults(report) {
  const body = $("#result-body");
  if (!report.detections.length) {
    body.innerHTML = '<tr><td colspan="7" class="muted-cell">未发现同步可信的编码标记</td></tr>';
    renderEvidence();
    return;
  }
  body.innerHTML = report.detections.map((item, index) => {
    const truth = report.evaluation?.ground_truth?.find(
      (entry) => entry.index === item.matched_truth_index,
    );
    const kindText = item.kind === "hit" ? "确认命中" : "损坏候选";
    const crcClass = item.data.crc_ok ? "ok" : "bad";
    const crcText = item.data.crc_ok ? "通过" : "失败";
    const deltaClass = (item.timing_offset_ms || 0) < 0 ? "negative" : "";
    const expected = truth
      ? `<span class="muted-cell">真值 #${truth.index + 1} · ${escapeHtml(truth.label)}</span>`
      : "";
    return `
      <tr data-index="${index}" tabindex="0">
        <td>${formatTime(item.marker_seconds)}</td>
        <td><span class="verdict-pill ${item.kind}">${kindText}</span></td>
        <td class="asset">${escapeHtml(item.asset_label)}<br>${expected}</td>
        <td class="frequency-cell">
          <b>${formatHz(item.carrier.leader_hz)}</b><br>
          ${formatHz(item.carrier.space_hz)} / ${formatHz(item.carrier.mark_hz)}
        </td>
        <td>${item.sync.hamming} / 16</td>
        <td class="${crcClass}">${crcText}</td>
        <td class="delta ${deltaClass}">${signedMs(item.timing_offset_ms)}</td>
      </tr>
    `;
  }).join("");

  body.querySelectorAll("tr").forEach((row) => {
    const activate = () => selectDetection(Number(row.dataset.index));
    row.addEventListener("click", activate);
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        activate();
      }
    });
  });

  const preferred = report.detections.findIndex((item) => item.kind === "candidate");
  selectDetection(preferred >= 0 ? preferred : 0);
}

function carrierBar(label, value, scale) {
  const width = Math.max(2, Math.min(100, (value / scale) * 100));
  return `
    <div class="carrier-bar">
      <span>${label}</span><span class="bar"><i style="width:${width}%"></i></span>
      <b>${Number(value).toFixed(1)} Hz</b>
    </div>
  `;
}

function renderEvidence() {
  const report = state.report;
  if (!report || !report.detections.length) {
    $("#evidence-title").textContent = "没有检测结果";
    $("#evidence-verdict").textContent = "—";
    $("#evidence-verdict").className = "verdict";
    $("#carrier-leader").textContent = "—";
    $("#carrier-bars").innerHTML = "";
    $("#carrier-facts").innerHTML = "";
    $("#sync-word").textContent = "—";
    $("#sync-bits").innerHTML = "";
    $("#sync-facts").innerHTML = "";
    $("#data-bits").textContent = "—";
    $("#crc-expected").textContent = "—";
    $("#crc-received").textContent = "—";
    $("#candidate-note").hidden = true;
    return;
  }
  const item = report.detections[state.selected] || report.detections[0];
  const carrier = item.carrier;
  const sync = item.sync;
  const data = item.data;
  const verdict = $("#evidence-verdict");
  verdict.className = `verdict ${item.kind}`;
  verdict.textContent = item.kind === "hit" ? "CRC VERIFIED / HIT" : "CANDIDATE ONLY";
  $("#evidence-title").textContent =
    `${item.asset_label} · ${item.asset_title}`;
  $("#carrier-leader").textContent = formatHz(carrier.leader_hz);
  $("#carrier-bars").innerHTML = [
    carrierBar("引导", carrier.leader_hz, 3000),
    carrierBar("SPACE 0", carrier.space_hz, 3000),
    carrierBar("MARK 1", carrier.mark_hz, 3000),
  ].join("");
  $("#carrier-facts").innerHTML = `
    <div><dt>载波锁定</dt><dd>${carrier.locked ? "LOCKED" : "UNSTABLE"}</dd></div>
    <div><dt>载波裕量</dt><dd>${Number(carrier.contrast_db).toFixed(1)} dB</dd></div>
    <div><dt>测得变速</dt><dd>${Number(carrier.speed_factor).toFixed(3)}×</dd></div>
    <div><dt>标记起点</dt><dd>${formatTime(item.marker_seconds)}</dd></div>
  `;

  const syncBits = parseInt(sync.word, 16).toString(2).padStart(16, "0").split("");
  $("#sync-word").textContent = sync.word;
  $("#sync-bits").innerHTML = syncBits.map((bit) => `
    <span class="bit-cell ${bit === "1" ? "one" : ""}">${bit}</span>
  `).join("");
  $("#sync-facts").innerHTML = `
    <div><dt>同步锁定</dt><dd>${sync.locked ? "SYNCED" : "UNLOCKED"}</dd></div>
    <div><dt>Hamming 距离</dt><dd>${sync.hamming} / 16</dd></div>
    <div><dt>软判决裕量</dt><dd>${Number(sync.soft_margin).toFixed(3)}</dd></div>
    <div><dt>同步时标</dt><dd>${formatTime(item.sync_seconds)}</dd></div>
  `;

  const bodyHex = data.body_hex.match(/.{1,2}/g)?.join(" ") || "—";
  $("#data-bits").textContent = bodyHex;
  const expected = $("#crc-expected");
  const received = $("#crc-received");
  expected.textContent = data.crc_expected;
  expected.className = data.crc_ok ? "ok" : "bad";
  received.textContent = data.crc_received;
  received.className = data.crc_ok ? "ok" : "bad";
  const note = $("#candidate-note");
  if (item.kind === "candidate") {
    note.hidden = false;
    note.textContent =
      `${item.candidate_reason}。编号 ${item.asset_label} 未被写入确认账本。`;
  } else {
    note.hidden = true;
    note.textContent = "";
  }
}

function renderTimelineLabels(duration) {
  const labels = $("#timeline-labels");
  const count = 6;
  labels.innerHTML = Array.from({ length: count }, (_, index) => {
    const ratio = index / (count - 1);
    return `<span class="timeline-label" style="left:${ratio * 100}%">${(duration * ratio).toFixed(1)}s</span>`;
  }).join("");
}

function renderReport(report, audioUrl, uploaded = false) {
  state.report = report;
  state.selected = 0;
  renderHeader(report, uploaded);
  renderMetrics(report);
  renderRegistry(report);
  renderDamage(report);
  setAudioSource(audioUrl);
  renderTimelineLabels(report.source.duration_seconds);
  renderResults(report);
  requestAnimationFrame(drawTimeline);
}

function drawTimeline() {
  const report = state.report;
  const canvas = $("#timeline");
  if (!report || !canvas) return;
  const bounds = canvas.getBoundingClientRect();
  if (bounds.width < 20) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (canvas.width !== Math.round(bounds.width * dpr) || canvas.height !== Math.round(bounds.height * dpr)) {
    canvas.width = Math.round(bounds.width * dpr);
    canvas.height = Math.round(bounds.height * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const width = bounds.width;
  const height = bounds.height;
  const center = height * 0.50;
  const duration = report.source.duration_seconds || 1;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "rgba(7, 11, 9, 0.45)";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(224, 218, 197, 0.065)";
  ctx.lineWidth = 1;
  for (let index = 0; index <= 6; index += 1) {
    const x = Math.round((index / 6) * width) + 0.5;
    ctx.beginPath();
    ctx.moveTo(x, 10);
    ctx.lineTo(x, height - 24);
    ctx.stroke();
  }
  ctx.strokeStyle = "rgba(224, 218, 197, 0.16)";
  ctx.beginPath();
  ctx.moveTo(0, center + 0.5);
  ctx.lineTo(width, center + 0.5);
  ctx.stroke();

  const peaks = report.overview || [];
  const step = width / Math.max(peaks.length, 1);
  ctx.fillStyle = "rgba(193, 188, 172, 0.27)";
  peaks.forEach((peak, index) => {
    const barHeight = Math.max(1, peak * (height - 52));
    ctx.fillRect(index * step, center - barHeight / 2, Math.max(0.7, step * 0.72), barHeight);
  });

  const truth = report.evaluation?.ground_truth || [];
  truth.forEach((item) => {
    const x = (item.expected_seconds / duration) * width;
    ctx.save();
    ctx.setLineDash([2, 4]);
    ctx.strokeStyle = "rgba(145, 143, 132, 0.48)";
    ctx.beginPath();
    ctx.moveTo(x, height - 21);
    ctx.lineTo(x, height - 8);
    ctx.stroke();
    ctx.restore();
  });

  report.detections.forEach((item, index) => {
    const x = (item.marker_seconds / duration) * width;
    const hit = item.kind === "hit";
    const color = hit ? "#76c8bb" : "#d6aa62";
    ctx.save();
    if (!hit) ctx.setLineDash([3, 4]);
    ctx.strokeStyle = color;
    ctx.globalAlpha = state.selected === index ? 1 : 0.72;
    ctx.lineWidth = state.selected === index ? 1.5 : 1;
    ctx.beginPath();
    ctx.moveTo(x, 16);
    ctx.lineTo(x, height - 18);
    ctx.stroke();
    ctx.restore();

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, 16, state.selected === index ? 3.2 : 2.2, 0, Math.PI * 2);
    ctx.fill();
    if (state.selected === index) {
      ctx.strokeStyle = "rgba(255,255,255,.55)";
      ctx.beginPath();
      ctx.arc(x, 16, 5.5, 0, Math.PI * 2);
      ctx.stroke();
    }
  });

  if (Number.isFinite(audio.currentTime) && audio.duration > 0) {
    const x = (audio.currentTime / audio.duration) * width;
    ctx.strokeStyle = "rgba(240, 199, 119, 0.95)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x + 0.5, 0);
    ctx.lineTo(x + 0.5, height);
    ctx.stroke();
    ctx.fillStyle = "#f0c777";
    ctx.beginPath();
    ctx.moveTo(x - 3, 0);
    ctx.lineTo(x + 3, 0);
    ctx.lineTo(x, 4);
    ctx.closePath();
    ctx.fill();
  }
}

function animateTimeline() {
  drawTimeline();
  if (!audio.paused) state.animation = requestAnimationFrame(animateTimeline);
}

async function loadSample() {
  setBusy(true);
  let report = null;
  let staticMode = false;
  try {
    const response = await fetch("/api/sample", { cache: "no-store" });
    if (response.ok) report = await response.json();
  } catch (error) {
    report = null;
  }
  if (!report && window.__SAMPLE_REPORT__) {
    report = window.__SAMPLE_REPORT__;
    staticMode = true;
  }
  if (!report) {
    showToast("样例报告未生成，请先运行 python3 -m assettone demo", true);
    setBusy(false);
    return;
  }
  if (state.objectUrl) {
    URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = "";
  }
  const audioUrl = new URL("../samples/museum_sample.wav", window.location.href).href;
  renderReport(report, audioUrl, false);
  showToast(staticMode ? "静态样例报告已载入" : "已载入两编号馆藏样例");
  setBusy(false);
}

async function fileToBase64(file) {
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunk = 0x8000;
  for (let index = 0; index < bytes.length; index += chunk) {
    binary += String.fromCharCode(...bytes.subarray(index, index + chunk));
  }
  return btoa(binary);
}

async function analyzeUpload() {
  if (!state.audioFile) return;
  setBusy(true);
  try {
    const health = await fetch("/api/health", { cache: "no-store" }).catch(() => null);
    if (!health || !health.ok) {
      throw new Error("上传分析需启动本地服务：python3 -m assettone serve");
    }
    let manifest = null;
    if (state.manifestFile) {
      try {
        manifest = JSON.parse(await state.manifestFile.text());
      } catch {
        throw new Error("真值清单不是有效 JSON");
      }
    }
    const payload = {
      filename: state.audioFile.name,
      audio_base64: await fileToBase64(state.audioFile),
      manifest,
    };
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const report = await response.json();
    if (!response.ok) throw new Error(report.error || "分析失败");
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = URL.createObjectURL(state.audioFile);
    renderReport(report, state.objectUrl, true);
    showToast(
      `分析完成：${report.summary.hits} 次命中，${report.summary.candidates} 条候选`,
    );
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setBusy(false);
  }
}

function acceptAudioFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".wav")) {
    showToast("请选择 WAV 录音", true);
    return;
  }
  state.audioFile = file;
  const zone = $("#dropzone");
  zone.classList.add("has-file");
  zone.querySelector("strong").textContent = file.name;
  zone.querySelector("small").textContent =
    `${(file.size / 1024 / 1024).toFixed(2)} MB · 本地待处理`;
  $("#analyze-button").disabled = false;
}

function bindEvents() {
  $("#sample-button").addEventListener("click", loadSample);
  $("#reload-sample").addEventListener("click", loadSample);
  $("#analyze-button").addEventListener("click", analyzeUpload);

  $("#audio-file").addEventListener("change", (event) => {
    acceptAudioFile(event.target.files[0]);
  });
  $("#manifest-file").addEventListener("change", (event) => {
    state.manifestFile = event.target.files[0] || null;
    $("#manifest-label").textContent = state.manifestFile
      ? state.manifestFile.name
      : "未选择";
  });

  const zone = $("#dropzone");
  ["dragenter", "dragover"].forEach((name) => {
    zone.addEventListener(name, (event) => {
      event.preventDefault();
      zone.classList.add("dragging");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    zone.addEventListener(name, (event) => {
      event.preventDefault();
      zone.classList.remove("dragging");
    });
  });
  zone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files[0];
    acceptAudioFile(file);
  });

  $("#play-button").addEventListener("click", async () => {
    if (!audio.src) return;
    if (audio.paused) {
      try {
        await audio.play();
      } catch {
        showToast("浏览器无法播放该音频", true);
      }
    } else {
      audio.pause();
    }
  });
  audio.addEventListener("play", () => {
    $("#play-button").textContent = "Ⅱ";
    cancelAnimationFrame(state.animation);
    animateTimeline();
  });
  audio.addEventListener("pause", () => {
    $("#play-button").textContent = "▶";
    cancelAnimationFrame(state.animation);
    drawTimeline();
  });
  audio.addEventListener("ended", () => {
    $("#play-button").textContent = "▶";
    drawTimeline();
  });
  audio.addEventListener("timeupdate", () => {
    $("#audio-time").textContent = formatTime(audio.currentTime);
    const duration = state.report?.source.duration_seconds || audio.duration || 1;
    $("#scrub-fill").style.width = `${Math.min(100, (audio.currentTime / duration) * 100)}%`;
    drawTimeline();
  });

  $(".scrub").addEventListener("click", (event) => {
    if (!audio.duration) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    audio.currentTime = ((event.clientX - bounds.left) / bounds.width) * audio.duration;
  });

  $("#timeline").addEventListener("click", (event) => {
    if (!state.report) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width));
    const target = ratio * (state.report.source.duration_seconds || audio.duration || 0);
    const marker = state.report.detections.reduce((nearest, item, index) => {
      const distance = Math.abs(item.marker_seconds - target);
      return distance < nearest.distance ? { distance, item, index } : nearest;
    }, { distance: Infinity, item: null, index: -1 });
    if (marker.item && marker.distance < 0.18) {
      selectDetection(marker.index);
      audio.currentTime = marker.item.sync_seconds;
    } else {
      audio.currentTime = target;
    }
    drawTimeline();
  });

  new ResizeObserver(() => drawTimeline()).observe($("#timeline"));
}

bindEvents();
loadSample();
