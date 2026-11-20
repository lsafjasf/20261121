from __future__ import annotations

import json
from pathlib import Path

from .calibration import CalibrationResult
from .config import LabConfig


def _css() -> str:
    return """
:root {
  --bg: #071018;
  --panel: #0d1823;
  --panel-2: #111f2c;
  --line: #23384a;
  --text: #e6eef5;
  --muted: #8aa0b2;
  --cyan: #5de1e6;
  --lime: #a6e884;
  --amber: #f5b14a;
  --red: #f4525e;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0; color: var(--text); background: var(--bg);
  font-family: "Inter", "Noto Sans SC", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
  line-height: 1.55;
}
body::before {
  content: ""; position: fixed; inset: 0; pointer-events: none; opacity: .16;
  background-image: linear-gradient(rgba(255,255,255,.04) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,.04) 1px, transparent 1px);
  background-size: 36px 36px;
}
main { width: min(1240px, calc(100% - 40px)); margin: 0 auto; padding-bottom: 80px; }
header { padding: 58px 0 30px; position: relative; }
.eyebrow {
  color: var(--cyan); font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px; letter-spacing: .18em; text-transform: uppercase;
}
h1 { max-width: 900px; margin: 12px 0 14px; font-size: clamp(2rem, 5vw, 4.4rem); line-height: 1.03; letter-spacing: -.045em; }
h1 span { color: var(--cyan); }
.lede { max-width: 780px; color: var(--muted); font-size: 17px; }
.badge { display: inline-flex; gap: 8px; align-items: center; padding: 6px 10px; border: 1px solid var(--line); border-radius: 999px; color: var(--muted); font-size: 12px; }
.badge i { width: 7px; height: 7px; border-radius: 50%; background: var(--lime); box-shadow: 0 0 14px var(--lime); }
.summary { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 28px 0 42px; }
.metric { padding: 18px 18px 15px; border: 1px solid var(--line); border-radius: 14px; background: linear-gradient(145deg, rgba(19,34,48,.92), rgba(10,19,28,.92)); }
.metric .label { color: var(--muted); font-size: 12px; letter-spacing: .04em; }
.metric .value { margin-top: 7px; font-size: 29px; font-weight: 720; letter-spacing: -.035em; }
.metric .sub { margin-top: 3px; color: var(--muted); font-size: 12px; }
.accent { color: var(--cyan); }
.accent-lime { color: var(--lime); }
.accent-amber { color: var(--amber); }
section { margin-top: 52px; }
.section-head { display: flex; justify-content: space-between; gap: 18px; align-items: end; margin-bottom: 16px; }
h2 { margin: 0; font-size: 25px; letter-spacing: -.025em; }
.section-head p { margin: 0; max-width: 600px; color: var(--muted); font-size: 14px; }
.video-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.video-card, .chart-card, .diag-card { border: 1px solid var(--line); border-radius: 15px; background: var(--panel); overflow: hidden; }
.video-title { display: flex; justify-content: space-between; padding: 11px 14px; font-size: 12px; font-family: ui-monospace, monospace; }
.video-title .tag { color: var(--amber); }
.video-title .tag.cyan { color: var(--cyan); }
.video-stage { background: #020508; }
.video-stage video { display: block; width: 100%; aspect-ratio: 4 / 3; }
.zoom-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 14px; }
.zoom-card { border: 1px solid var(--line); border-radius: 15px; overflow: hidden; background: var(--panel); }
.zoom-label { padding: 9px 13px; font-family: ui-monospace, monospace; font-size: 11px; color: var(--muted); }
.zoom-pane { position: relative; aspect-ratio: 4 / 3; overflow: hidden; background: #020508; }
.zoom-pane video { position: absolute; inset: 0; width: 100%; height: 100%; transform-origin: 0 0; image-rendering: pixelated; }
.controls { display: flex; align-items: center; gap: 12px; margin-top: 14px; padding: 12px 14px; border: 1px solid var(--line); border-radius: 12px; background: var(--panel); }
button { cursor: pointer; border: 1px solid #31536a; color: var(--text); background: #122535; border-radius: 8px; padding: 8px 14px; font: inherit; }
button:hover { border-color: var(--cyan); color: var(--cyan); }
input[type=range] { flex: 1; accent-color: var(--cyan); }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; color: var(--muted); white-space: nowrap; }
select { background: #122535; color: var(--text); border: 1px solid #31536a; border-radius: 8px; padding: 7px; }
.chart-grid { display: grid; grid-template-columns: 1fr; gap: 14px; }
.chart-card img, .diag-card img { display: block; width: 100%; height: auto; }
.chart-caption { padding: 12px 15px; color: var(--muted); font-size: 13px; border-top: 1px solid var(--line); }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.info-card { padding: 18px; border: 1px solid var(--line); border-radius: 14px; background: var(--panel); }
.info-card h3 { margin: 0 0 10px; font-size: 15px; }
.info-card p { margin: 0; color: var(--muted); font-size: 13px; }
.method { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.step { padding: 18px; border-left: 2px solid var(--cyan); background: linear-gradient(90deg, rgba(93,225,230,.08), transparent); }
.step b { display: block; margin-bottom: 7px; }
.step p { margin: 0; color: var(--muted); font-size: 13px; }
.step:nth-child(2) { border-color: var(--amber); background: linear-gradient(90deg, rgba(245,177,74,.07), transparent); }
.step:nth-child(3) { border-color: var(--lime); background: linear-gradient(90deg, rgba(166,232,132,.07), transparent); }
.table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 14px; }
table { width: 100%; border-collapse: collapse; font-size: 12px; font-family: ui-monospace, monospace; }
th, td { padding: 10px 12px; text-align: right; border-bottom: 1px solid #1a2b3a; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
th { color: var(--muted); font-weight: 500; background: #0c1722; }
tbody tr:last-child td { border-bottom: 0; }
details { margin-top: 13px; border: 1px solid var(--line); border-radius: 12px; background: var(--panel); }
summary { cursor: pointer; padding: 12px 14px; color: var(--muted); }
.downloads { display: flex; flex-wrap: wrap; gap: 9px; }
.downloads a { color: var(--text); text-decoration: none; padding: 8px 11px; border: 1px solid var(--line); border-radius: 8px; font-family: ui-monospace, monospace; font-size: 12px; }
.downloads a:hover { color: var(--cyan); border-color: var(--cyan); }
footer { margin-top: 58px; padding-top: 18px; border-top: 1px solid var(--line); color: var(--muted); font-size: 12px; }
@media (max-width: 900px) {
  .summary { grid-template-columns: 1fr 1fr; }
  .video-grid, .zoom-grid, .two-col, .method { grid-template-columns: 1fr; }
  .section-head { display: block; }
  .section-head p { margin-top: 8px; }
}
"""


def _script() -> str:
    return """
const master = document.getElementById("original-video");
const follower = document.getElementById("restored-video");
const zoomOriginal = document.getElementById("zoom-original");
const zoomRestored = document.getElementById("zoom-restored");
const videos = [master, follower, zoomOriginal, zoomRestored];
const seek = document.getElementById("seek");
const playButton = document.getElementById("play");
const timeLabel = document.getElementById("time");
const speed = document.getElementById("speed");
const track = ZOOM_TRACK;
let dragging = false;

function frameIndex(time) {
  return Math.max(0, Math.min(track.centers.length - 1, Math.floor(time * track.fps)));
}

function updateZoom(time) {
  const center = track.centers[frameIndex(time)];
  const panes = document.querySelectorAll(".zoom-pane");
  for (const pane of panes) {
    const video = pane.querySelector("video");
    const width = pane.clientWidth;
    const height = pane.clientHeight;
    const scale = 3.0;
    const x = width / 2 - (center[0] / 480) * width * scale;
    const y = height / 2 - (center[1] / 360) * height * scale;
    video.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
  }
}

function syncTo(time) {
  for (const video of videos) {
    if (video !== master && Math.abs(video.currentTime - time) > 0.045) {
      video.currentTime = time;
    }
  }
  updateZoom(time);
}

function updateTimeLabel() {
  const duration = Number.isFinite(master.duration) ? master.duration : 0;
  timeLabel.textContent = `${master.currentTime.toFixed(2)} / ${duration.toFixed(2)} s`;
  if (!dragging) seek.value = master.currentTime;
}

function playAll() {
  const atEnd = master.ended || master.currentTime >= master.duration - 0.01;
  const time = atEnd ? 0 : master.currentTime;
  for (const video of videos) video.currentTime = time;
  Promise.all(videos.map((video) => video.play())).then(() => {
    playButton.textContent = "Pause";
  }).catch(() => {});
}

function pauseAll() {
  for (const video of videos) video.pause();
  playButton.textContent = "Play";
}

playButton.addEventListener("click", () => {
  if (videos.some((video) => !video.paused)) pauseAll(); else playAll();
});

master.addEventListener("loadedmetadata", () => {
  seek.max = master.duration;
  for (const video of videos) video.currentTime = 0;
  updateTimeLabel();
  updateZoom(0);
});

master.addEventListener("timeupdate", () => {
  if (!master.paused) syncTo(master.currentTime);
  updateTimeLabel();
});

master.addEventListener("ended", () => {
  pauseAll();
  playButton.textContent = "Replay";
});

seek.addEventListener("input", () => {
  dragging = true;
  const time = Number(seek.value);
  for (const video of videos) video.currentTime = time;
  updateZoom(time);
  updateTimeLabel();
});
seek.addEventListener("change", () => { dragging = false; });

speed.addEventListener("change", () => {
  for (const video of videos) video.playbackRate = Number(speed.value);
});

for (const video of videos) {
  video.addEventListener("click", () => {
    if (video.paused) playAll(); else pauseAll();
  });
}

document.addEventListener("keydown", (event) => {
  if (event.code === "Space") {
    event.preventDefault();
    if (videos.some((video) => !video.paused)) pauseAll(); else playAll();
  } else if (event.code === "ArrowRight") {
    master.currentTime = Math.min(master.duration, master.currentTime + 0.25);
    syncTo(master.currentTime);
  } else if (event.code === "ArrowLeft") {
    master.currentTime = Math.max(0, master.currentTime - 0.25);
    syncTo(master.currentTime);
  }
});

window.addEventListener("resize", () => updateZoom(master.currentTime));
"""


def _channel_rows(calibration: CalibrationResult, config: LabConfig) -> str:
    rows: list[str] = []
    for index, channel in enumerate(config.channels):
        alpha_rise = calibration.params.alpha_rise[index]
        alpha_fall = calibration.params.alpha_fall[index]
        tau_rise = calibration.params.tau_rise[index]
        tau_fall = calibration.params.tau_fall[index]
        rows.append(
            "<tr>"
            f"<td>{channel}</td>"
            f"<td>{alpha_rise:.4f}</td>"
            f"<td>{alpha_fall:.4f}</td>"
            f"<td>{tau_rise:.2f}</td>"
            f"<td>{tau_fall:.2f}</td>"
            f"<td>{tau_rise / config.fps * 1000:.0f}</td>"
            f"<td>{tau_fall / config.fps * 1000:.0f}</td>"
            f"<td>{1 / alpha_rise:.2f}x</td>"
            "</tr>"
        )
    return "".join(rows)


def _direction_rows(calibration: CalibrationResult) -> str:
    rows: list[str] = []
    for fit in calibration.fits:
        if fit.scope != "directional":
            continue
        rows.append(
            "<tr>"
            f"<td>{fit.direction}</td>"
            f"<td>{fit.channel}</td>"
            f"<td>{fit.transition}</td>"
            f"<td>{fit.tau_frames:.2f}</td>"
            f"<td>{fit.alpha:.4f}</td>"
            f"<td>{fit.samples:,}</td>"
            f"<td>{fit.fit_rmse:.5f}</td>"
            "</tr>"
        )
    return "".join(rows)


def _load_zoom_centers(out_dir: Path) -> list[list[float]]:
    path = out_dir / "zoom_track.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("centers", [])


def write_report(
    calibration: CalibrationResult,
    metrics: dict,
    config: LabConfig,
    out_dir: Path,
) -> None:
    track_json = json.dumps(
        {
            "fps": config.fps,
            "crop_size": [174, 130],
            "centers": [
                [round(float(x), 2), round(float(y), 2)]
                for x, y in _load_zoom_centers(out_dir)
            ],
        },
        separators=(",", ":"),
    )
    ghost = metrics["ghost"]
    overshoot = metrics["overshoot"]
    overall = metrics["overall"]
    diag = metrics["diagnostics"]
    duration = metrics["video"]["duration_seconds"]
    unstable_pct = diag["unstable_pixel_fraction"] * 100.0
    saturated_pct = diag["saturated_pixel_fraction"] * 100.0
    channel_rows = _channel_rows(calibration, config)
    direction_rows = _direction_rows(calibration)

    head = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LCD 液晶拖影校准与视频复原实验</title>
<style>{_css()}</style>
</head>
<body>
<main>
<header>
  <div class="eyebrow">Liquid-crystal response / calibration + inversion</div>
  <h1>旧掌机屏幕的<br><span>拖影复原实验</span></h1>
  <p class="lede">用黑白阶跃片估计 R/G/B 三个通道各自的上升与下降时间常数，再把同屏演示片逐帧反演。对比视图、方向校准数据、残影与过冲指标都在同一页。</p>
  <div class="badge"><i></i> 合成基准 · {duration:.1f}s · {metrics['video']['width']}×{metrics['video']['height']} · {config.fps} fps</div>
</header>

<div class="summary">
  <div class="metric">
    <div class="label">运动轮廓残影 · RMS</div>
    <div class="value accent-lime">{ghost['restored']['rms_8bit']:.1f}</div>
    <div class="sub">原片 {ghost['captured']['rms_8bit']:.1f} / 255 · 改善 {ghost['improvement_db']:.1f} dB</div>
  </div>
  <div class="metric">
    <div class="label">棋盘过冲 · RMS</div>
    <div class="value accent">{overshoot['restored']['rms_excursion_8bit']:.2f}</div>
    <div class="sub">原片 {overshoot['captured']['rms_excursion_8bit']:.1f} / 255 · 改善 {overshoot['improvement_db']:.1f} dB</div>
  </div>
  <div class="metric">
    <div class="label">整帧误差 · RMS</div>
    <div class="value">{overall['restored']['rms_8bit']:.2f}</div>
    <div class="sub">原片 {overall['captured']['rms_8bit']:.2f} / 255 · {overall['improvement_db']:.1f} dB</div>
  </div>
  <div class="metric">
    <div class="label">诊断标记面积</div>
    <div class="value accent-amber">{unstable_pct:.2f}%</div>
    <div class="sub">不稳定反推 · 饱和/越界 {saturated_pct:.2f}%</div>
  </div>
</div>

<section>
  <div class="section-head">
    <h2>方法</h2>
    <p>模型是非对称切换一阶系统：误差为正时用上升系数，为负时用下降系数；反演同时保留局部增益与越界状态。</p>
  </div>
  <div class="method">
    <div class="step"><b>01 / 校准</b><p>全屏黑白阶跃覆盖 rise/fall，四个方向的亮带扫掠产生方向校准数据。每个通道分别做最小二乘拟合。</p></div>
    <div class="step"><b>02 / 反演</b><p>由相邻观测帧的符号选择上升或下降分支，反算源像素。反演增益 1/α 超过 {config.instability_gain:.1f} 时记为不稳定。</p></div>
    <div class="step"><b>03 / 验证</b><p>在运动轮廓重叠带测残影能量；在静止棋盘翻转后测越过目标平台的过冲；越界估计单独着色。</p></div>
  </div>
</section>
"""

    video_section = f"""
<section>
  <div class="section-head">
    <h2>原片 / 复原片 / 同步放大</h2>
    <p>四个画面共用时间轴。下方 3× 放大窗口会自动跟随运动圆形轮廓；空格播放/暂停，左右方向键微调。</p>
  </div>
  <div class="video-grid">
    <div class="video-card">
      <div class="video-title"><span>ORIGINAL RECORDING</span><span class="tag">拖影原片</span></div>
      <div class="video-stage"><video id="original-video" src="captured.mp4" muted playsinline preload="metadata"></video></div>
    </div>
    <div class="video-card">
      <div class="video-title"><span>RESTORED</span><span class="tag cyan">复原片</span></div>
      <div class="video-stage"><video id="restored-video" src="restored.mp4" muted playsinline preload="metadata"></video></div>
    </div>
  </div>
  <div class="zoom-grid">
    <div class="zoom-card">
      <div class="zoom-label">SYNCHRONIZED ZOOM 3× / ORIGINAL</div>
      <div class="zoom-pane"><video id="zoom-original" src="captured.mp4" muted playsinline preload="metadata"></video></div>
    </div>
    <div class="zoom-card">
      <div class="zoom-label">SYNCHRONIZED ZOOM 3× / RESTORED</div>
      <div class="zoom-pane"><video id="zoom-restored" src="restored.mp4" muted playsinline preload="metadata"></video></div>
    </div>
  </div>
  <div class="controls">
    <button id="play" type="button">Play</button>
    <input id="seek" type="range" min="0" max="{duration:.3f}" step="0.01" value="0" aria-label="播放进度">
    <span id="time" class="mono">0.00 / {duration:.2f} s</span>
    <select id="speed" aria-label="播放速度">
      <option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option>
    </select>
  </div>
</section>
"""

    calibration_section = f"""
<section>
  <div class="section-head">
    <h2>逐通道阶跃校准</h2>
    <p>全局拟合使用全部 190 帧校准数据；方向表来自左→右、右→左、上→下、下→上四段扫掠，时间单位为视频帧。</p>
  </div>
  <div class="chart-grid">
    <div class="chart-card">
      <img src="calibration_response.png" alt="逐通道黑白阶跃响应与拟合">
      <div class="chart-caption">实线为校准片实测响应，虚线为切换一阶模型。蓝通道上升最慢，因此需要最大的反演增益。</div>
    </div>
    <div class="chart-card">
      <img src="calibration_directions.png" alt="不同扫掠方向的响应时间热图">
      <div class="chart-caption">四方向扫掠的 rise/fall 时间常数。合成数据加入了轻微方向偏差，复原只采用全局估计，方向数据用于量化一致性。</div>
    </div>
  </div>
  <div class="table-wrap" style="margin-top:14px">
    <table>
      <thead><tr><th>通道</th><th>α rise</th><th>α fall</th><th>τ rise (帧)</th><th>τ fall (帧)</th><th>τ rise (ms)</th><th>τ fall (ms)</th><th>反演增益</th></tr></thead>
      <tbody>{channel_rows}</tbody>
    </table>
  </div>
  <details>
    <summary>展开四方向 × R/G/B × rise/fall 的 24 组校准记录</summary>
    <div class="table-wrap" style="border:0;border-top:1px solid var(--line);border-radius:0">
      <table>
        <thead><tr><th>方向</th><th>通道</th><th>过渡</th><th>τ (帧)</th><th>α</th><th>样本数</th><th>拟合 RMSE</th></tr></thead>
        <tbody>{direction_rows}</tbody>
      </table>
    </div>
  </details>
</section>
"""

    metrics_section = f"""
<section>
  <div class="section-head">
    <h2>残影与过冲测量</h2>
    <p>运动带用连续两帧膨胀轮廓的交集定义；棋盘指标只在第 72 帧相位翻转后的 18 帧内计算，并排除运动物体遮挡。</p>
  </div>
  <div class="chart-grid">
    <div class="chart-card">
      <img src="metrics.png" alt="残影能量和过冲指标柱状图">
      <div class="chart-caption">残影按逐通道 RMS 比较；过冲为越过黑白目标平台 3.5%（约 9/255）后的错误方向偏移。</div>
    </div>
  </div>
  <div class="two-col" style="margin-top:14px">
    <div class="info-card">
      <h3>重叠运动轮廓 / Ghost energy</h3>
      <p>原片 {ghost['captured']['rms_8bit']:.2f} → 复原 {ghost['restored']['rms_8bit']:.2f} RMS/255。逐通道改善：R {ghost['captured']['per_channel_rms_8bit'][0]:.1f}→{ghost['restored']['per_channel_rms_8bit'][0]:.1f}，G {ghost['captured']['per_channel_rms_8bit'][1]:.1f}→{ghost['restored']['per_channel_rms_8bit'][1]:.1f}，B {ghost['captured']['per_channel_rms_8bit'][2]:.1f}→{ghost['restored']['per_channel_rms_8bit'][2]:.1f}。</p>
    </div>
    <div class="info-card">
      <h3>静止棋盘 / Overshoot</h3>
      <p>原片 RMS 越界 {overshoot['captured']['rms_excursion_8bit']:.2f} → 复原 {overshoot['restored']['rms_excursion_8bit']:.2f}。受影响像素比例 {overshoot['captured']['affected_fraction']*100:.1f}% → {overshoot['restored']['affected_fraction']*100:.1f}%，峰值为 {overshoot['captured']['peak_excursion_8bit']:.1f} → {overshoot['restored']['peak_excursion_8bit']:.1f}/255。</p>
    </div>
  </div>
</section>
"""

    diagnostics_section = f"""
<section>
  <div class="section-head">
    <h2>诊断图</h2>
    <p>红色表示该处反演局部增益超过 {config.instability_gain:.1f}，橙色表示反算出的源像素越过 [0,1] 后被截断。两者都保留在逐帧掩膜中，而不是静默修复。</p>
  </div>
  <div class="diag-card">
    <img src="diagnostics.png" alt="不稳定与饱和反演位置诊断图">
    <div class="chart-caption">左上为第 78 帧叠加；右上、左下为整段视频的累计频率图；右下为逐帧标记数量。蓝色虚线是棋盘相位翻转。</div>
  </div>
  <div class="two-col" style="margin-top:14px">
    <div class="info-card">
      <h3>不稳定反推</h3>
      <p>占全图像素时间样本 {unstable_pct:.2f}%。主要出现在蓝通道慢速边沿：1/α ≈ 7.3×，会把采集噪声同步放大。</p>
    </div>
    <div class="info-card">
      <h3>饱和 / 越界</h3>
      <p>占 {saturated_pct:.2f}%。这些位置无法在不越过物理亮度范围的前提下精确反演，输出采用截断值，同时保留橙色标记。</p>
    </div>
  </div>
</section>
"""

    artifacts_section = f"""
<section>
  <div class="section-head">
    <h2>结果文件</h2>
    <p>报告为纯静态页面，视频与图片均使用相对路径，可以直接打开或通过本地 HTTP 服务查看。</p>
  </div>
  <div class="two-col">
    <div class="info-card">
      <h3>影像</h3>
      <div class="downloads" style="margin-top:12px">
        <a href="captured.mp4">原片 · captured.mp4</a>
        <a href="restored.mp4">复原片 · restored.mp4</a>
        <a href="comparison.mp4">同步放大 · comparison.mp4</a>
        <a href="ground_truth.mp4">参考真值 · ground_truth.mp4</a>
        <a href="calibration_capture.mp4">校准录制 · calibration_capture.mp4</a>
      </div>
    </div>
    <div class="info-card">
      <h3>数据与图表</h3>
      <div class="downloads" style="margin-top:12px">
        <a href="metrics.json">metrics.json</a>
        <a href="calibration.json">calibration.json</a>
        <a href="calibration_fits.csv">calibration_fits.csv</a>
        <a href="comparison_stills.png">comparison_stills.png</a>
        <a href="diagnostics.png">diagnostics.png</a>
      </div>
    </div>
  </div>
</section>

<footer>
  Synthetic benchmark: directional panel bias is generated only for calibration validation; restoration uses the global per-channel fit.
  The inversion is deterministic and does not use the reference frames.
</footer>
</main>
<script>const ZOOM_TRACK = {track_json};</script>
<script>{_script()}</script>
<script>updateZoom(0); updateTimeLabel();</script>
</body>
</html>
"""

    (out_dir / "report.html").write_text(
        head + video_section + calibration_section + metrics_section
        + diagnostics_section + artifacts_section,
        encoding="utf-8",
    )
