"""Static HTML report and aligned A/B player generation."""

from __future__ import annotations

import html
import math
import re
from pathlib import Path

from .dsp import rms_envelope_db
from .metrics import MetricSettings, OnsetMetrics
from .runner import CaseResult, CodecResult, LabResult

CLASS_LABELS = {
    "codec-added": "编码新增",
    "source-prelude": "源前奏占优",
    "mixed": "混合",
    "no-added-energy": "无可测新增",
}

CSS = """
:root {
  --bg: #f2f0ea;
  --paper: #fffdf8;
  --ink: #17191e;
  --muted: #6c727d;
  --line: #ded9ce;
  --accent: #cf4c2b;
  --accent-soft: #f9e2d9;
  --teal: #0d756f;
  --teal-soft: #dcefeb;
  --pass: #16734b;
  --pass-soft: #dff2e7;
  --warn: #a63d22;
  --warn-soft: #f9e0d7;
  --shadow: 0 18px 50px rgba(30, 27, 20, .08);
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: "Noto Sans SC", "PingFang SC", "Microsoft YaHei",
    ui-sans-serif, system-ui, sans-serif;
  line-height: 1.55;
}
a { color: var(--teal); }
.wrap { width: min(1180px, calc(100% - 36px)); margin: 0 auto; }
.hero {
  color: #f8f5ed; padding: 54px 0 30px;
  background:
    radial-gradient(circle at 82% 10%, rgba(207,76,43,.35), transparent 31%),
    radial-gradient(circle at 18% 92%, rgba(13,117,111,.3), transparent 34%),
    #17191e;
  border-bottom: 5px solid var(--accent);
}
.eyebrow {
  color: #d9a997; font: 700 11px/1.2 ui-monospace, SFMono-Regular, monospace;
  letter-spacing: .18em; text-transform: uppercase;
}
h1 { margin: 12px 0 8px; font-size: clamp(30px, 5vw, 54px); line-height: 1.08; }
.hero p { max-width: 760px; margin: 0; color: #c9c6c0; font-size: 16px; }
.hero-grid { display: grid; grid-template-columns: 1.5fr 1fr; gap: 28px; align-items: end; }
.hero-stats { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
.hero-stat {
  padding: 14px 16px; border: 1px solid rgba(255,255,255,.14);
  background: rgba(255,255,255,.055); border-radius: 13px;
}
.hero-stat span { display: block; color: #aaa9a5; font-size: 11px; letter-spacing: .08em; }
.hero-stat strong { display: block; margin-top: 5px; font-size: 23px; }
main { padding: 28px 0 64px; }
.status {
  display: flex; align-items: center; justify-content: space-between; gap: 16px;
  padding: 16px 20px; margin: -4px 0 24px; border: 1px solid var(--line);
  border-radius: 16px; background: var(--paper); box-shadow: var(--shadow);
}
.status.pass { border-left: 6px solid var(--pass); }
.status.fail { border-left: 6px solid var(--warn); }
.status h2 { margin: 0; font-size: 18px; }
.status p { margin: 3px 0 0; color: var(--muted); font-size: 13px; }
.pill {
  display: inline-flex; align-items: center; gap: 6px; white-space: nowrap;
  padding: 5px 10px; border-radius: 999px; font-size: 12px; font-weight: 700;
}
.pill.pass { color: var(--pass); background: var(--pass-soft); }
.pill.fail { color: var(--warn); background: var(--warn-soft); }
.pill.codec-added { color: #8d331c; background: var(--accent-soft); }
.pill.source-prelude { color: var(--teal); background: var(--teal-soft); }
.pill.mixed { color: #785a00; background: #f4eac1; }
.pill.no-added-energy { color: #5e6470; background: #e9e8e3; }
.method-grid, .codec-strip { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
.mini, .codec-chip {
  padding: 17px 18px; background: var(--paper); border: 1px solid var(--line);
  border-radius: 14px;
}
.mini span, .codec-chip span { color: var(--muted); font-size: 11px; letter-spacing: .08em; }
.mini h3, .codec-chip h3 { margin: 5px 0; font-size: 16px; }
.mini p, .codec-chip p { margin: 0; color: var(--muted); font-size: 12px; }
.codec-strip { margin: 14px 0 30px; }
.codec-chip { border-top: 4px solid var(--teal); }
.case { margin-top: 28px; }
.case-head {
  display: flex; justify-content: space-between; gap: 24px; align-items: end;
  padding: 28px; border: 1px solid var(--line); border-radius: 18px;
  background: var(--paper); box-shadow: var(--shadow);
}
.case-head h2 { margin: 8px 0 6px; font-size: clamp(24px, 3vw, 34px); }
.case-head p { margin: 0; color: var(--muted); max-width: 720px; }
.case-kpis { display: flex; gap: 24px; flex-wrap: wrap; margin-top: 16px; }
.kpi span { display: block; color: var(--muted); font-size: 11px; }
.kpi strong { font-size: 18px; font-variant-numeric: tabular-nums; }
.codec-grid { display: grid; grid-template-columns: 1fr; gap: 16px; margin-top: 16px; }
.codec-card {
  background: var(--paper); border: 1px solid var(--line); border-radius: 18px;
  overflow: hidden; box-shadow: 0 10px 30px rgba(30,27,20,.05);
}
.codec-card-head {
  display: flex; justify-content: space-between; align-items: center; gap: 12px;
  padding: 17px 20px; border-bottom: 1px solid var(--line);
}
.codec-card-head h3 { margin: 0; font-size: 18px; }
.codec-card-head small { color: var(--muted); font-weight: 400; }
.codec-body { padding: 18px 20px 20px; }
.metrics {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 0 0 16px;
}
.ab-player {
  padding: 12px; margin-bottom: 15px; border: 1px solid var(--line);
  border-radius: 12px; background: #f6f3ec;
}
.ab-controls { display: flex; align-items: center; gap: 9px; }
.ab-controls button {
  border: 1px solid #cfc9bd; background: #fffdf8; color: var(--ink);
  border-radius: 9px; padding: 8px 12px; cursor: pointer; font: inherit;
  font-size: 12px; font-weight: 700;
}
.ab-controls button:hover { border-color: var(--teal); }
.ab-controls .ab-play { min-width: 78px; background: var(--ink); color: #fff; border-color: var(--ink); }
.ab-sides { display: flex; padding: 2px; border-radius: 9px; background: #e8e4da; }
.ab-sides button { border: 0; background: transparent; padding: 6px 10px; }
.ab-sides button.active { background: var(--paper); box-shadow: 0 1px 4px rgba(0,0,0,.12); }
.ab-sides button[data-side="B"].active { color: var(--accent); }
.ab-progress { flex: 1; min-width: 70px; accent-color: var(--teal); }
.ab-time { min-width: 74px; text-align: right; color: var(--muted); font: 11px ui-monospace, monospace; }
.ab-caption { margin-top: 7px; color: var(--muted); font-size: 11px; }
.metric { padding: 11px 13px; border: 1px solid var(--line); border-radius: 11px; }
.metric dt { color: var(--muted); font-size: 10px; letter-spacing: .06em; }
.metric dd { margin: 4px 0 0; font-size: 17px; font-weight: 700; font-variant-numeric: tabular-nums; }
table { width: 100%; border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }
th { color: var(--muted); font-size: 10px; letter-spacing: .05em; text-align: right; }
th:first-child, th:nth-child(2), td:first-child, td:nth-child(2) { text-align: left; }
th, td { padding: 8px 7px; border-bottom: 1px solid #ece8df; }
tr.worst td { background: #fff5f0; }
td.num { text-align: right; }
.links { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 15px; font-size: 12px; }
.links a { text-decoration: none; font-weight: 600; }
details { margin-top: 16px; border-top: 1px dashed var(--line); padding-top: 13px; }
summary { cursor: pointer; color: var(--teal); font-weight: 700; font-size: 13px; }
.wave-note { color: var(--muted); font-size: 12px; margin: 10px 0 4px; }
.wave-svg { width: 100%; height: auto; display: block; border-radius: 10px; background: #faf8f2; }
.method { margin-top: 34px; padding: 28px; background: var(--paper); border: 1px solid var(--line); border-radius: 18px; }
.method h2 { margin-top: 0; }
.method code { background: #efede6; padding: 2px 5px; border-radius: 5px; }
.foot { margin-top: 22px; color: var(--muted); font-size: 11px; }
@media (max-width: 800px) {
  .hero-grid, .method-grid { grid-template-columns: 1fr; }
  .codec-strip { grid-template-columns: 1fr; }
  .case-head { align-items: start; flex-direction: column; }
  .metrics { grid-template-columns: repeat(2, 1fr); }
  .table-wrap { overflow-x: auto; }
}
"""


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _slug(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", str(value)).strip("-") or "item"


def _fmt_db(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "—"
    if value <= -179.0:
        return "−∞"
    prefix = "+" if signed and value > 0 else ""
    return f"{prefix}{value:.1f} dB"


def _pill(classification: str) -> str:
    label = CLASS_LABELS.get(classification, classification)
    return f'<span class="pill {_escape(classification)}">{_escape(label)}</span>'


def _acceptance_pill(passed: bool) -> str:
    if passed:
        return '<span class="pill pass">验收通过</span>'
    return '<span class="pill fail">需要复核</span>'


def _expected_label(expected: str) -> str:
    return {
        "codec-added": "预期：编码新增",
        "source-prelude": "预期：源前奏占优",
    }.get(expected, f"预期：{expected}")


def _waveform_svg(
    codec: CodecResult,
    onset: OnsetMetrics,
    settings: MetricSettings,
    unique: str,
) -> str:
    reference = codec.reference
    aligned = codec.aligned
    sample_rate = reference.sample_rate
    pre_span_s = max(0.050, (settings.window_ms + settings.guard_ms) / 1000.0 + 0.020)
    post_span_s = 0.020
    start = max(0, round((onset.onset_s - pre_span_s) * sample_rate))
    stop = min(len(reference.samples), round((onset.onset_s + post_span_s) * sample_rate))
    frame_size = max(1, round(sample_rate * 0.0005))
    if stop <= start + frame_size:
        return '<p class="wave-note">该起音附近没有可绘制的样本。</p>'
    reference_db = rms_envelope_db(
        reference.samples, start, stop, frame_size, floor_db=-90.0
    )
    decoded_db = rms_envelope_db(
        aligned.samples, start, stop, frame_size, floor_db=-90.0
    )
    width, height = 900, 154
    left, right, top, bottom = 45, 14, 16, 29
    plot_width = width - left - right
    plot_height = height - top - bottom

    def x_at(time_s: float) -> float:
        fraction = (time_s - start / sample_rate) / (stop / sample_rate - start / sample_rate)
        return left + max(0.0, min(1.0, fraction)) * plot_width

    def y_at(value_db: float) -> float:
        value_db = max(-90.0, min(0.0, value_db))
        return top + (-value_db / 90.0) * plot_height

    def path_for(values: list[float]) -> tuple[str, str]:
        points = [
            (left + index * plot_width / max(1, len(values) - 1), y_at(value))
            for index, value in enumerate(values)
        ]
        line = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in points)
        area = (
            line
            + f" L {points[-1][0]:.1f} {top + plot_height:.1f}"
            + f" L {points[0][0]:.1f} {top + plot_height:.1f} Z"
        )
        return line, area

    reference_line, reference_area = path_for(reference_db)
    decoded_line, decoded_area = path_for(decoded_db)
    window_start_x = x_at(onset.window_start_s)
    window_stop_x = x_at(onset.window_stop_s)
    onset_x = x_at(onset.onset_s)
    grid = []
    for level_db in (-20, -40, -60, -80):
        y = y_at(level_db)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" '
            'stroke="#ded9ce" stroke-width="1"/>'
        )
        grid.append(
            f'<text x="{left-7}" y="{y+3:.1f}" text-anchor="end" '
            f'fill="#8a8e96" font-size="9">{level_db}</text>'
        )
    ticks = []
    tick_start = math.ceil((start / sample_rate - onset.onset_s) * 10.0)
    tick_stop = math.floor((stop / sample_rate - onset.onset_s) * 10.0)
    for tick in range(tick_start, tick_stop + 1):
        tick_time = onset.onset_s + tick / 10.0
        tick_x = x_at(tick_time)
        ticks.append(
            f'<line x1="{tick_x:.1f}" y1="{top+plot_height}" '
            f'x2="{tick_x:.1f}" y2="{top+plot_height+4}" stroke="#8a8e96"/>'
            f'<text x="{tick_x:.1f}" y="{height-9}" text-anchor="middle" '
            f'fill="#7b7f86" font-size="9">{tick * 10} ms</text>'
        )
    title = (
        f"{codec.config.label} 第 {onset.index} 个起音前 "
        f"{pre_span_s * 1000:.0f} ms 能量包络"
    )
    title_id = f"wave-title-{_slug(unique)}-{_slug(codec.config.key)}-{onset.index}"
    return f"""
<svg class="wave-svg" viewBox="0 0 {width} {height}" role="img"
     aria-labelledby="{title_id}">
  <title id="{title_id}">{_escape(title)}</title>
  <rect x="0" y="0" width="{width}" height="{height}" fill="#faf8f2"/>
  {''.join(grid)}
  <rect x="{window_start_x:.1f}" y="{top}" width="{max(1.0, window_stop_x-window_start_x):.1f}"
        height="{plot_height}" fill="#cf4c2b" opacity=".10"/>
  <rect x="{window_stop_x:.1f}" y="{top}" width="{max(1.0, onset_x-window_stop_x):.1f}"
        height="{plot_height}" fill="#d7a829" opacity=".13"/>
  <line x1="{onset_x:.1f}" y1="{top}" x2="{onset_x:.1f}" y2="{top+plot_height}"
        stroke="#17191e" stroke-width="1.4" stroke-dasharray="3 3"/>
  <text x="{window_start_x+4:.1f}" y="{top+11}" fill="#9b3d26" font-size="9">
    测量窗
  </text>
  <text x="{onset_x+4:.1f}" y="{top+11}" fill="#5c5e64" font-size="9">起音</text>
  <path d="{reference_area}" fill="#6c727d" opacity=".10"/>
  <path d="{decoded_area}" fill="#cf4c2b" opacity=".13"/>
  <path d="{reference_line}" fill="none" stroke="#737985" stroke-width="1.5"/>
  <path d="{decoded_line}" fill="none" stroke="#cf4c2b" stroke-width="1.7"/>
  {''.join(ticks)}
  <text x="{width-right}" y="{top+10}" text-anchor="end" fill="#6c727d" font-size="9">
    灰：参考 · 橙：{_escape(codec.config.key)}
  </text>
</svg>"""


def _min_source_dominance(codec: CodecResult) -> float | None:
    values = [
        onset.source_dominance_db
        for onset in codec.metrics.onsets
        if onset.source_dominance_db is not None
    ]
    return min(values) if values else None


def _player_html(case: CaseResult, codec: CodecResult, unique: str) -> str:
    player_id = f"ab-{_slug(unique)}-{_slug(codec.config.key)}"
    return f"""
<div class="ab-player" data-player="{_escape(player_id)}">
  <audio id="{_escape(player_id)}-a" preload="metadata"
         src="{_escape(case.reference_path)}"></audio>
  <audio id="{_escape(player_id)}-b" preload="metadata"
         src="{_escape(codec.aligned_path)}"></audio>
  <div class="ab-controls">
    <button type="button" class="ab-play" data-action="play"
            aria-label="播放">▶ 播放</button>
    <div class="ab-sides" role="group" aria-label="A/B 声源">
      <button type="button" data-side="A" class="active">A 参考</button>
      <button type="button" data-side="B">B {_escape(codec.config.key.upper())}</button>
    </div>
    <input class="ab-progress" type="range" min="0" max="100" value="0"
           step="0.01" aria-label="播放进度">
    <span class="ab-time">0:00 / 0:00</span>
  </div>
  <div class="ab-caption">
    A 为参考，B 为延迟校准后的 {_escape(codec.config.label)}；
    切换时保持当前时间点，可用同一副耳机直接 A/B。
  </div>
</div>"""


def _onset_rows(codec: CodecResult) -> str:
    rows: list[str] = []
    for onset in codec.metrics.onsets:
        is_worst = onset.index == codec.metrics.worst_added_onset_index
        row_class = ' class="worst"' if is_worst else ""
        rows.append(
            f"""<tr{row_class}>
  <td>{onset.index}</td>
  <td>{onset.onset_s:.3f} s</td>
  <td class="num">{_fmt_db(onset.reference_pre_dbfs)}</td>
  <td class="num">{_fmt_db(onset.decoded_pre_dbfs)}</td>
  <td class="num">{_fmt_db(onset.codec_added_dbfs)}</td>
  <td class="num">{_fmt_db(onset.source_dominance_db, signed=True)}</td>
  <td class="num">{_fmt_db(onset.decoded_pre_to_attack_db)}</td>
  <td>{_pill(onset.classification)}</td>
</tr>"""
        )
    return "\n".join(rows)


def _codec_card(
    case: CaseResult,
    codec: CodecResult,
    unique: str,
    settings: MetricSettings,
) -> str:
    worst_index = codec.metrics.worst_added_onset_index
    selected_onset = next(
        (
            onset
            for onset in codec.metrics.onsets
            if onset.index == worst_index
        ),
        codec.metrics.onsets[0],
    )
    codec_count = len(codec.metrics.onsets)
    acceptance_text = _acceptance_pill(bool(codec.acceptance["passed"]))
    return f"""
<article class="codec-card" id="{_slug(unique)}-{_slug(codec.config.key)}">
  <header class="codec-card-head">
    <div>
      <h3>{_escape(codec.config.label)}
        <small>· {_escape(codec.config.target)}</small>
      </h3>
      <small>{codec_count}/{codec_count} 个起音分类为
        {_escape(CLASS_LABELS.get(case.expected, case.expected))}</small>
    </div>
    {acceptance_text}
  </header>
  <div class="codec-body">
    {_player_html(case, codec, unique)}
    <dl class="metrics">
      <div class="metric"><dt>残余延迟</dt><dd>{codec.delay.delay_ms:+.3f} ms</dd></div>
      <div class="metric"><dt>校准置信度</dt><dd>{codec.delay.confidence * 100:.1f}%</dd></div>
      <div class="metric"><dt>最差编码新增</dt><dd>{_fmt_db(codec.metrics.worst_added_dbfs)}</dd></div>
      <div class="metric"><dt>源高于新增（最小）</dt><dd>{_fmt_db(_min_source_dominance(codec), signed=True)}</dd></div>
    </dl>
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>#</th><th>起音</th><th>参考前窗</th><th>解码前窗</th>
          <th>编码新增</th><th>源 / 新增</th><th>前窗 / 起音</th><th>分类</th>
        </tr></thead>
        <tbody>{_onset_rows(codec)}</tbody>
      </table>
    </div>
    <div class="links">
      <a href="{_escape(codec.ab_path)}" download>下载 A/B WAV</a>
      <a href="{_escape(codec.aligned_path)}" download>对齐解码 WAV</a>
      <a href="{_escape(codec.encoded_path)}" download>
        下载编码文件（{codec.encoded_bytes / 1024:.0f} KiB）</a>
      <a href="{_escape(codec.unaligned_path)}" download>未对齐解码 WAV</a>
    </div>
    <details>
      <summary>查看延迟校准与波形诊断</summary>
      <p class="wave-note">
        延迟估计：{_escape(codec.delay.method)}。下图以第 {selected_onset.index}
        个起音（最差编码新增）为中心，红色区域为 {settings.window_ms:g} ms 测量窗，黄色区域为 {settings.guard_ms:g} ms
        保护带；纵轴为 0.5 ms RMS 包络的 dBFS。
      </p>
      {_waveform_svg(codec, selected_onset, settings, unique)}
    </details>
  </div>
</article>"""


def _delay_range(case: CaseResult) -> str:
    delays = [codec.delay.delay_ms for codec in case.codecs]
    if not delays:
        return "—"
    return f"{min(delays):+.2f} … {max(delays):+.2f} ms"


def _max_added(case: CaseResult) -> float | None:
    values = [
        codec.metrics.worst_added_dbfs
        for codec in case.codecs
        if codec.metrics.worst_added_dbfs is not None
    ]
    return max(values) if values else None


def _reference_pre_db(case: CaseResult) -> float:
    if not case.codecs:
        return -180.0
    return max(
        onset.reference_pre_dbfs for onset in case.codecs[0].metrics.onsets
    )


def _case_section(case: CaseResult, settings: MetricSettings) -> str:
    reference_db = _reference_pre_db(case)
    source_text = (
        "数字静音"
        if reference_db <= -170.0
        else f"约 {_fmt_db(reference_db)}"
    )
    failed = case.acceptance.get("failed_codecs", [])
    warning = ""
    if failed:
        names = "、".join(str(item["codec"]) for item in failed)
        warning = (
            f'<p class="pill fail" style="margin-top:10px">'
            f"需要复核：{_escape(names)} 的分类与预期不符</p>"
        )
    cards = "\n".join(
        _codec_card(case, codec, case.case_id, settings) for codec in case.codecs
    )
    return f"""
<section class="case" id="case-{_escape(case.case_id)}">
  <header class="case-head">
    <div>
      <div>{_pill(case.expected)} <span class="pill">{_escape(_expected_label(case.expected))}</span></div>
      <h2>{_escape(case.title)}</h2>
      <p>{_escape(case.description)}</p>
      <div class="case-kpis">
        <div class="kpi"><span>参考前窗能量</span><strong>{_fmt_db(reference_db)}</strong></div>
        <div class="kpi"><span>最差编码新增</span><strong>{_fmt_db(_max_added(case))}</strong></div>
        <div class="kpi"><span>起音数</span><strong>{len(case.onsets_s)}</strong></div>
        <div class="kpi"><span>延迟范围</span><strong>{_delay_range(case)}</strong></div>
      </div>
      {warning}
    </div>
    <div style="text-align:right">
      {_acceptance_pill(bool(case.acceptance["passed"]))}
      <p style="margin-top:8px">源前窗：{_escape(source_text)}</p>
      <p style="margin-top:3px">{len(case.codecs)} 种编码配置</p>
    </div>
  </header>
  <div class="codec-grid">
    {cards}
  </div>
</section>"""


def _status_banner(result: LabResult) -> str:
    passed = bool(result.acceptance["passed"])
    passed_cases = result.acceptance["passed_cases"]
    total_cases = result.acceptance["total_cases"]
    if passed:
        title = "所有样例的验收分类均符合预期"
        body = (
            "编码新增能量已从源自身前奏中分离；检查报告中的 A/B 音频和逐起音数据"
            "即可复核。"
        )
    else:
        title = "有样例需要人工复核"
        body = "请查看具体编码卡片中与预期不一致的起音分类和波形。"
    css = "pass" if passed else "fail"
    return f"""
<div class="status {css}">
  <div><h2>{_escape(title)}</h2><p>{_escape(body)}</p></div>
  <div class="pill {css}">{passed_cases}/{total_cases} 样例通过</div>
</div>"""


def _hero(result: LabResult) -> str:
    codec_runs = sum(len(case.codecs) for case in result.cases)
    max_added_values = [
        value for value in (_max_added(case) for case in result.cases) if value is not None
    ]
    max_added = max(max_added_values) if max_added_values else -180.0
    passed_cases = result.acceptance["passed_cases"]
    total_cases = result.acceptance["total_cases"]
    return f"""
<header class="hero">
  <div class="wrap hero-grid">
    <div>
      <div class="eyebrow">Codec pre-onset leakage · lab batch</div>
      <h1>强瞬态有损编码<br>起音前泄漏报告</h1>
      <p>
        同一 {result.sample_rate / 1000:g} kHz 单声道参考分别经过 MP3、Vorbis 与 Opus；
        先校准编解码延迟，再用 {result.settings.window_ms:g} ms 起音前窗口拆解真实前奏与编码新增能量。
      </p>
    </div>
    <div class="hero-stats">
      <div class="hero-stat"><span>样例</span><strong>{len(result.cases)}</strong></div>
      <div class="hero-stat"><span>编码往返</span><strong>{codec_runs}</strong></div>
      <div class="hero-stat"><span>最差编码新增</span><strong>{_fmt_db(max_added)}</strong></div>
      <div class="hero-stat"><span>验收样例</span><strong>{passed_cases}/{total_cases}</strong></div>
    </div>
  </div>
</header>"""


def _codec_strip(result: LabResult) -> str:
    if not result.cases:
        return ""
    chips = []
    for codec in result.cases[0].codecs:
        chips.append(
            f"""<div class="codec-chip">
  <span>{_escape(codec.config.encoder)}</span>
  <h3>{_escape(codec.config.label)}</h3>
  <p>{_escape(codec.config.target)} · 实际
     {codec.encoded_bytes * 8 / max(result.cases[0].reference.duration_s, 1e-9) / 1000:.1f} kbps</p>
</div>"""
        )
    return f'<div class="codec-strip">{"".join(chips)}</div>'


def _method_section(settings: MetricSettings) -> str:
    return f"""
<section class="method">
  <h2>测量与判定口径</h2>
  <p>
    设参考和校准后解码在起音前窗口内的平均能量为
    <code>E_ref</code> 与 <code>E_dec</code>。报告不会把总的前窗能量直接叫成
    预回声，而是先拆分：
  </p>
  <p>
    <code>E_added = max(0, E_dec − E_ref)</code>，
    <code>源/新增 = 10·log10(E_ref / E_added)</code>。
  </p>
  <div class="method-grid">
    <div class="mini">
      <span>01 / ALIGN</span>
      <h3>先校准，后测量</h3>
      <p>用能量包络归一化互相关搜索残余编解码延迟；正延迟从解码信号头部裁掉，
      负延迟补零，A/B 文件与单声道对齐文件严格同长度。报告的是往返后的残余延迟；
      FFmpeg 能使用 codec 的 gapless 信息时，该值可为 0 ms。</p>
    </div>
    <div class="mini">
      <span>02 / WINDOW</span>
      <h3>{settings.window_ms:g} ms 测量窗</h3>
      <p>窗口结束于起音前 {settings.guard_ms:g} ms；保护带避免主敲击直接落入测量。
      表中“前窗 / 起音”只作上下文参考，不是判定依据。</p>
    </div>
    <div class="mini">
      <span>03 / PROVENANCE</span>
      <h3>区分来源</h3>
      <p>参考能量 ≥ {settings.source_visible_dbfs:g} dBFS 且源高于新增
      ≥ {settings.source_dominance_db:g} dB 时标为“源前奏占优”；
      参考 ≤ {settings.silence_dbfs:g} dBFS 且新增可测时标为“编码新增”。</p>
    </div>
  </div>
  <p class="foot">
    本页所有数值由 <code>python3 lab_batch.py</code> 生成。逐起音原始数值见
    <a href="metrics.csv">metrics.csv</a>，延迟与文件路径见
    <a href="summary.json">summary.json</a>。
  </p>
</section>"""


SCRIPT = r"""
(() => {
  const fmtTime = (seconds) => {
    if (!Number.isFinite(seconds) || seconds < 0) seconds = 0;
    const whole = Math.floor(seconds);
    return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
  };
  document.querySelectorAll(".ab-player").forEach((player) => {
    const id = player.dataset.player;
    const audioA = document.getElementById(`${id}-a`);
    const audioB = document.getElementById(`${id}-b`);
    if (!audioA || !audioB) return;
    const playButton = player.querySelector('[data-action="play"]');
    const progress = player.querySelector(".ab-progress");
    const timeLabel = player.querySelector(".ab-time");
    const sideButtons = Array.from(player.querySelectorAll("[data-side]"));
    let side = "A";
    const active = () => (side === "A" ? audioA : audioB);
    const inactive = () => (side === "A" ? audioB : audioA);
    const showTime = () => {
      const current = active();
      const duration = Number.isFinite(current.duration) ? current.duration : 0;
      timeLabel.textContent = `${fmtTime(current.currentTime)} / ${fmtTime(duration)}`;
      if (!progress.matches(":active")) {
        const fraction = duration > 0 ? current.currentTime / duration : 0;
        progress.value = String(Math.min(100, Math.max(0, fraction * 100)));
      }
    };
    const syncInactive = (force = false) => {
      const currentTime = active().currentTime;
      if (!Number.isFinite(currentTime)) return;
      if (force || Math.abs(inactive().currentTime - currentTime) > 0.03) {
        try { inactive().currentTime = currentTime; } catch (_) {}
      }
    };
    const playActive = () => {
      const current = active();
      if (current.ended) current.currentTime = 0;
      syncInactive(true);
      current.play().catch(() => {
        playButton.textContent = "▶ 播放";
      });
    };
    playButton.addEventListener("click", () => {
      if (active().paused) playActive(); else active().pause();
    });
    sideButtons.forEach((button) => {
      button.addEventListener("click", () => {
        const next = button.dataset.side;
        if (next === side) return;
        const wasPlaying = !active().paused;
        const currentTime = active().currentTime;
        active().pause();
        side = next;
        sideButtons.forEach((item) => {
          item.classList.toggle("active", item.dataset.side === side);
        });
        try { active().currentTime = currentTime; } catch (_) {}
        playButton.textContent = wasPlaying ? "❚❚ 暂停" : "▶ 播放";
        if (wasPlaying) playActive();
        showTime();
      });
    });
    progress.addEventListener("input", () => {
      const duration = Number.isFinite(active().duration) ? active().duration : 0;
      const target = duration * Number(progress.value) / 100;
      [audioA, audioB].forEach((audio) => {
        try { audio.currentTime = target; } catch (_) {}
      });
      showTime();
    });
    [audioA, audioB].forEach((audio) => {
      audio.addEventListener("timeupdate", () => {
        if (audio !== active()) return;
        showTime();
        syncInactive();
      });
      audio.addEventListener("loadedmetadata", () => {
        if (audio !== active()) {
          try { audio.currentTime = active().currentTime; } catch (_) {}
        }
        showTime();
      });
      audio.addEventListener("play", () => {
        if (audio === active()) playButton.textContent = "❚❚ 暂停";
      });
      audio.addEventListener("pause", () => {
        if (audio === active()) playButton.textContent = "▶ 播放";
      });
      audio.addEventListener("ended", () => {
        if (audio === active()) playButton.textContent = "▶ 播放";
      });
    });
    showTime();
  });
})();
"""


def build_report(result: LabResult) -> Path:
    sections = "\n".join(_case_section(case, result.settings) for case in result.cases)
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>强瞬态有损编码泄漏报告</title>
  <style>{CSS}</style>
</head>
<body>
{_hero(result)}
<main class="wrap">
  {_status_banner(result)}
  <div class="method-grid">
    <div class="mini"><span>统一参考</span><h3>同一强瞬态源</h3>
      <p>{result.sample_rate // 1000} kHz 单声道 PCM，三种配置从同一文件编码。</p></div>
    <div class="mini"><span>测量顺序</span><h3>解码 → 延迟校准 → 能量</h3>
      <p>延迟在计算泄漏前校正；未对齐文件也保留，便于审计。</p></div>
    <div class="mini"><span>核心判据</span><h3>源能量与编码新增分离</h3>
      <p>先看参考前窗是否有真实能量，再看扣除后的新增能量。</p></div>
  </div>
  {_codec_strip(result)}
  {sections}
  {_method_section(result.settings)}
  <p class="foot">生成时间：{_escape(result.generated_at)} · 报告目录：
    {_escape(result.output_dir.resolve())}</p>
</main>
<script>{SCRIPT}</script>
</body>
</html>
"""
    path = result.output_dir / "report.html"
    path.write_text(document, encoding="utf-8")
    return path
