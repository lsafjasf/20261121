from __future__ import annotations

import csv
import html
import json
import shutil
from pathlib import Path

from . import media
from .analyze import time_label


def _prepare_layers(
    cues: list[dict],
    manifest: dict,
    work: Path,
    outline_width: int,
) -> dict[str, dict[str, Path]]:
    work.mkdir(parents=True, exist_ok=True)
    style = manifest.get("subtitle_style") or {}
    fill_color = tuple(style.get("text_color") or (255, 255, 255))
    outline_color = tuple(style.get("outline_color") or (0, 0, 0))
    layers: dict[str, dict[str, Path]] = {}
    for cue in cues:
        x, y, width, height = cue["bbox"]
        mask = media.read_gray(cue["mask_path"], width, height)
        fill = media.make_rgba_layer(mask, width, height, fill_color)
        outline_mask = media.dilate_binary(mask, width, height, outline_width)
        outline = media.make_rgba_layer(outline_mask, width, height, outline_color)
        fill_path = work / f"{cue['id']}-fill.png"
        outline_path = work / f"{cue['id']}-outline.png"
        media.save_rgba_png(fill, width, height, fill_path)
        media.save_rgba_png(outline, width, height, outline_path)
        layers[cue["id"]] = {
            "fill": fill_path,
            "outline": outline_path,
            "x": x,
            "y": y,
        }
    return layers


def render_videos(
    base: Path,
    manifest: dict,
    cues: list[dict],
    output_dir: Path,
    video_info: dict,
    outline_width: int,
) -> None:
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    work = output_dir / ".layers"
    layers = _prepare_layers(cues, manifest, work, outline_width)
    common = {
        "base": base,
        "fps": video_info["fps"],
        "duration": video_info["duration"],
        "output": video_dir / "plain.mp4",
    }
    plain_cues = []
    outlined_cues = []
    for cue in cues:
        prepared = layers[cue["id"]]
        timing = {"start": cue["start"], "end": cue["end"]}
        plain_cues.append({
            **timing,
            "layers": [{
                "path": prepared["fill"],
                "x": prepared["x"],
                "y": prepared["y"],
            }],
        })
        outlined_cues.append({
            **timing,
            "layers": [
                {
                    "path": prepared["outline"],
                    "x": prepared["x"],
                    "y": prepared["y"],
                },
                {
                    "path": prepared["fill"],
                    "x": prepared["x"],
                    "y": prepared["y"],
                },
            ],
        })
    media.compose_videos(**common, cue_layers=plain_cues)
    media.compose_videos(
        **{**common, "output": video_dir / "outlined.mp4"},
        cue_layers=outlined_cues,
    )
    shutil.rmtree(work, ignore_errors=True)


def render_evidence(report: dict, output_dir: Path) -> None:
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for segment in report["segments"]:
        time = segment["plain_worst"]["time"]
        for version in ("plain", "outlined"):
            media.extract_frame(
                output_dir / "videos" / f"{version}.mp4",
                time,
                evidence_dir / f"{segment['id']}-{version}.jpg",
            )
        segment["evidence"] = {
            "plain": f"evidence/{segment['id']}-plain.jpg",
            "outlined": f"evidence/{segment['id']}-outlined.jpg",
        }


def render_csv(metrics: list[dict], path: Path) -> None:
    fields = [
        "cue_id", "version", "time", "contrast_p10", "contrast_p20",
        "contrast_median", "coverage", "bg_luma", "bg_p90", "fill_p20",
        "fill_coverage", "edge_p20", "edge_coverage", "global_luma",
        "readable", "flash", "reason",
    ]
    ordered = sorted(
        metrics,
        key=lambda item: (item["cue_id"], item["version"], item["time"]),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in ordered:
            writer.writerow({key: item.get(key, "") for key in fields})


def render_data(report: dict, output_dir: Path) -> None:
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    compact = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    (output_dir / "report-data.js").write_text(
        f"window.REVIEW_DATA = {compact};\n",
        encoding="utf-8",
    )


def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def _summary_html(report: dict) -> str:
    summary = report["summary"]
    lowest = summary["lowest"]
    stats = [
        (summary["segments"], "字幕段", ""),
        (summary["risk"], "需要回看", "warn"),
        (summary["sustained"], "持续偏弱段", "warn"),
        (summary["transient"], "短时风险段", ""),
        (f"{lowest['contrast']:.2f}:1", "全局最低 P20", "warn"),
    ]
    return "".join(
        f'<div class="stat"><div class="stat-number {class_name}">{_esc(value)}</div>'
        f'<div class="stat-label">{_esc(label)}</div></div>'
        for value, label, class_name in stats
    )


def _episode_html(episode: dict) -> str:
    version = episode["version"]
    kind = episode["kind"]
    if version == "plain":
        outcome = "描边救回" if episode.get("rescued") else "描边仍不可读"
    else:
        outcome = "描边仍不可读"
    return (
        f'<div class="event {_esc(version)}">'
        f'<span class="event-version">{"无描边" if version == "plain" else "有描边"}</span>'
        f'<span class="event-text"><span class="event-kind {_esc(kind)}">'
        f'{"持续" if kind == "sustained" else "短时"} {episode["duration"]:.2f}s</span> '
        f'{_esc(episode["reason"])} · {_esc(outcome)}</span>'
        f'<button class="jump" data-time="{episode["worst_time"]:.6f}" '
        f'data-version="{_esc(version)}" data-cue="{_esc(episode["cue_id"])}">看帧</button>'
        f"</div>"
    )


def _segment_html(segment: dict, threshold: float) -> str:
    number = f"{segment['index']:02d}"
    low_class = " low" if segment["risk"] else ""
    worst_time = segment["plain_worst"]["time"]
    out_time = segment["outlined_worst"]["time"]
    events = segment["plain_episodes"] + segment["outlined_episodes"]
    event_html = "".join(_episode_html(item) for item in events)
    if not event_html:
        event_html = '<div class="no-events">✓ 双版本均未触发低对比事件</div>'
    gain = (
        segment["outlined_worst"]["contrast"]
        - segment["plain_worst"]["contrast"]
    )
    evidence = segment["evidence"]
    return f"""
<article class="segment-card" id="cue-{_esc(segment['id'])}" data-cue="{_esc(segment['id'])}">
  <div class="segment-index">{number}</div>
  <div class="segment-main">
    <div class="segment-head">
      <span class="cue-time">{_esc(segment['start_label'])} — {_esc(segment['end_label'])}</span>
      <span class="status {_esc(segment['status'])}">{_esc(segment['status_label'])}</span>
    </div>
    <p class="caption">{_esc(segment['text'])}</p>
    <div class="metric-line">
      <span class="metric-big{low_class}">{segment['plain_worst']['contrast']:.2f}</span>
      <span class="metric-unit">:1 · P20 最低对比</span>
    </div>
    <div class="reason">{_esc(segment['worst']['reason'])}</div>
    <div class="mini-stats">
      <div class="mini-stat"><span>无描边最低</span><b>{segment['plain_worst']['contrast']:.2f}:1</b></div>
      <div class="mini-stat"><span>有描边最低</span><b>{segment['outlined_worst']['contrast']:.2f}:1</b></div>
      <div class="mini-stat"><span>描边增益</span><b>{gain:+.2f}</b></div>
    </div>
    <div class="card-actions">
      <button class="jump" data-time="{worst_time:.6f}" data-version="plain" data-cue="{_esc(segment['id'])}" data-scroll="true">定位最低对比帧</button>
      <button class="jump secondary" data-time="{worst_time:.6f}" data-version="plain" data-cue="{_esc(segment['id'])}">无描边最低帧</button>
      <button class="jump secondary" data-time="{out_time:.6f}" data-version="outlined" data-cue="{_esc(segment['id'])}">有描边最低帧</button>
    </div>
  </div>
  <div class="segment-side">
    <div class="evidence-pair">
      <button class="evidence jump" data-time="{worst_time:.6f}" data-version="plain" data-cue="{_esc(segment['id'])}" aria-label="定位无描边证据帧">
        <img src="{_esc(evidence['plain'])}" alt="无描边证据帧">
        <span class="evidence-label">A · 无描边</span>
      </button>
      <button class="evidence jump" data-time="{worst_time:.6f}" data-version="outlined" data-cue="{_esc(segment['id'])}" aria-label="定位有描边证据帧">
        <img src="{_esc(evidence['outlined'])}" alt="有描边证据帧">
        <span class="evidence-label">B · 有描边</span>
      </button>
    </div>
    <div class="event-list">{event_html}</div>
  </div>
</article>"""


def _page_context(report: dict) -> dict[str, str]:
    meta = report["meta"]
    summary = report["summary"]
    thresholds = report["thresholds"]
    if summary["risk"]:
        headline = f'{summary["risk"]} 段字幕<em>需要回看</em>'
    else:
        headline = "字幕对比<em>全部通过</em>"
    lede = (
        f'在 {summary["segments"]} 个字幕段中识别出 '
        f'{summary["sustained"]} 段持续高亮阻塞与 '
        f'{summary["transient"]} 段短时风险。全局最低帧位于 '
        f'{summary["lowest"]["time_label"]}，'
        f'{summary["lowest"]["reason"]}。'
    )
    return {
        "TITLE": _esc(meta["title"]),
        "DURATION": _esc(meta["duration_label"]),
        "DURATION_SECONDS": f'{meta["duration"]:.6f}',
        "FPS": f'{meta["fps"]:g}',
        "WIDTH": str(meta["width"]),
        "HEIGHT": str(meta["height"]),
        "HEADLINE": headline,
        "LEDE": _esc(lede),
        "THRESHOLD": f'{thresholds["contrast"]:g}',
        "COVERAGE": f'{thresholds["coverage"] * 100:.0f}%',
        "TRANSIENT_MAX": f'{thresholds["transient_max"]:g}',
        "OUTLINE_WIDTH": str(thresholds["outline_width"]),
        "SUMMARY": _summary_html(report),
        "SEGMENTS": "".join(
            _segment_html(segment, thresholds["contrast"])
            for segment in report["segments"]
        ),
        "GENERATED_AT": _esc(report["generated_at"]),
        "MANIFEST": _esc(meta["manifest"]),
    }


def _write_output_manifest(
    manifest: dict,
    cues: list[dict],
    output_dir: Path,
) -> None:
    output_manifest = json.loads(json.dumps(manifest, ensure_ascii=False))
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for source_cue, output_cue in zip(cues, output_manifest["cues"]):
        suffix = source_cue["mask_path"].suffix or ".png"
        name = f"{source_cue['id']}{suffix}"
        destination = mask_dir / name
        if source_cue["mask_path"].resolve() != destination.resolve():
            shutil.copy2(source_cue["mask_path"], destination)
        output_cue["mask"] = f"masks/{name}"
        output_cue["bbox"] = source_cue["bbox"]
    (output_dir / "manifest.json").write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def render_site(
    report: dict,
    manifest: dict,
    cues: list[dict],
    output_dir: Path,
    template_dir: Path,
    base: Path,
) -> None:
    assets = output_dir / "assets"
    fonts = assets / "fonts"
    fonts.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template_dir / "review.css", assets / "review.css")
    shutil.copy2(template_dir / "review.js", assets / "review.js")
    font = template_dir / "assets" / "fonts" / "NotoSansSC-UI.woff2"
    if font.exists():
        shutil.copy2(font, fonts / "NotoSansSC-UI.woff2")
    template = (template_dir / "page.html").read_text(encoding="utf-8")
    page = template
    for key, value in _page_context(report).items():
        page = page.replace("{{" + key + "}}", value)
    if "{{" in page:
        missing = sorted({part.split("}}", 1)[0] for part in page.split("{{")[1:]})
        raise ValueError(f"页面模板仍有未替换字段：{', '.join(missing)}")
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    (output_dir / "review.html").write_text(page, encoding="utf-8")
    _write_output_manifest(manifest, cues, output_dir)
    target_base = output_dir / "base.mp4"
    if not target_base.exists() or target_base.resolve() != base.resolve():
        try:
            if target_base.resolve() != base.resolve():
                shutil.copy2(base, target_base)
        except OSError:
            shutil.copy2(base, target_base)
