from __future__ import annotations

import math
from pathlib import Path
from statistics import median

from . import media


def time_label(value: float) -> str:
    value = max(0.0, float(value))
    minutes = int(value // 60)
    seconds = value - minutes * 60
    return f"{minutes:02d}:{seconds:05.2f}"


def _lut() -> list[float]:
    values = []
    for value in range(256):
        channel = value / 255.0
        values.append(
            channel / 12.92
            if channel <= 0.04045
            else ((channel + 0.055) / 1.055) ** 2.4
        )
    return values


LINEAR = _lut()


def _contrast(foreground: float, background: float) -> float:
    high = max(foreground, background) + 0.05
    low = min(foreground, background) + 0.05
    return high / low


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _mean_filter(values: list[float], width: int, height: int, radius: int) -> list[float]:
    stride = width + 1
    integral = [0.0] * (stride * (height + 1))
    for y in range(height):
        row_sum = 0.0
        source_offset = y * width
        target_offset = (y + 1) * stride
        above_offset = y * stride
        for x in range(width):
            row_sum += values[source_offset + x]
            integral[target_offset + x + 1] = integral[above_offset + x + 1] + row_sum
    output = [0.0] * (width * height)
    for y in range(height):
        top = max(0, y - radius)
        bottom = min(height, y + radius + 1)
        for x in range(width):
            left = max(0, x - radius)
            right = min(width, x + radius + 1)
            total = (
                integral[bottom * stride + right]
                - integral[top * stride + right]
                - integral[bottom * stride + left]
                + integral[top * stride + left]
            )
            count = (right - left) * (bottom - top)
            output[y * width + x] = total / count if count else 0.0
    return output


def _luma_map(frame: bytes, width: int, height: int) -> list[float]:
    luma = [0.0] * (width * height)
    for index in range(width * height):
        offset = index * 3
        luma[index] = (
            LINEAR[frame[offset]] * 0.2126
            + LINEAR[frame[offset + 1]] * 0.7152
            + LINEAR[frame[offset + 2]] * 0.0722
        )
    return luma


def _padded_coordinates(
    mask: bytes,
    width: int,
    height: int,
    pad_left: int,
    pad_top: int,
    stride: int,
    predicate,
) -> list[int]:
    coordinates = []
    for y in range(height):
        row = y * width
        padded_row = (y + pad_top) * stride + pad_left
        for x in range(width):
            if predicate(mask[row + x]):
                coordinates.append(padded_row + x)
    return coordinates


def _frame_metric(
    cue: dict,
    frame_index: int,
    fps: float,
    background: list[float],
    core: list[int],
    ring: list[int],
    global_luma: float,
    previous_global: list[float],
    thresholds: dict,
    version: str,
) -> dict:
    background_core = [background[index] for index in core]
    background_ring = [background[index] for index in ring] or background_core
    fill_values = [_contrast(1.0, value) for value in background_core]
    edge_values = [_contrast(0.0, value) for value in background_ring]
    fill_p10 = _percentile(fill_values, 0.10)
    fill_p20 = _percentile(fill_values, 0.20)
    fill_median = median(fill_values) if fill_values else 0.0
    edge_p10 = _percentile(edge_values, 0.10)
    edge_p20 = _percentile(edge_values, 0.20)
    edge_median = median(edge_values) if edge_values else 0.0
    contrast_threshold = thresholds["contrast"]
    fill_coverage = (
        sum(value >= contrast_threshold for value in fill_values) / len(fill_values)
        if fill_values else 0.0
    )
    edge_coverage = (
        sum(value >= contrast_threshold for value in edge_values) / len(edge_values)
        if edge_values else 0.0
    )
    if version == "outlined":
        contrast_p10 = max(fill_p10, edge_p10)
        contrast_p20 = max(fill_p20, edge_p20)
        contrast_median = max(fill_median, edge_median)
        coverage = max(fill_coverage, edge_coverage)
    else:
        contrast_p10 = fill_p10
        contrast_p20 = fill_p20
        contrast_median = fill_median
        coverage = fill_coverage
    background_median = median(background_core) if background_core else 0.0
    background_p90 = _percentile(background_core, 0.90)
    recent = previous_global[-8:] + [global_luma]
    recent_floor = min(recent) if recent else global_luma
    flash = global_luma > 0.68 and global_luma - recent_floor > 0.075
    readable = (
        contrast_p20 >= contrast_threshold
        and coverage >= thresholds["coverage"]
    )
    if flash:
        reason = "闪光过曝，白字与背景亮度接近"
    elif background_median > 0.42:
        reason = "高亮背景吞没白色笔画"
    elif background_p90 > 0.40:
        reason = "局部高光压低笔画对比"
    else:
        reason = "笔画与局部背景亮度差不足"
    return {
        "time": round(frame_index / fps, 6),
        "frame_index": frame_index,
        "cue_id": cue["id"],
        "contrast_p10": round(contrast_p10, 4),
        "contrast_p20": round(contrast_p20, 4),
        "contrast_median": round(contrast_median, 4),
        "coverage": round(coverage, 4),
        "bg_luma": round(background_median, 4),
        "bg_p90": round(background_p90, 4),
        "fill_p20": round(fill_p20, 4),
        "fill_coverage": round(fill_coverage, 4),
        "edge_p20": round(edge_p20, 4),
        "edge_coverage": round(edge_coverage, 4),
        "global_luma": round(global_luma, 4),
        "readable": readable,
        "flash": flash,
        "reason": reason,
    }


def _cue_frame_indices(cue: dict, fps: float, frame_count: int) -> range:
    first = int(math.ceil(cue["start"] * fps - 1e-9))
    last = int(math.floor(cue["end"] * fps - 1e-9))
    first = max(0, first)
    last = min(frame_count - 1, last)
    return range(first, last + 1)


def _analyze_segment(
    cue: dict,
    base: Path,
    video_info: dict,
    global_values: list[float],
    thresholds: dict,
) -> dict:
    width, height = cue["bbox"][2], cue["bbox"][3]
    margin = max(18, thresholds["outline_width"] + 12)
    x, y = cue["bbox"][0], cue["bbox"][1]
    pad_left = x - max(0, x - margin)
    pad_top = y - max(0, y - margin)
    crop_right = min(
        video_info["width"],
        x + width + margin,
    )
    crop_bottom = min(
        video_info["height"],
        y + height + margin,
    )
    crop_width = crop_right - max(0, x - margin)
    crop_height = crop_bottom - max(0, y - margin)
    mask = media.read_gray(cue["mask_path"], width, height)
    core = _padded_coordinates(
        mask,
        width,
        height,
        pad_left,
        pad_top,
        crop_width,
        lambda value: value >= 128,
    )
    ring_mask = media.dilate_binary(mask, width, height, thresholds["outline_width"])
    ring = _padded_coordinates(
        ring_mask,
        width,
        height,
        pad_left,
        pad_top,
        crop_width,
        lambda value: value >= 128,
    )
    ring = [
        index
        for index in ring
        if mask[
            (index // crop_width - pad_top) * width
            + (index % crop_width - pad_left)
        ] < 128
    ]
    if not core:
        raise media.MediaError(f"字幕段 {cue['id']} 的掩模没有前景像素。")

    active = _cue_frame_indices(cue, video_info["fps"], video_info["frame_count"])
    by_version = {"plain": [], "outlined": []}
    frames = media.iter_crop_frames(
        base,
        tuple(cue["bbox"]),
        margin,
        video_info["frame_count"],
        video_info["width"],
        video_info["height"],
    )
    for frame_index, frame in enumerate(frames):
        if frame_index not in active:
            continue
        luma = _luma_map(frame, crop_width, crop_height)
        background = _mean_filter(luma, crop_width, crop_height, 12)
        previous = global_values[max(0, frame_index - 8):frame_index]
        global_luma = (
            global_values[frame_index]
            if frame_index < len(global_values)
            else 0.0
        )
        for version in ("plain", "outlined"):
            metric = _frame_metric(
                cue,
                frame_index,
                video_info["fps"],
                background,
                core,
                ring,
                global_luma,
                previous,
                thresholds,
                version,
            )
            metric["version"] = version
            by_version[version].append(metric)
    if not by_version["plain"]:
        raise media.MediaError(f"字幕段 {cue['id']} 未解码到任何帧。")
    return by_version


def _episodes(metrics: list[dict], fps: float, transient_max: float) -> list[dict]:
    groups: list[list[dict]] = []
    current: list[dict] = []
    for metric in metrics:
        if not metric["readable"]:
            current.append(metric)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    episodes = []
    for group in groups:
        first = group[0]
        last = group[-1]
        worst = min(group, key=lambda item: (item["contrast_p20"], item["coverage"]))
        reason = worst["reason"]
        if any(item["flash"] for item in group):
            reason = "闪光过曝，白字与背景亮度接近"
        start = first["time"]
        end = (last["frame_index"] + 1) / fps
        duration = len(group) / fps
        episodes.append({
            "cue_id": first["cue_id"],
            "start": round(start, 6),
            "end": round(end, 6),
            "duration": round(duration, 6),
            "kind": "sustained" if duration > transient_max else "transient",
            "worst_time": worst["time"],
            "worst_contrast": worst["contrast_p20"],
            "worst_coverage": worst["coverage"],
            "reason": reason,
            "flash": any(item["flash"] for item in group),
            "frame_indices": [item["frame_index"] for item in group],
        })
    return episodes


def _attach_episode_links(
    cue_id: str,
    plain_episodes: list[dict],
    outlined_episodes: list[dict],
) -> None:
    outlined_frames = {
        frame
        for episode in outlined_episodes
        for frame in episode["frame_indices"]
    }
    for index, episode in enumerate(plain_episodes, start=1):
        overlap = len(set(episode["frame_indices"]) & outlined_frames)
        episode["id"] = f"{cue_id}-p{index:02d}"
        episode["version"] = "plain"
        episode["overlap_other"] = overlap
        episode["rescued"] = overlap == 0
        episode["start_label"] = time_label(episode["start"])
        episode["end_label"] = time_label(episode["end"])
        episode["worst_label"] = time_label(episode["worst_time"])
    for index, episode in enumerate(outlined_episodes, start=1):
        overlap = 0
        plain_frames = {
            frame
            for item in plain_episodes
            for frame in item["frame_indices"]
        }
        overlap = len(set(episode["frame_indices"]) & plain_frames)
        episode["id"] = f"{cue_id}-o{index:02d}"
        episode["version"] = "outlined"
        episode["overlap_other"] = overlap
        episode["rescued"] = False
        episode["start_label"] = time_label(episode["start"])
        episode["end_label"] = time_label(episode["end"])
        episode["worst_label"] = time_label(episode["worst_time"])


def _clean_metric(metric: dict) -> dict:
    return {
        key: value
        for key, value in metric.items()
        if key not in ("frame_indices", "version")
    }


def load_manifest(path: Path) -> tuple[dict, list[dict]]:
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    cues = []
    for index, item in enumerate(data.get("cues") or [], start=1):
        cue_id = str(item.get("id") or f"c{index:02d}")
        bbox = item.get("bbox")
        if not bbox or len(bbox) != 4:
            raise ValueError(f"字幕段 {cue_id} 缺少 bbox=[x,y,w,h]。")
        cue = dict(item)
        cue["id"] = cue_id
        cue["start"] = float(item["start"])
        cue["end"] = float(item["end"])
        cue["bbox"] = [int(value) for value in bbox]
        mask_value = item.get("mask")
        if not mask_value:
            raise ValueError(f"字幕段 {cue_id} 缺少 mask。")
        mask_path = Path(mask_value)
        cue["mask_path"] = mask_path if mask_path.is_absolute() else path.parent / mask_path
        if not cue["mask_path"].exists():
            raise FileNotFoundError(f"找不到字幕掩模：{cue['mask_path']}")
        if cue["end"] <= cue["start"]:
            raise ValueError(f"字幕段 {cue_id} 的时间范围无效。")
        cues.append(cue)
    if not cues:
        raise ValueError("manifest 中没有字幕段。")
    return data, cues


def _status(plain_episodes: list[dict], outlined_episodes: list[dict]) -> dict:
    if not plain_episodes:
        return {
            "status": "pass",
            "status_label": "通过",
            "severity": 0,
            "risk": False,
        }
    sustained = any(item["kind"] == "sustained" for item in plain_episodes)
    if outlined_episodes:
        label = "双版本持续不可读" if sustained else "双版本短时不可读"
        return {
            "status": "both_versions",
            "status_label": label,
            "severity": 3 if sustained else 2,
            "risk": True,
        }
    if sustained:
        return {
            "status": "sustained_mitigated",
            "status_label": "持续偏弱 · 描边可解",
            "severity": 2,
            "risk": True,
        }
    return {
        "status": "transient_mitigated",
        "status_label": "短时偏弱 · 描边可解",
        "severity": 1,
        "risk": True,
    }


def analyze_project(
    base: Path,
    manifest_path: Path,
    thresholds: dict,
) -> tuple[dict, list[dict]]:
    manifest, cues = load_manifest(manifest_path)
    video_info = media.probe_video(base)
    global_values = media.global_luma_series(base, video_info["frame_count"])
    inside = []
    for cue in cues:
        x, y, width, height = cue["bbox"]
        if width <= 0 or height <= 0:
            raise ValueError(f"字幕段 {cue['id']} 的 bbox 尺寸无效。")
        if x < 0 or y < 0 or x + width > video_info["width"] or y + height > video_info["height"]:
            raise ValueError(f"字幕段 {cue['id']} 的 bbox 超出视频画面。")
        mask = media.read_gray(cue["mask_path"], width, height)
        if not any(value >= 128 for value in mask):
            raise ValueError(f"字幕段 {cue['id']} 的掩模没有前景。")
        inside.append(cue)

    segments = []
    all_metrics = []
    for index, cue in enumerate(inside, start=1):
        raw = _analyze_segment(cue, base, video_info, global_values, thresholds)
        plain_metrics = raw["plain"]
        outlined_metrics = raw["outlined"]
        all_metrics.extend(plain_metrics)
        all_metrics.extend(outlined_metrics)
        plain_episodes = _episodes(
            plain_metrics,
            video_info["fps"],
            thresholds["transient_max"],
        )
        outlined_episodes = _episodes(
            outlined_metrics,
            video_info["fps"],
            thresholds["transient_max"],
        )
        _attach_episode_links(cue["id"], plain_episodes, outlined_episodes)
        plain_worst = min(
            plain_metrics,
            key=lambda item: (item["contrast_p20"], item["coverage"]),
        )
        outlined_worst = min(
            outlined_metrics,
            key=lambda item: (item["contrast_p20"], item["coverage"]),
        )
        outlined_by_frame = {
            item["frame_index"]: item for item in outlined_metrics
        }
        outlined_at_worst = outlined_by_frame.get(
            plain_worst["frame_index"],
            outlined_worst,
        )
        state = _status(plain_episodes, outlined_episodes)
        worst_reason = (
            plain_worst["reason"]
            if plain_episodes
            else "本段未触发低对比事件"
        )
        segment = {
            "id": cue["id"],
            "index": index,
            "text": cue.get("text", ""),
            "start": cue["start"],
            "end": cue["end"],
            "start_label": time_label(cue["start"]),
            "end_label": time_label(cue["end"]),
            **state,
            "worst": {
                "time": plain_worst["time"],
                "time_label": time_label(plain_worst["time"]),
                "version": "plain",
                "contrast": plain_worst["contrast_p20"],
                "reason": worst_reason,
            },
            "plain_worst": {
                "time": plain_worst["time"],
                "time_label": time_label(plain_worst["time"]),
                "contrast": plain_worst["contrast_p20"],
                "fill": plain_worst["fill_p20"],
                "coverage": plain_worst["coverage"],
            },
            "outlined_worst": {
                "time": outlined_worst["time"],
                "time_label": time_label(outlined_worst["time"]),
                "contrast": outlined_worst["contrast_p20"],
                "fill": outlined_worst["fill_p20"],
                "edge": outlined_worst["edge_p20"],
                "coverage": outlined_worst["coverage"],
            },
            "at_worst": {
                "plain": _clean_metric(plain_worst),
                "outlined": _clean_metric(outlined_at_worst),
            },
            "mean_plain": round(
                sum(item["contrast_p20"] for item in plain_metrics)
                / len(plain_metrics),
                2,
            ),
            "mean_outlined": round(
                sum(item["contrast_p20"] for item in outlined_metrics)
                / len(outlined_metrics),
                2,
            ),
            "episodes": [
                {key: value for key, value in item.items() if key != "frame_indices"}
                for item in plain_episodes
            ],
            "plain_episodes": [
                {key: value for key, value in item.items() if key != "frame_indices"}
                for item in plain_episodes
            ],
            "outlined_episodes": [
                {key: value for key, value in item.items() if key != "frame_indices"}
                for item in outlined_episodes
            ],
            "series": {
                "times": [round(item["time"], 3) for item in plain_metrics],
                "plain": [round(item["contrast_p20"], 3) for item in plain_metrics],
                "outlined": [
                    round(item["contrast_p20"], 3)
                    for item in outlined_metrics
                ],
            },
        }
        segments.append(segment)

    report = build_report(
        manifest,
        video_info,
        thresholds,
        segments,
    )
    return report, all_metrics


def build_report(
    manifest: dict,
    video_info: dict,
    thresholds: dict,
    segments: list[dict],
) -> dict:
    from datetime import datetime, timezone

    risk_segments = [item for item in segments if item["risk"]]
    sustained = [
        item
        for item in risk_segments
        if item["status"] in ("sustained_mitigated", "both_versions")
        and any(
            episode["kind"] == "sustained"
            for episode in item["plain_episodes"]
        )
    ]
    transient = [
        item
        for item in risk_segments
        if item not in sustained
    ]
    plain_episodes = [
        episode
        for item in segments
        for episode in item["plain_episodes"]
    ]
    outlined_episodes = [
        episode
        for item in segments
        for episode in item["outlined_episodes"]
    ]
    lowest = min(
        segments,
        key=lambda item: (
            item["plain_worst"]["contrast"],
            item["plain_worst"]["coverage"],
        ),
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "meta": {
            "title": manifest.get("title", "硬字幕对比审查"),
            "manifest": "manifest.json",
            "duration": round(video_info["duration"], 6),
            "duration_label": time_label(video_info["duration"]),
            "fps": round(video_info["fps"], 6),
            "width": video_info["width"],
            "height": video_info["height"],
        },
        "thresholds": thresholds,
        "videos": {
            "plain": "videos/plain.mp4",
            "outlined": "videos/outlined.mp4",
        },
        "segments": segments,
        "summary": {
            "segments": len(segments),
            "pass": len(segments) - len(risk_segments),
            "risk": len(risk_segments),
            "sustained": len(sustained),
            "transient": len(transient),
            "both_versions": sum(
                item["status"] == "both_versions" for item in segments
            ),
            "mitigated": sum(
                bool(item["plain_episodes"]) and not item["outlined_episodes"]
                for item in segments
            ),
            "plain_episodes": len(plain_episodes),
            "outlined_episodes": len(outlined_episodes),
            "outline_rescued_episodes": sum(
                episode["rescued"] for episode in plain_episodes
            ),
            "plain_fail_seconds": round(
                sum(episode["duration"] for episode in plain_episodes),
                3,
            ),
            "outlined_fail_seconds": round(
                sum(episode["duration"] for episode in outlined_episodes),
                3,
            ),
            "transient_max": thresholds["transient_max"],
            "lowest": {
                "cue_id": lowest["id"],
                "time": lowest["plain_worst"]["time"],
                "time_label": lowest["plain_worst"]["time_label"],
                "contrast": lowest["plain_worst"]["contrast"],
                "reason": lowest["worst"]["reason"],
                "status": lowest["status_label"],
            },
        },
    }
