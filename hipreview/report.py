"""审片报告：JSON 结论、文本摘要与问题卷波形片段导出。"""

from __future__ import annotations

import json
import os

import numpy as np

from .audioio import export_wav, read_wav_mono, resample_linear, slice_segment
from .analyzer import analyze_playlist
from .synth import SAMPLE_RATE

CATEGORY_LABELS = {
    "DELAY": "声轨延后",
    "DUP_TRACK": "声轨重复(整轨复制主混音)",
    "DUP_SEGMENT": "声轨重复(卷内片段)",
    "LANG_MISMATCH": "语言标记与参考片段不符",
}


def load_playlist_tracks(playlist_path: str,
                         sr: int = SAMPLE_RATE) -> tuple[dict, list[dict]]:
    """读取 playlist.json 并把每卷两轨解码到统一采样率。"""
    base = os.path.dirname(os.path.abspath(playlist_path))
    with open(playlist_path, encoding="utf-8") as fh:
        playlist = json.load(fh)
    reels = []
    for item in playlist["reels"]:
        main, main_sr = read_wav_mono(os.path.join(base, item["main"]))
        hi, hi_sr = read_wav_mono(os.path.join(base, item["hearing"]))
        reels.append({
            "reel_id": item["reel_id"],
            "main": resample_linear(main, main_sr, sr),
            "hi": resample_linear(hi, hi_sr, sr),
            "expected_lang": item.get(
                "expected_lang",
                playlist.get("reference", {}).get("expected_lang")),
        })
    return playlist, reels


def run_audit(playlist_path: str, out_dir: str,
              sr: int = SAMPLE_RATE) -> dict:
    """跑完整审片，落盘报告与问题片段，返回可直接给前端的结果字典。"""
    playlist, reels = load_playlist_tracks(playlist_path, sr)
    results = analyze_playlist(reels, sr)
    clips_dir = os.path.join(out_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    payload = {
        "playlist": playlist,
        "clips_dir": os.path.relpath(clips_dir, out_dir),
        "results": [],
    }
    track_by_id = {r["reel_id"]: r for r in reels}
    for result in results:
        data = result.to_dict()
        for issue in data["issues"]:
            tag = f"{result.reel_id}_{issue['category']}"
            hi_clip = slice_segment(track_by_id[result.reel_id]["hi"], sr,
                                    issue["start_sec"], issue["end_sec"])
            main_clip = slice_segment(track_by_id[result.reel_id]["main"], sr,
                                      issue["start_sec"], issue["end_sec"])
            hi_name = f"{tag}_hi.wav"
            main_name = f"{tag}_main.wav"
            export_wav(os.path.join(clips_dir, hi_name), hi_clip, sr)
            export_wav(os.path.join(clips_dir, main_name), main_clip, sr)
            issue["clip_hi"] = f"clips/{hi_name}"
            issue["clip_main"] = f"clips/{main_name}"
            issue["category_label"] = CATEGORY_LABELS.get(
                issue["category"], issue["category"])
        payload["results"].append(data)

    with open(os.path.join(out_dir, "report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "report.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(render_text(payload))
    return payload


def render_text(payload: dict) -> str:
    """生成人类可读的审片摘要。"""
    lines = [f"审片清单：{payload['playlist']['name']}", "=" * 48]
    total_issues = 0
    for result in payload["results"]:
        lines.append(
            f"\n[{result['reel_id']}] 时长 {result['duration_sec']}s | "
            f"时延 {result['delay_ms']} ms | 相关 "
            f"{result['best_corr']:.3f} | 语言 "
            f"{result['detected_lang'] or '未识别'}"
            f"（参考 {result['expected_lang']}）")
        if not result["issues"]:
            lines.append("  ✓ 未发现问题")
        for issue in result["issues"]:
            total_issues += 1
            lines.append(
                f"  ✗ {issue['category_label']} | "
                f"{issue['start_sec']:.2f}s–{issue['end_sec']:.2f}s | "
                f"{issue['detail']}")
    lines.append("\n" + "=" * 48)
    lines.append(f"问题卷合计：{total_issues} 条")
    return "\n".join(lines) + "\n"
