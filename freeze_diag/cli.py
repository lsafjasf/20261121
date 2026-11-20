from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyzer import analyze_video
from .config import DiagnosticConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="固定机位监控冻结诊断：图像噪声 + 重复帧 + 解码时间戳"
    )
    parser.add_argument("videos", nargs="+", type=Path, help="待诊断视频")
    parser.add_argument("--json", type=Path, help="将完整证据写入 JSON 文件")
    parser.add_argument(
        "--min-freeze-ms", type=float, default=1000.0, help="最短冻结/黑屏区间（毫秒）"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = DiagnosticConfig(min_freeze_ms=args.min_freeze_ms)
    reports = []
    for video in args.videos:
        try:
            report = analyze_video(video, config)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"错误: {video}: {exc}", file=sys.stderr)
            return 2
        reports.append(report)
        print(format_human(report))
    if args.json:
        payload = reports[0] if len(reports) == 1 else reports
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"JSON 证据已写入: {args.json}")
    return 0


def format_human(report: dict) -> str:
    video = report["video"]
    stamps = report["timestamps"]
    lines = [
        f"文件: {report['input']}",
        (
            f"视频: {video['width']}x{video['height']}, "
            f"{_fmt(video['fps'])} fps, {video['frame_count']} 帧, "
            f"{_seconds(video['duration_ms'])}"
        ),
        (
            f"时间戳: {stamps['continuity']}, 规则率 {stamps['regular_ratio']:.3f}, "
            f"间隔 {_fmt(stamps['nominal_delta_ms'])} ms, "
            f"空洞 {stamps['gap_count']}, 回退 {stamps['backstep_count']}, "
            f"缺失 {stamps['missing_count']}"
        ),
        f"判定: {_verdict_zh(report['verdict'])}",
        f"说明: {report['diagnosis']}",
    ]

    if report["freeze_intervals"]:
        lines.append("冻结区间:")
        for number, item in enumerate(report["freeze_intervals"], 1):
            lines.extend(_format_interval(number, item, "冻结"))
    else:
        lines.append("冻结区间: 未发现")

    if report["black_intervals"]:
        lines.append("不可判定区间:")
        for number, item in enumerate(report["black_intervals"], 1):
            lines.extend(_format_interval(number, item, "全黑"))
    lines.append(
        "全局证据: "
        f"重复帧占比 {report['metrics']['duplicate_fraction']:.3f}, "
        f"全黑占比 {report['metrics']['black_fraction']:.3f}, "
        f"有噪声帧占比 {report['metrics']['noise_above_floor_fraction']:.3f}, "
        f"局部噪声中位数 {_fmt(report['metrics']['median_local_noise'])}, "
        f"基线 {_fmt(report['metrics']['baseline_local_noise'])}"
    )
    return "\n".join(lines) + "\n"


def _format_interval(number: int, item: dict, label: str) -> list[str]:
    evidence = item["evidence"]
    lines = [
        (
            f"  #{number} {label} {_seconds(item['start_ms'])} → "
            f"{_seconds(item['end_ms'])}，持续 {_seconds(item['duration_ms'])}，"
            f"帧 {item['start_frame']}-{item['end_frame']}"
        ),
        (
            f"    重复 {evidence['duplicate_transitions']}/{max(evidence['frame_count'] - 1, 1)}"
            f" 次，局部噪声中位数 {_fmt(evidence['local_noise_median'])}，"
            f"阈值 {evidence['noise_floor']:.3f}，"
            f"相对基线 {_ratio(evidence['local_noise_ratio'])}"
        ),
        (
            f"    PTS {evidence['timestamp_continuity']}，"
            f"规则率 {evidence['timestamp_regular_ratio']:.3f}，"
            f"空洞 {evidence['timestamp_gap_count']}，"
            f"回退 {evidence['timestamp_backstep_count']}；"
            f"原因 {', '.join(evidence['reason_codes'])}"
        ),
    ]
    if item["status"] == "blackout_indeterminate":
        lines.append(
            "    边界: 全黑且无可测局部噪声，只能证明 PTS 在推进，"
            "不能证明图像内容仍在实时采集。"
        )
    return lines


def _verdict_zh(verdict: str) -> str:
    return {
        "frozen_repeat_detected": "检测到重复帧冻结",
        "indeterminate_blackout": "全黑，冻结不可判定",
        "mixed_frozen_and_indeterminate_black": "冻结与全黑不可判定区间并存",
        "no_freeze_detected": "未检测到冻结",
    }.get(verdict, verdict)


def _seconds(value: float | None) -> str:
    return "N/A" if value is None else f"{value / 1000.0:.3f}s"


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _ratio(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}x"
