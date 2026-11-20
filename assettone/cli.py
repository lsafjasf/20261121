"""Command line entry points for the offline marker recognizer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .detector import analyze_file
from .report import attach_ground_truth, generate_demo, save_json


def _print_summary(report: dict) -> None:
    summary = report["summary"]
    false_positives = summary["false_positives"]
    false_text = "—" if false_positives is None else str(false_positives)
    print(
        f"命中 {summary['hits']} | 候选 {summary['candidates']} | "
        f"误报 {false_text} | 漏检 {summary.get('missed', 0)}"
    )
    if summary.get("timing_mae_ms") is not None:
        print(
            f"时标偏差 MAE {summary['timing_mae_ms']:.1f} ms | "
            f"最大 {summary['timing_max_abs_ms']:.1f} ms"
        )
    for item in report["detections"]:
        offset = item.get("timing_offset_ms")
        suffix = f" | Δt {offset:+.1f} ms" if offset is not None else ""
        print(
            f"{item['marker_seconds']:8.3f}s  {item['kind']:9s} "
            f"{item['asset_label']:12s} sync={item['sync']['hamming']}/16 "
            f"crc={'OK' if item['data']['crc_ok'] else 'FAIL'}{suffix}"
        )


def command_demo(args: argparse.Namespace) -> int:
    report, manifest = generate_demo(args.root)
    _print_summary(report)
    print(f"音频: {Path(args.root) / 'samples' / 'museum_sample.wav'}")
    print(f"报告: {Path(args.root) / 'reports' / 'museum_sample_report.json'}")
    return 0


def command_analyze(args: argparse.Namespace) -> int:
    report = analyze_file(args.audio, Path(args.audio).name)
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        report = attach_ground_truth(report, manifest)
    if args.out:
        save_json(args.out, report)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_summary(report)
    return 0


def command_serve(args: argparse.Namespace) -> int:
    from .server import serve

    return serve(args.host, args.port, Path(args.root), regenerate=args.regenerate)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="assettone",
        description="离线编码提示音资产识别",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="生成馆藏音乐样例、检测并写出报告")
    demo.add_argument("--root", default=".", help="工程项目根目录")
    demo.set_defaults(func=command_demo)

    analyze = subparsers.add_parser("analyze", help="分析单个 WAV 录音")
    analyze.add_argument("audio")
    analyze.add_argument("--manifest", help="带真值时标的手工/样例清单")
    analyze.add_argument("--out", help="报告 JSON 输出路径")
    analyze.add_argument("--json", action="store_true", help="直接打印完整 JSON")
    analyze.set_defaults(func=command_analyze)

    serve_parser = subparsers.add_parser("serve", help="启动本地离线界面")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--root", default=".")
    serve_parser.add_argument("--regenerate", action="store_true")
    serve_parser.set_defaults(func=command_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, KeyError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
