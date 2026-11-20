#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from audit_lib.analyze import analyze_project, load_manifest
from audit_lib.render import (
    render_csv,
    render_data,
    render_evidence,
    render_site,
    render_videos,
)


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="离线硬字幕局部对比审查：合成双版本素材并生成可跳转审查页。"
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "artifacts/demo/base.mp4",
        help="无字幕底片视频。",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="字幕段与已知掩模清单；默认读取底片旁的 manifest.json。",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts/demo",
        help="输出目录。",
    )
    parser.add_argument("--contrast", type=float, default=3.0)
    parser.add_argument("--coverage", type=float, default=0.82)
    parser.add_argument("--transient-max", type=float, default=1.0)
    parser.add_argument("--outline-width", type=int, default=3)
    return parser


def validate_thresholds(args: argparse.Namespace) -> None:
    if args.contrast <= 1:
        raise SystemExit("--contrast 必须大于 1。")
    if not 0 < args.coverage <= 1:
        raise SystemExit("--coverage 必须在 (0, 1] 范围内。")
    if args.transient_max < 0:
        raise SystemExit("--transient-max 不能为负数。")
    if args.outline_width < 0:
        raise SystemExit("--outline-width 不能为负数。")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_thresholds(args)
    base = args.base.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else base.parent / "manifest.json"
    )
    output_dir = args.out.expanduser().resolve()
    if not base.exists():
        parser.error(f"底片不存在：{base}")
    if not manifest_path.exists():
        parser.error(f"manifest 不存在：{manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    template_dir = ROOT / "tools/templates"
    thresholds = {
        "contrast": args.contrast,
        "coverage": args.coverage,
        "transient_max": args.transient_max,
        "outline_width": args.outline_width,
    }

    manifest, cues = load_manifest(manifest_path)
    report, metrics = analyze_project(base, manifest_path, thresholds)
    render_videos(
        base,
        manifest,
        cues,
        output_dir,
        {
            "fps": report["meta"]["fps"],
            "duration": report["meta"]["duration"],
        },
        args.outline_width,
    )
    render_evidence(report, output_dir)
    render_csv(metrics, output_dir / "metrics.csv")
    render_data(report, output_dir)
    render_site(
        report,
        manifest,
        cues,
        output_dir,
        template_dir,
        base,
    )
    summary = {
        "output": str(output_dir),
        "summary": report["summary"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
