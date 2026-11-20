from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .ambiguity import catalog_layouts
from .beam import compute_beam_pattern, write_pattern_csv, write_pattern_svg
from .config import Scene, SceneError
from .demo import demo_scene
from .estimate import AnalysisError, analyze_recordings
from .report import write_markdown_report, write_results
from .simulate import simulate_scene


def _demo(args: argparse.Namespace) -> int:
    scene = Scene.load(args.scene) if args.scene else demo_scene()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    recordings = output / "recordings"
    truth = simulate_scene(scene, recordings)
    result = analyze_recordings(scene, recordings)
    pattern = compute_beam_pattern(scene, result)
    write_results(result, pattern, output, truth)
    write_pattern_csv(pattern, output / "beam_pattern.csv")
    write_pattern_svg(
        pattern,
        output / "beam_before_after.svg",
        "Four-microphone beam main lobe",
        f"{scene.name}: before vs after delay/gain/polarity calibration",
    )
    write_markdown_report(result, pattern, output / "report.md", truth)
    scene.save(output / "scene.json")
    print(f"status: {result.status}")
    print(f"map: {list(result.best_map)}")
    print(f"report: {output / 'report.md'}")
    print(f"beam: {output / 'beam_before_after.svg'}")
    return 0


def _simulate(args: argparse.Namespace) -> int:
    scene = Scene.load(args.scene)
    manifest = simulate_scene(scene, args.out)
    print(f"wrote {len(manifest['recordings'])} recording(s) to {args.out}")
    return 0


def _analyze(args: argparse.Namespace) -> int:
    scene = Scene.load(args.scene)
    result = analyze_recordings(scene, args.recordings)
    pattern = compute_beam_pattern(scene, result)
    output = Path(args.out)
    truth = None
    truth_path = Path(args.recordings) / "truth.json"
    if truth_path.exists():
        import json

        truth = json.loads(truth_path.read_text(encoding="utf-8"))
    write_results(result, pattern, output, truth)
    write_pattern_csv(pattern, output / "beam_pattern.csv")
    write_pattern_svg(
        pattern,
        output / "beam_before_after.svg",
        "Four-microphone beam main lobe",
        f"{scene.name}: before vs after delay/gain/polarity calibration",
    )
    write_markdown_report(result, pattern, output / "report.md", truth)
    print(f"status: {result.status}")
    print(f"map: {list(result.best_map)}")
    print(f"report: {output / 'report.md'}")
    return 0


def _audit(args: argparse.Namespace) -> int:
    lines = [
        "# 四麦克风布局可辨识性审计",
        "",
        "方向秩小于 3 表示声源方向未张满三维，阵列平面法向的镜像或离面几何"
        "存在额外盲区。",
        "",
        "| 布局 | 方向秩 | 等价排列 | 反射替代排列 | 唯一可判 |",
        "|---|---:|---:|---:|:---:|",
    ]
    items = catalog_layouts()
    for item in items:
        lines.append(
            f"| {item.name} | {item.source_rank} | "
            f"{len(item.equivalent_orders)} | {len(item.reflection_orders)} | "
            f"{'是' if item.identifiable else '否'} |"
        )
    lines.extend(["", "## 无法唯一判断的布局", ""])
    ambiguous = [item for item in items if not item.identifiable]
    if ambiguous:
        for item in ambiguous:
            reason = "；".join(item.notes) or "存在等价几何解"
            lines.append(f"- `{item.name}`：{reason}")
    else:
        lines.append("- 巡检集合中未发现不可辨识布局。")
    lines.extend(["", "## 全部备注", ""])
    for item in items:
        for note in item.notes:
            lines.append(f"- `{item.name}`：{note}")
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"audit: {args.out}")
    else:
        print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="miccal",
        description=(
            "Four-microphone swept-sine calibration with geometry-constrained "
            "direct-path selection and reflection rejection."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run the complete synthetic demo")
    demo.add_argument("--out", default="artifacts/demo", help="output directory")
    demo.add_argument("--scene", help="optional scene JSON instead of the built-in demo")
    demo.set_defaults(handler=_demo)
    simulate = subparsers.add_parser("simulate", help="write four-channel sweep WAVs")
    simulate.add_argument("scene")
    simulate.add_argument("--out", required=True)
    simulate.set_defaults(handler=_simulate)
    analyze = subparsers.add_parser("analyze", help="analyze four-channel sweep WAVs")
    analyze.add_argument("scene")
    analyze.add_argument("--recordings", required=True)
    analyze.add_argument("--out", default="artifacts/analysis")
    analyze.set_defaults(handler=_analyze)
    audit = subparsers.add_parser("audit", help="list non-identifiable layouts")
    audit.add_argument("--out")
    audit.set_defaults(handler=_audit)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (SceneError, AnalysisError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
