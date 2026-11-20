#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flashrgb.render import render_sequence
from flashrgb.synth import generate_dataset
from flashrgb.sync import (
    FrameRecord,
    build_sync_report,
    decode_flash,
    discover_frames,
    group_exposures,
    parse_roi,
)
from flashrgb.imageio import read_gray


def _cmd_synth(args) -> int:
    metadata = generate_dataset(
        args.out,
        group_count=args.groups,
        width=args.width,
        height=args.height,
    )
    complete = sum(group["status"] == "complete" for group in metadata["groups"])
    print(
        f"Generated {len(metadata['groups'])} groups in {args.out} "
        f"({complete} complete, {len(metadata['groups']) - complete} incomplete)."
    )
    return 0


def _cmd_render(args) -> int:
    report = render_sequence(
        input_path=args.input,
        output_dir=args.output,
        flash_roi_value=args.flash_roi,
        search_radius=args.search_radius,
        fps=args.fps,
        truth_path=args.truth,
        compensate=not args.no_compensation,
    )
    metrics = report["edge_metrics"]
    print(
        f"Rendered {report['group_count']} groups to {report['output']} "
        f"({report['complete_groups']} complete, "
        f"{report['groups_missing_two_exposures']} missing-two)."
    )
    if metrics.get("evaluated_group_count"):
        print(
            "Visible-edge chroma error: "
            f"{metrics['before_mean']:.3f} -> {metrics['after_mean']:.3f} "
            f"({metrics['improvement_percent']:.2f}% reduction)."
        )
    return 0


def _cmd_inspect(args) -> int:
    paths = discover_frames(args.input)
    images = [read_gray(path) for path in paths]
    width = images[0].width
    height = images[0].height
    roi = parse_roi(args.flash_roi, width, height)
    records = []
    for order, (path, image) in enumerate(zip(paths, images)):
        channel, confidence, means = decode_flash(image, roi)
        records.append(FrameRecord(path, order, channel, confidence, means))
    groups = group_exposures(records)
    report = build_sync_report(records, groups, roi)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        for group in groups:
            files = ", ".join(
                f"{channel}={record.path.name}"
                for channel, record in sorted(group.frames.items())
            )
            print(f"G{group.group_id:03d} {group.status:18s} {files}")
        undecoded = sum(record.channel is None for record in records)
        print(f"frames={len(records)} groups={len(groups)} undecoded={undecoded}")
    return 0


def _cmd_demo(args) -> int:
    data = Path(args.data)
    output = Path(args.out)
    generate_dataset(data, group_count=args.groups, width=args.width, height=args.height)
    report = render_sequence(
        input_path=str(data / "frames"),
        output_dir=output,
        flash_roi_value=None,
        search_radius=args.search_radius,
        fps=args.fps,
        truth_path=str(data / "truth.json"),
        compensate=not args.no_compensation,
    )
    metrics = report["edge_metrics"]
    print(
        f"Demo data: {data}\nDemo output: {output}\n"
        f"Groups: {report['group_count']} total, "
        f"{report['complete_groups']} complete, "
        f"{report['groups_missing_two_exposures']} missing two exposures."
    )
    if metrics.get("evaluated_group_count"):
        print(
            f"Edge chroma error: {metrics['before_mean']:.3f} -> "
            f"{metrics['after_mean']:.3f} "
            f"({metrics['improvement_percent']:.2f}% reduction)"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct synchronized RGB video from mono frames captured "
            "under sequential red/green/blue illumination."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    synth = subparsers.add_parser("synth", help="generate a known-truth test dataset")
    synth.add_argument("--out", required=True, help="dataset directory")
    synth.add_argument("--groups", type=int, default=12)
    synth.add_argument("--width", type=int, default=320)
    synth.add_argument("--height", type=int, default=240)
    synth.set_defaults(func=_cmd_synth)

    render = subparsers.add_parser("render", help="decode frames and render outputs")
    render.add_argument("--input", required=True, help="frame directory or glob")
    render.add_argument("--output", required=True, help="output directory")
    render.add_argument(
        "--flash-roi",
        default=None,
        help="X,Y,WIDTH,HEIGHT; defaults to auto or truth.json",
    )
    render.add_argument("--search-radius", type=int, default=16)
    render.add_argument("--fps", type=float, default=4.0)
    render.add_argument("--truth", default=None, help="truth.json for edge metrics")
    render.add_argument(
        "--no-compensation",
        action="store_true",
        help="skip intra-group motion compensation",
    )
    render.set_defaults(func=_cmd_render)

    inspect = subparsers.add_parser("inspect", help="show decoded channels and groups")
    inspect.add_argument("--input", required=True, help="frame directory or glob")
    inspect.add_argument("--flash-roi", default=None)
    inspect.add_argument("--json", action="store_true")
    inspect.set_defaults(func=_cmd_inspect)

    demo = subparsers.add_parser("demo", help="generate test data and render it")
    demo.add_argument("--data", default="data/synthetic")
    demo.add_argument("--out", default="output")
    demo.add_argument("--groups", type=int, default=12)
    demo.add_argument("--width", type=int, default=320)
    demo.add_argument("--height", type=int, default=240)
    demo.add_argument("--search-radius", type=int, default=16)
    demo.add_argument("--fps", type=float, default=4.0)
    demo.add_argument("--no-compensation", action="store_true")
    demo.set_defaults(func=_cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
