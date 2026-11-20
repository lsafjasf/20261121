"""Command-line interface for generation, analysis, and the bundled demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analyze import AnalysisConfig, analyze
from .report import write_report
from .synth import GenerationConfig, generate_scenario
from .wavio import WavError


def _labels(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",")]


def _offsets(value: str) -> list[float]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if len(parts) != 8:
        raise argparse.ArgumentTypeError("must provide exactly eight millisecond values")
    try:
        values = [float(item) for item in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if any(value < 0.0 for value in values):
        raise argparse.ArgumentTypeError("offsets must not be negative")
    return values


def _analysis_config(args: argparse.Namespace) -> AnalysisConfig:
    return AnalysisConfig(
        accept_threshold=args.accept_threshold,
        open_threshold=args.open_threshold,
        margin_threshold=args.margin_threshold,
        min_prominence_db=args.min_prominence_db,
        min_overlap=args.min_overlap,
        max_lag_seconds=args.max_lag_seconds,
        duplicate_ratio=args.duplicate_ratio,
        duplicate_similarity=args.duplicate_similarity,
    )


def _print_result(report: dict, paths: dict[str, Path]) -> None:
    summary = report["summary"]
    print(
        "分析完成："
        f"正常 {summary['ok']}，反相 {summary['inverted']}，"
        f"重复 {summary['duplicate']}，断路 {summary['open']}，"
        f"待复核 {summary['review']}"
    )
    print(f"人工复核：{'需要' if summary['manual_review_required'] else '不需要'}")
    for item in report["manual_review"]:
        print(f"  - {item['return_channel']}: {item['reason']}")
    print(f"JSON: {paths['json']}")
    print(f"Markdown: {paths['markdown']}")
    print(f"证据 CSV: {paths['evidence_csv']}")


def _run_analysis(
    reference: str | Path,
    recording: str | Path,
    output: str | Path,
    config: AnalysisConfig,
    reference_labels: list[str] | None = None,
    return_labels: list[str] | None = None,
    quiet: bool = False,
) -> tuple[dict, dict[str, Path]]:
    report = analyze(
        reference,
        recording,
        config=config,
        reference_labels=reference_labels,
        return_labels=return_labels,
    )
    paths = write_report(report, output)
    if not quiet:
        _print_result(report, paths)
    return report, paths


def _add_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--accept-threshold", type=float, default=0.28)
    parser.add_argument("--open-threshold", type=float, default=0.12)
    parser.add_argument("--margin-threshold", type=float, default=0.030)
    parser.add_argument("--min-prominence-db", type=float, default=1.5)
    parser.add_argument("--min-overlap", type=float, default=0.30)
    parser.add_argument("--max-lag-seconds", type=float, default=1.5)
    parser.add_argument("--duplicate-ratio", type=float, default=0.72)
    parser.add_argument("--duplicate-similarity", type=float, default=0.35)
    parser.add_argument(
        "--fail-on-review",
        action="store_true",
        help="return exit status 3 when any channel requires manual review",
    )


def _add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--duration", type=float, default=3.2)
    parser.add_argument("--reverb", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--snr-db", type=float, default=32.0)
    parser.add_argument(
        "--start-offsets-ms",
        type=_offsets,
        default=[0.0, 34.0, 91.0, 17.0, 123.0, 58.0, 7.0, 76.0],
        help="comma-separated start offset for each of the eight returns",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="venue-wiring",
        description="离线八路回录：测试源映射、反相、重复接线与断路识别",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="生成带真值的合成场馆数据")
    generate.add_argument("--output", default="demo")
    _add_generation_arguments(generate)

    check = commands.add_parser("analyze", help="分析已有 reference/recording WAV")
    check.add_argument("--reference", required=True, help="八路已知测试源 WAV")
    check.add_argument("--recording", required=True, help="八路回录 WAV")
    check.add_argument("--output", default="wiring-report")
    check.add_argument("--reference-labels", help="逗号分隔的八个源通道名")
    check.add_argument("--return-labels", help="逗号分隔的八个回录通道名")
    _add_analysis_arguments(check)

    demo = commands.add_parser("demo", help="生成混合故障场景并立即分析")
    demo.add_argument("--output", default="demo")
    _add_generation_arguments(demo)
    demo.add_argument("--report-dir", default=None)
    _add_analysis_arguments(demo)
    return parser


def _generation_config(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(
        sample_rate=args.sample_rate,
        duration_seconds=args.duration,
        reverb_seconds=args.reverb,
        seed=args.seed,
        snr_db=args.snr_db,
        start_offsets_ms=args.start_offsets_ms,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            truth = generate_scenario(args.output, _generation_config(args))
            print(f"已生成：{Path(args.output).resolve()}")
            print(
                f"源 {len(truth['source_labels'])} 路，回录 "
                f"{len(truth['return_labels'])} 路，"
                f"重复组 {len(truth['duplicate_groups'])}，"
                f"断路 {len(truth['open_channels'])}"
            )
            return 0

        if args.command == "analyze":
            report, paths = _run_analysis(
                args.reference,
                args.recording,
                args.output,
                _analysis_config(args),
                _labels(args.reference_labels),
                _labels(args.return_labels),
            )
        else:
            truth = generate_scenario(args.output, _generation_config(args))
            report_dir = args.report_dir or str(Path(args.output) / "report")
            report, paths = _run_analysis(
                Path(args.output) / truth["reference_file"],
                Path(args.output) / truth["recording_file"],
                report_dir,
                _analysis_config(args),
                truth["source_labels"],
                truth["return_labels"],
            )
        if args.fail_on_review and report["summary"]["manual_review_required"]:
            return 3
        return 0
    except (ValueError, OSError, WavError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
