"""离线快门积分演示的命令行入口。

示例
----
运行全部预置实验（移动条 / 闪光 / VFR / 线性光混合）::

    python -m shutter_demo.cli demo --out artifacts/run

用自定义快门时间函数跑移动条::

    python -m shutter_demo.cli custom --source bars --angle 200 \\
        --waveform-file examples/my_shutter.py --out artifacts/custom
"""
from __future__ import annotations

import argparse
import sys

from .contact import ContactRow, render_contact_sheet
from .pipeline import run_all_demos
from .report import Experiment, build_report, write_json, write_text_summary
from .shutter import Shutter, load_custom_waveform, make_shutter
from .sources import FlashField, MovingBars
from .vfr import cfr_schedule, parse_pts_text, pulldown_2_1_schedule
from .integrator import integrate_schedule


def _build_source(name: str, args: argparse.Namespace):
    if name == "bars":
        return MovingBars(
            width=args.width,
            height=args.height,
            speed_px_s=args.speed,
            bar_period_px=args.bar_period,
        )
    if name == "flash":
        return FlashField(
            width=args.width,
            height=args.height,
            flash_times_s=tuple(parse_pts_text(args.flash_times)),
            flash_duration_s=args.flash_duration,
            flash_radiance=args.flash_radiance,
        )
    raise ValueError(f"未知光源 {name}")


def _build_schedule(args: argparse.Namespace):
    if args.pts:
        from .vfr import explicit_schedule

        return explicit_schedule(parse_pts_text(args.pts))
    if args.pulldown:
        return pulldown_2_1_schedule(args.fps, cycle_count=max(args.frames // 3, 1))
    return cfr_schedule(args.fps, args.frames)


def _resolve_shutter(args: argparse.Namespace) -> Shutter:
    if args.waveform_file:
        waveform = load_custom_waveform(args.waveform_file, args.waveform_func)
        return Shutter(
            shutter_angle_deg=args.angle,
            phase_offset_ratio=args.phase_offset,
            waveform_name=f"custom:{args.waveform_file}",
            waveform=waveform,
        )
    return make_shutter(args.angle, args.waveform, args.phase_offset)


def run_custom(args: argparse.Namespace) -> str:
    source = _build_source(args.source, args)
    schedule = _build_schedule(args)
    shutter = _resolve_shutter(args)
    result = integrate_schedule(source, shutter, schedule, args.sample_rate)
    row = ContactRow(title=args.title, result=result, shutter=shutter)
    experiment = Experiment(
        name=args.title,
        rows=[row],
        source_description={"type": args.source},
    )
    sheet = render_contact_sheet([row], args.out + "/contact_sheet.png", title=args.title)
    report = build_report(experiment)
    write_json(report, args.out + "/report.json")
    write_text_summary(report, args.out + "/report.txt")
    return sheet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shutter_demo",
        description="高帧率输入按快门角/开合曲线积分的离线演示（VFR + 线性光混合）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="运行全部预置实验")
    demo.add_argument("--out", default="artifacts/demo", help="输出目录")

    custom = sub.add_parser("custom", help="单组自定义实验")
    custom.add_argument("--source", choices=["bars", "flash"], default="bars")
    custom.add_argument("--out", default="artifacts/custom")
    custom.add_argument("--title", default="自定义快门实验")
    custom.add_argument("--angle", type=float, default=180.0, help="快门角（度，0..360）")
    custom.add_argument(
        "--waveform",
        choices=["rect", "trapezoid", "gaussian", "double_slot"],
        default="rect",
    )
    custom.add_argument("--waveform-file", help="自定义快门时间函数的 Python 文件")
    custom.add_argument("--waveform-func", default="shutter", help="文件内函数名")
    custom.add_argument("--phase-offset", type=float, default=0.0, help="窗口相位偏移 0..1")
    custom.add_argument("--fps", type=float, default=24.0)
    custom.add_argument("--frames", type=int, default=8)
    custom.add_argument("--pts", help="逗号分隔的显式 PTS（秒），给出后覆盖 --fps/--frames")
    custom.add_argument("--pulldown", action="store_true", help="使用 2:1 VFR 调度")
    custom.add_argument("--sample-rate", type=float, default=40000.0, help="积分采样率 Hz")
    custom.add_argument("--width", type=int, default=160)
    custom.add_argument("--height", type=int, default=90)
    custom.add_argument("--speed", type=float, default=160.0, help="移动条速度 px/s")
    custom.add_argument("--bar-period", type=float, default=48.0)
    custom.add_argument("--flash-times", default="0.1,0.3,0.5")
    custom.add_argument("--flash-duration", type=float, default=0.001)
    custom.add_argument("--flash-radiance", type=float, default=4.0)
    custom.set_defaults(func=run_custom)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "demo":
        experiments = run_all_demos(args.out)
        print(f"完成 {len(experiments)} 组实验，输出目录: {args.out}")
        for experiment in experiments:
            print(f"  - {experiment.name}: {len(experiment.rows)} 行配置")
        return 0
    sheet = args.func(args)
    print(f"接触表已生成: {sheet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
