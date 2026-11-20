"""命令行入口：离线生成配置、实测数据、演示音频与报告。"""

from __future__ import annotations

import argparse
import csv
import json
import os

from . import SAMPLE_RATE
from .config import (load_audiogram, build_config, save_config, load_config,
                     export_band_csv, SAMPLE_AUDIOGRAM)
from .signals import demo_program, sine
from .wavio import write_wav
from .measurements import (tone_crossover_measure, level_transition_measure,
                           recombination_error, demo_band_levels)
from .chain import ProcessingChain
from .report import build_report


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_timeline_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "rms_dbfs", "peak_dbfs"])
        w.writerows(rows)


def cmd_config(args):
    data = (load_audiogram(args.audiogram) if args.audiogram
            else SAMPLE_AUDIOGRAM)
    cfg = build_config(data)
    save_config(cfg, os.path.join(args.out, "compression_config.json"))
    export_band_csv(cfg, os.path.join(args.out, "compression_bands.csv"))
    print(f"[ok] 配置已写出: {args.out}/compression_config.json")
    return cfg


def _run_measurements(cfg):
    fs = cfg["sample_rate_hz"]
    xovers = tuple(cfg["crossover_hz"])
    demo = demo_program(fs)
    meas = {}
    for side in ("left", "right"):
        params = cfg["ears"][side]["bands"]
        probe = sine(1000, 1.0, fs, -20.0)
        chain = ProcessingChain(params, fs, xovers)
        meas[side] = {
            "tone_active": tone_crossover_measure(params, fs, xovers, False),
            "tone_bypass": tone_crossover_measure(params, fs, xovers, True),
            "recomb_error_db": round(recombination_error(probe, chain), 2),
            "transition": level_transition_measure(params, fs),
            "demo": {},
        }
        for mode, bypass in (("active", False), ("bypass", True)):
            out, summary, timeline = demo_band_levels(
                params, fs, xovers, demo, bypass)
            meas[side]["demo"][mode] = {
                "band_summary": summary, "timeline": timeline,
                "audio": out,
            }
    return meas, demo


def cmd_measure(args):
    cfg = load_config(os.path.join(args.out, "compression_config.json"))
    meas, demo = _run_measurements(cfg)
    serializable = json.loads(json.dumps(meas, default=lambda o: None))
    for side in meas:
        for mode in ("active", "bypass"):
            del serializable[side]["demo"][mode]["audio"]
            _write_timeline_csv(
                os.path.join(args.out, f"timeline_{side}_{mode}.csv"),
                meas[side]["demo"][mode]["timeline"])
        _write_timeline_csv(
            os.path.join(args.out, f"transition_gain_{side}.csv"),
            meas[side]["transition"]["gain_timeline"])
    _write_json(os.path.join(args.out, "measurements.json"), serializable)
    print(f"[ok] 实测数据已写出: {args.out}/measurements.json")
    return cfg, meas, demo


def cmd_export(args, cfg=None, meas=None, demo=None):
    if cfg is None:
        cfg = load_config(os.path.join(args.out, "compression_config.json"))
    if meas is None:
        meas, demo = _run_measurements(cfg)
    audio_dir = os.path.join(args.out, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    write_wav(os.path.join(audio_dir, "source_demo.wav"), demo)
    outputs = {
        "config_json": "compression_config.json",
        "bands_csv": "compression_bands.csv",
        "measurements_json": "measurements.json",
    }
    for side_cn, side in (("left", "left"), ("right", "right")):
        for mode in ("active", "bypass"):
            sig = meas[side]["demo"][mode]["audio"]
            name = f"demo_{side}_{mode}.wav"
            write_wav(os.path.join(audio_dir, name), sig)
            outputs[f"{side}_{mode}_wav"] = f"audio/{name}"
    outputs["source_wav"] = "audio/source_demo.wav"
    report = build_report(cfg, meas, outputs)
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[ok] 演示音频与报告已写出: {audio_dir}, {args.out}/report.md")


def cmd_all(args):
    os.makedirs(args.out, exist_ok=True)
    cfg = cmd_config(args)
    cfg, meas, demo = cmd_measure(args)
    cmd_export(args, cfg, meas, demo)


def main(argv=None):
    p = argparse.ArgumentParser(description="离线听力演示工具（零依赖）")
    sub = p.add_subparsers(dest="cmd", required=True)
    common = dict()
    for name in ("config", "measure", "export", "all"):
        sp = sub.add_parser(name)
        sp.add_argument("--audiogram", default=None,
                        help="听阈 JSON；缺省使用内置示例")
        sp.add_argument("--out", default="out", help="输出目录")
        sp.set_defaults(func=globals()[f"cmd_{name}"])
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

