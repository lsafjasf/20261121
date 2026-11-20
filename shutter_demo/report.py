"""把实验配置与量化测量写成 JSON 与人类可读的文本摘要。"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from .contact import ContactRow
from .metrics import count_duplicates, summarize_frames


@dataclass
class Experiment:
    """一组接触表行及其元信息。"""

    name: str
    rows: list[ContactRow]
    source_description: dict


def build_report(experiment: Experiment) -> dict:
    """生成可 JSON 序列化的报告字典。"""
    report = {
        "experiment": experiment.name,
        "source": experiment.source_description,
        "rows": [],
    }
    for row in experiment.rows:
        result = row.result
        energies = [frame.energy for frame in result.frames]
        energy_array = np.asarray(energies, dtype=np.float64)
        report["rows"].append(
            {
                "title": row.title,
                "shutter": row.shutter.describe(),
                "schedule": {
                    "mode": result.mode,
                    "samples_per_second": result.samples_per_second,
                    "frame_count": len(result.frames),
                },
                "duplicate_frames": count_duplicates(result.frames),
                "energy_stats": {
                    "min": float(np.min(energy_array)) if energies else 0.0,
                    "max": float(np.max(energy_array)) if energies else 0.0,
                    "mean": float(np.mean(energy_array)) if energies else 0.0,
                },
                "frames": summarize_frames(result),
            }
        )
    return report


def write_json(report: dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return path


def write_text_summary(report: dict, path: str) -> str:
    """生成面向剪辑师的纯文本摘要，突出重心、能量、边界与重复采样。"""
    lines = [
        f"实验: {report['experiment']}",
        "=" * 72,
        "说明: t* 为曝光时间重心；E 为帧平均线性光能量；",
        "      红框=重复采样，黄框=能量明显低于本行峰值（闪光漏捕/部分捕获）。",
        "",
    ]
    for row in report["rows"]:
        shutter = row["shutter"]
        lines.append(
            f"[{row['title']}] {shutter['waveform']} 快门角={shutter['shutter_angle_deg']:.0f}° "
            f"duty={shutter['duty']:.3f} 偏移={shutter['phase_offset_ratio']:.2f} "
            f"模式={row['schedule']['mode']}"
        )
        stats = row["energy_stats"]
        lines.append(
            f"  能量: min={stats['min']:.4f} max={stats['max']:.4f} "
            f"mean={stats['mean']:.4f} | 重复采样帧={row['duplicate_frames']}"
        )
        for frame in row["frames"]:
            dup = (
                f" 重复自#{frame['duplicate_of']}"
                if frame["duplicate_of"] is not None
                else ""
            )
            lines.append(
                "  #{index:02d} {label:<6} 窗=[{ws:.4f},{we:.4f}] "
                "t*={ct:.4f} E={en:.4f} 重心x={cx:6.1f}px 边沿宽={ew:4.1f}px{dup}".format(
                    index=frame["index"],
                    label=frame["label"],
                    ws=frame["window_start_s"],
                    we=frame["window_end_s"],
                    ct=frame["exposure_centroid_s"],
                    en=frame["energy_mean_linear"],
                    cx=frame["spatial_centroid_x_px"],
                    ew=(
                        frame["edge_width_px"]
                        if not np.isnan(frame["edge_width_px"])
                        else -1.0
                    ),
                    dup=dup,
                )
            )
        lines.append("")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path
