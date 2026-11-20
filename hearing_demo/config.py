"""配置的导入/导出：听阈 JSON 载入、压缩配置 JSON/CSV 写出与回读。"""

from __future__ import annotations

import csv
import json
import os

from . import __version__
from .audiogram import (STD_FREQS, BAND_EDGES, BAND_NAMES,
                        interpolate_threshold, build_prescription)


def load_audiogram(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "left" not in data or "right" not in data:
        raise ValueError("听阈文件需包含 left/right 两个耳的测点")
    return data


def build_config(data: dict, crossovers=(500.0, 2000.0, 6000.0)) -> dict:
    ears = {}
    for side in ("left", "right"):
        thr, info = interpolate_threshold(data[side])
        params = build_prescription(thr)
        ears[side] = {
            "thresholds_dbhl": {str(f): round(thr[f], 1) for f in STD_FREQS},
            "point_status": {str(f): info[f] for f in STD_FREQS},
            "bands": params,
        }
    return {
        "schema": "hearing-demo-compressor/v1",
        "tool_version": __version__,
        "sample_rate_hz": 48000,
        "crossover_hz": list(crossovers),
        "band_edges_hz": list(BAND_EDGES),
        "band_names": list(BAND_NAMES),
        "crossover_type": "一阶互补分频树（LP+HP 恒等，四带全频带平坦重组）",
        "prescription_note": "简化线性处方（半增益 + 线性压缩比），仅用于原型检查，非临床验配",
        "level_calibration_note": "内部以 dBFS 表示；0 dBFS 对应设备满量程，HL->dBFS 映射需在固件标定后替换",
        "ears": ears,
    }


def save_config(cfg: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("schema") != "hearing-demo-compressor/v1":
        raise ValueError("配置 schema 不匹配")
    return cfg


def export_band_csv(cfg: dict, path: str):
    fields = ["ear", "band", "edges_hz", "hl_band_db", "insertion_gain_db",
              "ratio", "knee_dbfs", "ceiling_dbfs", "attack_ms", "release_ms"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for side in ("left", "right"):
            for i, p in enumerate(cfg["ears"][side]["bands"]):
                row = {"ear": side, "band": BAND_NAMES[i],
                       "edges_hz": f"{BAND_EDGES[i]}-{BAND_EDGES[i+1]}"}
                row.update(p)
                w.writerow(row)


SAMPLE_AUDIOGRAM = {
    "note": "示例听阈（dB HL），null 为空白测点；左耳 4k 缺失、右耳 250 缺失用于演示插值",
    "left": {
        "125": 25, "250": 30, "500": 35, "1000": 45,
        "2000": 55, "4000": None, "8000": 70
    },
    "right": {
        "125": 15, "250": None, "500": 20, "1000": 25,
        "2000": 40, "4000": 55, "8000": 60
    }
}

