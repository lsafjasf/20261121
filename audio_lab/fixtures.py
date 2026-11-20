"""Deterministic example signals for strong attacks and a real weak prelude."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .wavio import write_mono_wav

SAMPLE_RATE = 48_000
DURATION_S = 5.2
ONSETS_S = [0.85, 1.75, 2.65, 3.55, 4.45]
PRELUDE_S = 0.060
HIT_TAIL_S = 0.380


def _synthesize(real_prelude: bool) -> list[float]:
    rng = random.Random(20261121)
    raw: list[float] = []
    peak = 0.0
    frame_count = round(DURATION_S * SAMPLE_RATE)
    for frame in range(frame_count):
        time_s = frame / SAMPLE_RATE
        noise = rng.uniform(-1.0, 1.0)
        value = 0.0
        for onset in ONSETS_S:
            relative = time_s - onset
            if 0.0 <= relative < HIT_TAIL_S:
                value += (
                    0.82 * math.exp(-relative / 0.018)
                    + 0.18 * math.exp(-relative / 0.100)
                ) * noise
                value += 0.55 * math.exp(-relative / 0.140) * math.sin(
                    2.0 * math.pi * 62.0 * relative
                )
                value += 0.30 * math.exp(-relative / 0.012) * math.sin(
                    2.0 * math.pi * 1800.0 * relative
                )
                value += 0.50 * math.exp(-relative / 0.0015) * (
                    1.0 if noise >= 0.0 else -1.0
                )
            elif real_prelude and -PRELUDE_S <= relative < 0.0:
                lead = relative + PRELUDE_S
                rise = (lead / PRELUDE_S) ** 2
                value += 0.080 * rise * (
                    0.65 * math.sin(2.0 * math.pi * 2350.0 * lead)
                    + 0.35 * noise
                )
        raw.append(value)
        peak = max(peak, abs(value))
    scale = 0.89 / peak
    return [scale * value for value in raw]


def build_fixtures(destination: str | Path) -> dict[str, object]:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    strong = _synthesize(real_prelude=False)
    weak = _synthesize(real_prelude=True)
    write_mono_wav(destination / "strong_hits.wav", strong, SAMPLE_RATE)
    write_mono_wav(destination / "true_weak_prelude.wav", weak, SAMPLE_RATE)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "sample_rate": SAMPLE_RATE,
        "window_ms": 20.0,
        "guard_ms": 2.0,
        "description": (
            "同一组强瞬态敲击；strong_hits 在起音前为数字静音，"
            "true_weak_prelude 在每个起音前 60 ms 加入真实低电平前奏。"
        ),
        "cases": [
            {
                "id": "strong_hits",
                "title": "强敲击 / 静音前窗",
                "file": "strong_hits.wav",
                "onsets_s": ONSETS_S,
                "expect": "codec-added",
                "description": (
                    "测量窗口内参考源为数字静音；任何窗口能量都可归因于编码。"
                ),
            },
            {
                "id": "true_weak_prelude",
                "title": "强敲击 / 真实弱前奏",
                "file": "true_weak_prelude.wav",
                "onsets_s": ONSETS_S,
                "expect": "source-prelude",
                "description": (
                    "每个起音前有 60 ms 渐强低电平拾音；窗口内源能量真实存在，"
                    "验收时必须扣除后再判断编码新增。"
                ),
            },
        ],
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
