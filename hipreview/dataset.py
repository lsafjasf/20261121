"""生成多卷播放清单测试素材，并在助听轨注入典型换片故障。

清单含 5 卷，参考语言均为 CN：
  R1 正常卷（同步、独立助听轨、语言正确）
  R2 延后卷（助听轨整体滞后 320 ms）
  R3 重复轨卷（助听轨直接复制主混音中心）
  R4 片段重复卷（助听轨内一段测试词序列重复）
  R5 语言标记错误卷（助听轨被换成 EN 代号）
"""

from __future__ import annotations

import json
import os

import numpy as np

from .synth import SAMPLE_RATE, WORD_COUNT, build_slot, build_track, save_wav

REEL_SECONDS = 16


def _shift(track: np.ndarray, shift_ms: float,
           sr: int = SAMPLE_RATE) -> np.ndarray:
    """整体平移：正值表示延后（前面补零，尾部截掉）。"""
    shift = int(round(shift_ms / 1000.0 * sr))
    out = np.zeros_like(track)
    if shift >= 0:
        out[shift:] = track[:len(track) - shift]
    else:
        out[:shift] = track[-shift:]
    return out


def generate(out_dir: str, sr: int = SAMPLE_RATE) -> dict:
    """生成全部卷的主声轨/助听轨 WAV 与 playlist.json。"""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(42)
    reels_meta = []

    # 全部使用非周期词序，避免词号循环造成天然高相关；
    # R4 的后 4 秒再显式复制第 4..8 秒，作为唯一注入的片段重复。
    patterns = {
        "R1": [2, 7, 4, 9, 1, 6, 3, 8, 0, 5, 2, 9, 4, 7, 1, 6],
        "R2": [5, 1, 8, 3, 9, 0, 4, 7, 2, 6, 5, 8, 1, 3, 9, 2],
        "R3": [8, 2, 6, 0, 4, 9, 3, 7, 1, 5, 8, 0, 6, 2, 9, 4],
        "R4": [3, 7, 1, 8, 4, 0, 6, 2, 9, 5, 3, 7, 4, 0, 6, 2],
        "R5": [1, 6, 9, 2, 7, 0, 5, 8, 3, 4, 1, 9, 6, 2, 0, 7],
    }

    for reel_id, pattern in patterns.items():
        # 主混音：中心对白内容（立体声下混素材），语言 CN
        main = build_track(REEL_SECONDS, "CN", pattern, sr,
                           seed=int(reel_id[1]))
        main_path = os.path.join(out_dir, f"{reel_id}_main.wav")
        save_wav(main_path, main, sr, mono=False)

        # 独立助听轨：对白增强轨与主混音是两条独立制作的轨，
        # 词序一致（语义同步）但每槽独立合成、增益不同，
        # 因此不会出现主混音被直接复制的高相关。
        hi = np.zeros_like(main)
        gains = rng.uniform(0.75, 1.05, size=REEL_SECONDS)
        for sec, word_id in enumerate(pattern):
            slot, _ = build_slot(int(word_id), "CN", sr)
            hi[sec * sr:(sec + 1) * sr] = slot * gains[sec]
        hi += rng.normal(0.0, 0.008, size=hi.shape)
        hi_lang = "CN"

        if reel_id == "R2":
            hi = _shift(hi, 320.0, sr)          # 延后 320 ms
        elif reel_id == "R3":
            hi = main.copy()                    # 直接复制主混音
        elif reel_id == "R4":
            # 把 4..8s 的四秒原样平移到 12..16s，保持槽内相位一致，
            # 模拟换片时重复挂了同一段助听素材
            hi[12 * sr:16 * sr] = hi[4 * sr:8 * sr]
        elif reel_id == "R5":
            hi = build_track(REEL_SECONDS, "EN", pattern, sr, seed=99)
            hi_lang = "EN"

        hi_path = os.path.join(out_dir, f"{reel_id}_hi.wav")
        save_wav(hi_path, hi, sr, mono=True)
        reels_meta.append({
            "reel_id": reel_id,
            "main": os.path.basename(main_path),
            "hearing": os.path.basename(hi_path),
            "expected_lang": "CN",
            "injected_lang": hi_lang,
            "duration_sec": REEL_SECONDS,
        })

    playlist = {
        "name": "助听轨审片示范清单",
        "sample_rate": sr,
        "reference": {
            "description": "参考片段：卷首 6 秒 DTMF 语言代号应为 CN",
            "expected_lang": "CN",
        },
        "reels": reels_meta,
    }
    with open(os.path.join(out_dir, "playlist.json"), "w",
              encoding="utf-8") as fh:
        json.dump(playlist, fh, ensure_ascii=False, indent=2)
    return playlist
