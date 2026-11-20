"""合成替代音：每 1 秒一个“可辨识测试词”（线性扫频啁啾）。

真实对白被两类可辨识信号替代，全部由程序生成、无需外部素材：

1. 测试词 W0..W9：参数化的线性扫频啁啾，起始频率随词号变化，
   听感上是一串音高不同的“哔呜”词，且在频谱/相关分析中唯一可辨。
2. 语言代号：每个词前用 DTMF 双音多频短音拼出两位语言代号
   （如 CN / EN / JA），便于自动识别声轨语言标记。

每个整数秒还带一个 2 kHz 的弱时间码鸣点，方便人工与波形对齐。
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000

# DTMF 行/列频率（标准电话键盘布局）
_DTMF_LOW = (697.0, 770.0, 852.0, 941.0)
_DTMF_HIGH = (1209.0, 1336.0, 1477.0, 1633.0)
_KEYS = {
    "1": (0, 0), "2": (0, 1), "3": (0, 2), "A": (0, 3),
    "4": (1, 0), "5": (1, 1), "6": (1, 2), "B": (1, 3),
    "7": (2, 0), "8": (2, 1), "9": (2, 2), "C": (2, 3),
    "*": (3, 0), "0": (3, 1), "#": (3, 2), "D": (3, 3),
}

# 语言代号 -> 两位 DTMF 编码（数字键，便于检测，保证两两唯一）
LANG_CODES = {
    "CN": "26",
    "EN": "36",
    "JA": "52",
    "FR": "37",
    "DE": "39",
    "ES": "73",
}

WORD_COUNT = 10
CODE_DURATION = 0.06       # 单个 DTMF 音时长（秒）
CODE_GAP = 0.02
WORD_DURATION = 0.62       # 测试词啁啾时长
WORD_GAP = 0.18
BEEP_DURATION = 0.03
BEEP_FREQ = 2000.0


def _envelope(length: int, attack: int, release: int) -> np.ndarray:
    """线性起音/释音包络，消除拼接爆音。"""
    env = np.ones(length, dtype=np.float64)
    if attack > 0:
        env[:attack] = np.linspace(0.0, 1.0, attack)
    if release > 0:
        env[-release:] = np.linspace(1.0, 0.0, release)
    return env


def _sine(freq: float, duration: float, sr: int = SAMPLE_RATE,
          amplitude: float = 0.5) -> np.ndarray:
    n = int(round(duration * sr))
    t = np.arange(n, dtype=np.float64) / sr
    return amplitude * np.sin(2.0 * math.pi * freq * t)


def dtmf_tone(key: str, duration: float = CODE_DURATION,
              sr: int = SAMPLE_RATE) -> np.ndarray:
    """生成单个 DTMF 双音。"""
    if key not in _KEYS:
        raise ValueError(f"未知 DTMF 键: {key!r}")
    row, col = _KEYS[key]
    n = int(round(duration * sr))
    t = np.arange(n, dtype=np.float64) / sr
    sig = (np.sin(2 * math.pi * _DTMF_LOW[row] * t)
           + np.sin(2 * math.pi * _DTMF_HIGH[col] * t)) * 0.28
    return sig * _envelope(n, max(1, n // 40), max(1, n // 20))


def word_chirp(word_id: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    """生成测试词 W{word_id} 的线性扫频啁啾。

    起始频率 300 + 90*word_id Hz，结束频率固定 1500 Hz，
    不同词号在频谱上清晰可分，且整段波形可作为互相关模板。
    """
    if not 0 <= word_id < WORD_COUNT:
        raise ValueError(f"word_id 需在 0..{WORD_COUNT - 1} 之间")
    n = int(round(WORD_DURATION * sr))
    t = np.arange(n, dtype=np.float64) / sr
    f0 = 300.0 + 90.0 * word_id
    f1 = 1500.0
    phase = 2.0 * math.pi * (f0 * t + 0.5 * (f1 - f0) * t * t / WORD_DURATION)
    sig = 0.55 * np.sin(phase)
    return sig * _envelope(n, max(1, n // 50), max(1, n // 25))


@dataclass
class Slot:
    """一秒槽位内各成分的采样区间（相对槽位起点）。"""
    code_intervals: list[tuple[int, int, str]]
    word_interval: tuple[int, int]
    word_id: int
    beep_interval: tuple[int, int]


def build_slot(word_id: int, lang: str, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, Slot]:
    """构建 1 秒槽位：DTMF 语言代号 + 时间码鸣点 + 测试词。"""
    slot = np.zeros(sr, dtype=np.float64)
    cursor = int(0.04 * sr)
    intervals: list[tuple[int, int, str]] = []
    for key in LANG_CODES[lang]:
        tone = dtmf_tone(key, sr=sr)
        end = cursor + len(tone)
        slot[cursor:end] += tone
        intervals.append((cursor, end, key))
        cursor = end + int(CODE_GAP * sr)

    beep = _sine(BEEP_FREQ, BEEP_DURATION, sr, amplitude=0.18)
    bstart = int(0.40 * sr)
    slot[bstart:bstart + len(beep)] += beep

    chirp = word_chirp(word_id, sr)
    wstart = int(0.46 * sr)
    wend = min(sr, wstart + len(chirp))
    slot[wstart:wend] += chirp[:wend - wstart]
    meta = Slot(intervals, (wstart, wend), word_id,
                (bstart, bstart + len(beep)))
    return slot, meta


def build_track(seconds: int, lang: str, word_pattern=None,
                sr: int = SAMPLE_RATE, seed: int = 0) -> np.ndarray:
    """生成整轨替代音。

    word_pattern 为 None 时按 0..9 循环铺测试词；也可传入
    与秒等长的序列，用于制造重复/缺词等测试场景。
    """
    rng = np.random.default_rng(seed)
    if word_pattern is None:
        word_pattern = [i % WORD_COUNT for i in range(seconds)]
    if len(word_pattern) != seconds:
        raise ValueError("word_pattern 长度必须等于 seconds")
    track = np.zeros(seconds * sr, dtype=np.float64)
    for sec, word_id in enumerate(word_pattern):
        slot, _ = build_slot(int(word_id), lang, sr)
        track[sec * sr:(sec + 1) * sr] = slot
    # 低电平底噪，模拟现场环境并避免完全静音段
    track += rng.normal(0.0, 0.006, size=track.shape)
    return track


def save_wav(path: str, signal: np.ndarray, sr: int = SAMPLE_RATE,
             mono: bool = True) -> None:
    """写 16-bit PCM WAV。mono=False 时复制为立体声（主混音用）。"""
    pcm = np.clip(signal, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).round().astype("<i2")
    channels = 1 if mono else 2
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        if channels == 2:
            pcm16 = np.repeat(pcm16[:, None], 2, axis=1).reshape(-1)
        wf.writeframes(pcm16.tobytes())
