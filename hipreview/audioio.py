"""通过 ffmpeg 把任意声轨解码为单声道 float64，并生成波形缩略。"""

from __future__ import annotations

import json
import shutil
import subprocess
import wave

import numpy as np

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


def probe_duration(path: str) -> float:
    """用 ffprobe 取时长（秒）。"""
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "json", path],
        check=True, capture_output=True, text=True,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


def decode_mono(path: str, sr: int = 16000) -> np.ndarray:
    """解码并下混为单声道、重采样到 sr，返回 float64 数组。"""
    cmd = [
        FFMPEG, "-v", "error", "-i", path,
        "-ac", "1", "-ar", str(sr), "-f", "f32le", "-",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True)
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float64)


def read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    """直接读 PCM WAV（合成素材用，免 ffmpeg）。多声道取均值下混。"""
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sr = wf.getframerate()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError(f"仅支持 16-bit PCM WAV，收到 sampwidth={width}")
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm, sr


def resample_linear(signal: np.ndarray, src_sr: int,
                    dst_sr: int) -> np.ndarray:
    """轻量线性重采样（供本机生成的 WAV 对齐分析率）。"""
    if src_sr == dst_sr:
        return signal
    duration = len(signal) / src_sr
    dst_n = int(round(duration * dst_sr))
    src_idx = np.linspace(0.0, len(signal) - 1.0, dst_n)
    lo = np.floor(src_idx).astype(int)
    frac = (src_idx - lo).astype(np.float64)
    hi = np.minimum(lo + 1, len(signal) - 1)
    return signal[lo] * (1.0 - frac) + signal[hi] * frac


def waveform_peaks(signal: np.ndarray, bins: int = 1000) -> dict:
    """把波形压缩成 bins 个 min/max 对，供前端画布绘制。"""
    bins = max(1, min(bins, len(signal)))
    trimmed = signal
    if len(trimmed) < bins:
        padded = np.zeros(bins)
        padded[:len(trimmed)] = trimmed
        trimmed = padded
    trimmed = trimmed[:len(trimmed) - len(trimmed) % bins]
    frames = trimmed.reshape(bins, -1)
    return {
        "min": np.min(frames, axis=1).round(4).tolist(),
        "max": np.max(frames, axis=1).round(4).tolist(),
        "rms": np.sqrt(np.mean(frames * frames, axis=1)).round(4).tolist(),
        "bins": bins,
    }


def slice_segment(signal: np.ndarray, sr: int, start: float,
                  end: float) -> np.ndarray:
    """按相对时刻（秒）切片段并做静音保护。"""
    a = max(0, int(round(start * sr)))
    b = min(len(signal), int(round(end * sr)))
    return signal[a:b]


def export_wav(path: str, signal: np.ndarray, sr: int = 16000) -> None:
    """把片段导出为审片用 WAV。"""
    pcm16 = (np.clip(signal, -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
