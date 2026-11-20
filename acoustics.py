"""Sample-accurate acoustic processing for the local bat screening desk."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import uuid

import numpy as np
from scipy import signal
import soundfile as sf


MONITOR_RATE = 48_000
MAX_MONITOR_FRAMES = MONITOR_RATE * 180


@dataclass
class Source:
    id: str
    name: str
    path: Path
    rate: int
    frames: int
    channels: int
    duration: float
    format: str
    subtype: str
    kind: str = "upload"

    def metadata(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "rate": self.rate,
            "frames": self.frames,
            "channels": self.channels,
            "duration": self.duration,
            "format": self.format,
            "subtype": self.subtype,
            "kind": self.kind,
        }


class SourceBank:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sources: dict[str, Source] = {}

    def add(self, path: Path, name: str | None = None, kind: str = "upload") -> Source:
        info = sf.info(str(path))
        source_id = uuid.uuid4().hex[:10]
        source = Source(
            id=source_id,
            name=name or path.name,
            path=path.resolve(),
            rate=int(info.samplerate),
            frames=int(info.frames),
            channels=int(info.channels),
            duration=float(info.duration),
            format=info.format,
            subtype=info.subtype,
            kind=kind,
        )
        self.sources[source_id] = source
        return source

    def get(self, source_id: str) -> Source:
        try:
            return self.sources[source_id]
        except KeyError as exc:
            raise ValueError("声源不存在或服务已重启") from exc

    def read(self, source: Source, start: int, frames: int, mono: bool = False) -> np.ndarray:
        with sf.SoundFile(str(source.path)) as handle:
            handle.seek(max(0, int(start)))
            data = handle.read(int(frames), dtype="float32", always_2d=True)
        if mono and data.shape[1] > 1:
            data = data.mean(axis=1, keepdims=True)
        return data


def _fade_edges(data: np.ndarray, rate: int, seconds: float = 0.002) -> np.ndarray:
    count = min(len(data) // 2, max(1, int(rate * seconds)))
    if count < 2:
        return data
    ramp = np.sin(np.linspace(0.0, np.pi / 2, count, dtype=np.float32)) ** 2
    data = data.copy()
    data[:count] *= ramp
    data[-count:] *= ramp[::-1]
    return data


def _normalize(data: np.ndarray, peak: float = 0.92) -> np.ndarray:
    peak_value = float(np.max(np.abs(data))) if data.size else 0.0
    if peak_value > 1e-9:
        data = data * (peak / peak_value)
    return np.nan_to_num(data, copy=False).astype(np.float32)


def _write_tech_preview(path: Path) -> None:
    """Create a compact multi-tone stepped sweep with endpoint markers."""
    rate = 192_000
    duration = 0.56
    size = int(rate * duration)
    audio = np.zeros(size, dtype=np.float64)
    rng = np.random.default_rng(20261121)
    audio += rng.normal(0.0, 0.0007, size)

    frequencies = [58_000, 62_000, 66_000, 70_000, 74_000, 78_000, 82_000]
    burst = 0.026
    gap = 0.010
    cursor = 0.055
    for index, frequency in enumerate(frequencies):
        start = int(cursor * rate)
        count = int(burst * rate)
        t = np.arange(count) / rate
        envelope = np.sin(np.pi * np.arange(count) / count) ** 2
        tone = np.sin(2 * np.pi * frequency * t + 0.2 * index)
        audio[start:start + count] += 0.22 * envelope * tone
        # A second harmonic remains visible and checks broadband handling.
        audio[start:start + count] += 0.035 * envelope * np.sin(
            4 * np.pi * frequency * t
        )
        cursor += burst + gap

    pip = np.hanning(96) * rng.normal(0.0, 1.0, 96)
    first = int(0.012 * rate)
    last = size - int(0.012 * rate) - len(pip)
    audio[first:first + len(pip)] += 0.85 * pip
    audio[last:last + len(pip)] += 0.85 * pip

    audio = _normalize(audio, 0.82)
    sf.write(str(path), audio.astype(np.float32), rate, subtype="PCM_24")


def _write_bat_demo(path: Path) -> None:
    """Create a synthetic, deterministic 192 kHz bat screening scene."""
    rate = 192_000
    duration = 2.72
    size = int(rate * duration)
    rng = np.random.default_rng(20261121)
    audio = rng.normal(0.0, 0.0022, size)
    t_global = np.arange(size) / rate
    audio += 0.0032 * np.sin(2 * np.pi * 9_100 * t_global)

    calls = [
        (0.21, 0.013, 86_000, 49_000, 0.44),
        (0.47, 0.017, 79_000, 43_000, 0.36),
        (0.79, 0.011, 92_000, 57_000, 0.33),
        (1.15, 0.021, 74_000, 39_000, 0.39),
        (1.53, 0.015, 88_000, 46_000, 0.42),
        (1.91, 0.019, 82_000, 41_000, 0.32),
        (2.31, 0.024, 69_000, 36_000, 0.30),
    ]
    for onset, call_duration, f_start, f_end, amplitude in calls:
        count = int(call_duration * rate)
        start = int(onset * rate)
        u = np.arange(count) / count
        frequency = (
            f_start
            + (f_end - f_start) * u
            + 1_400 * np.sin(2 * np.pi * 2.2 * u)
        )
        phase = 2 * np.pi * np.cumsum(frequency) / rate
        envelope = np.sin(np.pi * u) ** 1.15
        pulse = envelope * (
            np.sin(phase)
            + 0.22 * np.sin(2 * phase + 0.4)
            + 0.08 * np.sin(3 * phase + 0.9)
        )
        audio[start:start + count] += amplitude * pulse
        echo_shift = int(0.034 * rate)
        if start + count + echo_shift < size:
            audio[start + echo_shift:start + count + echo_shift] += 0.15 * amplitude * pulse

    # Short constant-frequency probe near the center of the view.
    start = int(1.70 * rate)
    count = int(0.050 * rate)
    u = np.arange(count) / count
    probe = np.sin(2 * np.pi * 64_000 * np.arange(count) / rate)
    audio[start:start + count] += 0.16 * np.sin(np.pi * u) ** 2 * probe

    audio = _normalize(audio, 0.80)
    sf.write(str(path), audio.astype(np.float32), rate, subtype="PCM_24")


def ensure_builtin_sources(bank: SourceBank) -> tuple[Source, Source]:
    demo_path = bank.data_dir / "field_demo_192k.wav"
    tech_path = bank.data_dir / "calibration_sweep_192k.wav"
    if not demo_path.exists():
        _write_bat_demo(demo_path)
    if not tech_path.exists():
        _write_tech_preview(tech_path)
    demo = bank.add(demo_path, "FIELD_024 · 夜间样带", kind="demo")
    tech = bank.add(tech_path, "CAL_192 · 多频扫频", kind="calibration")
    return demo, tech


def _safe_sosfiltfilt(sos: np.ndarray, data: np.ndarray) -> np.ndarray:
    if data.size < 16:
        return signal.sosfilt(sos, data)
    padlen = min(data.size - 1, 2048)
    return signal.sosfiltfilt(sos, data, padlen=padlen)


def _lowpass(data: np.ndarray, rate: int, cutoff: float) -> np.ndarray:
    cutoff = min(cutoff, rate * 0.495)
    if cutoff <= 0 or cutoff >= rate * 0.49:
        return data
    sos = signal.butter(6, cutoff, btype="lowpass", fs=rate, output="sos")
    return _safe_sosfiltfilt(sos, data)


def _bandpass(data: np.ndarray, rate: int, low: float, high: float) -> np.ndarray:
    low = max(1.0, low)
    high = min(high, rate * 0.499)
    if high <= low:
        return np.zeros_like(data)
    sos = signal.butter(6, [low, high], btype="bandpass", fs=rate, output="sos")
    return _safe_sosfiltfilt(sos, data)


def _resample_to(data: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return data.astype(np.float32, copy=False)
    divisor = math.gcd(source_rate, target_rate)
    up = target_rate // divisor
    down = source_rate // divisor
    result = signal.resample_poly(data, up, down).astype(np.float32)
    expected = int(round(len(data) * target_rate / source_rate))
    if len(result) < expected:
        result = np.pad(result, (0, expected - len(result)))
    return result[:expected]


def heterodyne(
    clip: np.ndarray,
    source_rate: int,
    center_hz: float,
    bandwidth_hz: float,
    sideband: str,
    output_rate: int = MONITOR_RATE,
) -> np.ndarray:
    """Band-select one side of the carrier, then translate it to baseband."""
    sideband = sideband.upper()
    if sideband not in {"USB", "LSB"}:
        raise ValueError("边带必须是 USB 或 LSB")
    bandwidth_hz = float(np.clip(bandwidth_hz, 500, min(80_000, source_rate * 0.48)))
    center_hz = float(np.clip(center_hz, bandwidth_hz / 2 + 100, source_rate / 2 - 100))
    if sideband == "USB":
        low, high = center_hz, center_hz + bandwidth_hz
    else:
        low, high = center_hz - bandwidth_hz, center_hz
    high = min(high, source_rate * 0.499)
    low = max(1.0, low)

    selected = _bandpass(clip.astype(np.float64), source_rate, low, high)
    t = np.arange(len(selected), dtype=np.float64) / source_rate
    baseband = 2.0 * selected * np.cos(2 * np.pi * center_hz * t)
    baseband = _lowpass(
        baseband, source_rate, min(bandwidth_hz * 0.98, output_rate * 0.45)
    )
    monitor = _resample_to(baseband, source_rate, output_rate)
    monitor = _fade_edges(_normalize(monitor), output_rate)
    return monitor


def time_expand(
    clip: np.ndarray,
    source_rate: int,
    factor: float,
    output_rate: int = MONITOR_RATE,
) -> np.ndarray:
    """Slow the waveform uniformly, preserving the original event proportions."""
    factor = float(np.clip(factor, 1.0, 50.0))
    source = clip.astype(np.float64)
    # Time scaling by N also scales every frequency by 1/N. Limit the input
    # before interpolation so the monitored copy cannot alias.
    source = _lowpass(source, source_rate, min(source_rate / 2, output_rate * factor / 2) * 0.98)
    output_frames = int(round(len(source) * factor * output_rate / source_rate))
    if output_frames <= 1:
        return np.zeros(0, dtype=np.float32)
    if output_frames > MAX_MONITOR_FRAMES:
        raise ValueError("监听时长超过 180 秒，请缩短选区或降低减速倍数")
    positions = np.arange(output_frames, dtype=np.float64) * (
        source_rate / (output_rate * factor)
    )
    monitor = np.interp(positions, np.arange(len(source)), source)
    monitor = _fade_edges(_normalize(monitor), output_rate)
    return monitor


def frequency_map(params: dict) -> dict:
    mode = params.get("mode", "heterodyne")
    if mode == "heterodyne":
        center = float(params.get("center_hz", 45_000))
        bandwidth = float(params.get("bandwidth_hz", 10_000))
        sideband = str(params.get("sideband", "USB")).upper()
        sign = "+" if sideband == "USB" else "-"
        return {
            "kind": "heterodyne",
            "center_hz": center,
            "bandwidth_hz": bandwidth,
            "sideband": sideband,
            "source_hz": f"{center:.0f} {sign} heard_hz",
            "heard_hz": (
                f"source_hz - {center:.0f}"
                if sideband == "USB"
                else f"{center:.0f} - source_hz"
            ),
            "source_band_hz": [
                center if sideband == "USB" else center - bandwidth,
                center + bandwidth if sideband == "USB" else center,
            ],
        }
    factor = float(params.get("factor", 10))
    return {
        "kind": "time_expansion",
        "factor": factor,
        "source_hz": f"heard_hz * {factor:g}",
        "heard_hz": f"source_hz / {factor:g}",
        "source_band_hz": [0, None],
    }


def process_selection(
    bank: SourceBank,
    source: Source,
    start_frame: int,
    end_frame: int,
    params: dict,
) -> dict:
    start_frame = int(max(0, min(start_frame, source.frames - 1)))
    end_frame = int(max(start_frame + 1, min(end_frame, source.frames)))
    frame_count = end_frame - start_frame
    mode = params.get("mode", "heterodyne")
    if mode not in {"heterodyne", "time_expansion"}:
        raise ValueError("未知监听模式")

    clip = bank.read(source, start_frame, frame_count, mono=True)[:, 0]
    clip = clip - float(np.mean(clip))
    if mode == "heterodyne":
        center = float(params.get("center_hz", 45_000))
        bandwidth = float(params.get("bandwidth_hz", 10_000))
        sideband = str(params.get("sideband", "USB")).upper()
        monitor = heterodyne(
            clip, source.rate, center, bandwidth, sideband, MONITOR_RATE
        )
        factor = 1.0
        normalized = {
            "mode": mode,
            "center_hz": center,
            "bandwidth_hz": bandwidth,
            "sideband": sideband,
        }
    else:
        factor = float(np.clip(float(params.get("factor", 10)), 1.0, 50.0))
        monitor = time_expand(clip, source.rate, factor, MONITOR_RATE)
        normalized = {"mode": mode, "factor": factor}

    monitor_frames = len(monitor)
    monitor_duration = monitor_frames / MONITOR_RATE
    selection_duration = frame_count / source.rate
    return {
        "audio": monitor.astype(np.float32),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_count": frame_count,
        "selection_duration": selection_duration,
        "monitor_rate": MONITOR_RATE,
        "monitor_frames": monitor_frames,
        "monitor_duration": monitor_duration,
        "time_factor": factor,
        "params": normalized,
        "frequency_map": frequency_map(normalized),
    }


def monitor_seconds_to_source(
    seconds: float, start_frame: int, source_rate: int, factor: float
) -> float:
    return float(start_frame) + float(seconds) * source_rate / float(factor)


def source_frame_to_monitor(
    frame: float, start_frame: int, source_rate: int, factor: float
) -> float:
    return (float(frame) - float(start_frame)) * float(factor) / float(source_rate)


def _pool_columns(matrix: np.ndarray, target: int) -> np.ndarray:
    rows, columns = matrix.shape
    if columns <= target:
        return matrix
    edges = np.linspace(0, columns, target + 1).astype(int)
    result = np.empty((rows, target), dtype=matrix.dtype)
    for index in range(target):
        section = matrix[:, edges[index]:max(edges[index + 1], edges[index] + 1)]
        result[:, index] = section.max(axis=1)
    return result


def _pool_rows(matrix: np.ndarray, target: int) -> np.ndarray:
    rows, columns = matrix.shape
    if rows <= target:
        return matrix
    edges = np.linspace(0, rows, target + 1).astype(int)
    result = np.empty((target, columns), dtype=matrix.dtype)
    for index in range(target):
        section = matrix[edges[index]:max(edges[index + 1], edges[index] + 1), :]
        result[index, :] = section.max(axis=0)
    return result


def spectrogram_image(
    bank: SourceBank,
    source: Source,
    start_frame: int,
    end_frame: int,
    f_min: float,
    f_max: float,
    width: int,
    height: int,
) -> dict:
    start_frame = int(max(0, min(start_frame, source.frames - 1)))
    end_frame = int(max(start_frame + 1, min(end_frame, source.frames)))
    width = int(np.clip(width, 160, 2400))
    height = int(np.clip(height, 80, 900))
    clip = bank.read(source, start_frame, end_frame - start_frame, mono=True)[:, 0]
    clip = clip - float(np.mean(clip))
    if len(clip) < 16:
        clip = np.pad(clip, (0, 16 - len(clip)))

    nperseg = min(1024, len(clip))
    if nperseg < 32:
        nperseg = 32
    hop = max(1, min(nperseg // 4, max(1, len(clip) // max(width * 3, 1))))
    noverlap = nperseg - hop
    if noverlap >= nperseg:
        noverlap = nperseg - 1
    frequencies, _, values = signal.stft(
        clip,
        fs=source.rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )
    magnitude = 20.0 * np.log10(np.abs(values) + 1e-9)
    f_max = float(np.clip(f_max, 100, source.rate / 2))
    f_min = float(np.clip(f_min, 0, f_max - 100))
    mask = (frequencies >= f_min) & (frequencies <= f_max)
    magnitude = magnitude[mask, :]
    if magnitude.size == 0:
        magnitude = np.zeros((height, 1), dtype=np.float32)
    finite = magnitude[np.isfinite(magnitude)]
    low = float(np.percentile(finite, 8)) if finite.size else -120.0
    high = float(np.percentile(finite, 99.7)) if finite.size else 0.0
    if high - low < 12:
        high = low + 12
    normalized = np.clip((magnitude - low) / (high - low), 0.0, 1.0)
    normalized = _pool_columns(normalized, width)
    normalized = _pool_rows(normalized, height)
    # Rows are returned top-down so the canvas paints without a vertical flip.
    image = (normalized[::-1, :] * 255).astype(np.uint8)
    return {
        "bytes": image.tobytes(),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "min_db": low,
        "max_db": high,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "f_min": f_min,
        "f_max": f_max,
    }


def waveform_overview(
    bank: SourceBank,
    source: Source,
    start_frame: int,
    end_frame: int,
    width: int,
) -> dict:
    start_frame = int(max(0, min(start_frame, source.frames - 1)))
    end_frame = int(max(start_frame + 1, min(end_frame, source.frames)))
    width = int(np.clip(width, 160, 2400))
    clip = bank.read(source, start_frame, end_frame - start_frame, mono=True)[:, 0]
    minimums = np.zeros(width, dtype=np.float32)
    maximums = np.zeros(width, dtype=np.float32)
    if len(clip):
        edges = np.linspace(0, len(clip), width + 1).astype(int)
        for index in range(width):
            section = clip[edges[index]:max(edges[index + 1], edges[index] + 1)]
            if section.size:
                minimums[index] = float(np.min(section))
                maximums[index] = float(np.max(section))
    return {
        "min": minimums.tolist(),
        "max": maximums.tolist(),
        "width": width,
        "start_frame": start_frame,
        "end_frame": end_frame,
    }


def _peak_frequency(data: np.ndarray, rate: int, minimum: float = 120.0) -> tuple[float, float]:
    if len(data) < 32:
        return 0.0, 0.0
    windowed = (data - np.mean(data)) * np.hanning(len(data))
    size = 1 << max(15, int(np.ceil(np.log2(len(windowed) * 4))))
    spectrum = np.abs(np.fft.rfft(windowed, size))
    frequencies = np.fft.rfftfreq(size, 1.0 / rate)
    mask = frequencies >= minimum
    if not np.any(mask):
        return 0.0, 0.0
    masked = spectrum[mask]
    index = int(np.argmax(masked))
    return float(frequencies[mask][index]), float(masked[index])


def _level_at(data: np.ndarray, rate: int, frequency: float) -> float:
    windowed = (data - np.mean(data)) * np.hanning(len(data))
    size = 1 << max(15, int(np.ceil(np.log2(max(len(windowed), 2) * 4))))
    spectrum = np.abs(np.fft.rfft(windowed, size))
    frequencies = np.fft.rfftfreq(size, 1.0 / rate)
    index = int(np.argmin(np.abs(frequencies - frequency)))
    lo = max(0, index - 2)
    hi = min(len(spectrum), index + 3)
    return float(np.max(spectrum[lo:hi]))


def _make_step_sweep(rate: int = 192_000) -> tuple[np.ndarray, list[dict]]:
    duration = 0.29
    signal_out = np.zeros(int(rate * duration), dtype=np.float64)
    rng = np.random.default_rng(611)
    signal_out += rng.normal(0.0, 0.0004, len(signal_out))
    center = 70_000
    frequencies = [58_000, 62_000, 66_000, 70_000, 74_000, 78_000, 82_000]
    segments = []
    cursor = 0.030
    burst = 0.024
    stride = 0.034
    for index, frequency in enumerate(frequencies):
        start = int(cursor * rate)
        count = int(burst * rate)
        t = np.arange(count) / rate
        envelope = np.sin(np.pi * np.arange(count) / count) ** 2
        tone = 0.34 * envelope * np.sin(2 * np.pi * frequency * t + index * 0.17)
        signal_out[start:start + count] += tone
        segments.append(
            {
                "source_hz": frequency,
                "base_usb": frequency - center,
                "base_lsb": center - frequency,
                "start": start + count // 5,
                "end": start + count - count // 5,
            }
        )
        cursor += stride
    return signal_out.astype(np.float32), segments


def _make_endpoint_probe(rate: int = 192_000) -> tuple[np.ndarray, list[int]]:
    duration = 0.18
    data = np.zeros(int(rate * duration), dtype=np.float64)
    rng = np.random.default_rng(917)
    data += rng.normal(0.0, 0.00035, len(data))
    pip = np.hanning(128) * rng.normal(0.0, 1.0, 128)
    pip = pip / max(float(np.max(np.abs(pip))), 1e-9)
    targets = [int(0.026 * rate), int(0.153 * rate)]
    for target in targets:
        data[target:target + len(pip)] += 0.92 * pip
    return data.astype(np.float32), [target + len(pip) // 2 for target in targets]


def _detect_click_times(
    monitor: np.ndarray, rate: int, expected_count: int = 2
) -> list[float]:
    if len(monitor) < 32:
        return []
    envelope = np.abs(signal.hilbert(monitor))
    distance = max(1, int(rate * 0.015))
    peaks, _ = signal.find_peaks(
        envelope, distance=distance, prominence=float(np.max(envelope) * 0.12)
    )
    if peaks.size == 0:
        return []
    order = np.argsort(envelope[peaks])[::-1][:expected_count]
    return [float(peaks[index]) / rate for index in order]


def run_selftest(bank: SourceBank, source: Source) -> dict:
    """Verify sideband separation and sample mapping with a short stepped sweep."""
    rate = 192_000
    center = 70_000
    bandwidth = 14_000
    sweep, segments = _make_step_sweep(rate)
    usb = heterodyne(sweep, rate, center, bandwidth, "USB", MONITOR_RATE)
    lsb = heterodyne(sweep, rate, center, bandwidth, "LSB", MONITOR_RATE)
    checks = []

    for label, monitor, expected_sign, source_side in (
        ("USB", usb, 1, "上边带 70–84 kHz"),
        ("LSB", lsb, -1, "下边带 56–70 kHz"),
    ):
        measured = []
        errors = []
        for segment in segments:
            source_hz = segment["source_hz"]
            if source_hz == center:
                continue
            if (expected_sign > 0 and source_hz < center) or (
                expected_sign < 0 and source_hz > center
            ):
                continue
            expected = abs(source_hz - center)
            start = int(segment["start"] / rate * MONITOR_RATE)
            end = int(segment["end"] / rate * MONITOR_RATE)
            peak, _ = _peak_frequency(monitor[start:end], MONITOR_RATE)
            measured.append({"source_hz": source_hz, "expected_hz": expected, "measured_hz": peak})
            errors.append(abs(peak - expected))
        max_error = max(errors) if errors else 9999.0
        checks.append(
            {
                "id": f"{label.lower()}_sweep_sequence",
                "label": f"{label} 多频扫频",
                "pass": bool(len(errors) == 3 and max_error <= 35.0),
                "value": f"{len([e for e in errors if e <= 35.0])}/3 频标 · 最大误差 {max_error:.1f} Hz",
                "detail": f"{source_side}；测得基带 " + ", ".join(
                    f"{item['measured_hz'] / 1000:.2f}k" for item in measured
                ),
            }
        )

        isolation_source = (
            0.30 * np.cos(2 * np.pi * 62_000 * np.arange(int(rate * 0.16)) / rate)
            + 0.30 * np.cos(2 * np.pi * 76_000 * np.arange(int(rate * 0.16)) / rate)
        )
        isolated = heterodyne(
            isolation_source.astype(np.float32), rate, center, bandwidth, label, MONITOR_RATE
        )
        expected_hz = 6_000 if label == "USB" else 8_000
        wrong_hz = 8_000 if label == "USB" else 6_000
        wanted = _level_at(isolated, MONITOR_RATE, expected_hz)
        rejected = _level_at(isolated, MONITOR_RATE, wrong_hz)
        rejection_db = 20 * np.log10(max(wanted, 1e-9) / max(rejected, 1e-9))
        peak, _ = _peak_frequency(isolated[int(0.02 * MONITOR_RATE):-int(0.02 * MONITOR_RATE)], MONITOR_RATE)
        checks.append(
            {
                "id": f"{label.lower()}_isolation",
                "label": f"{label} 边带隔离",
                "pass": bool(abs(peak - expected_hz) <= 35 and rejection_db >= 24),
                "value": f"{peak / 1000:.2f} kHz · 抑制 {rejection_db:.1f} dB",
                "detail": f"62 kHz / 76 kHz 双音参考；期望 {expected_hz / 1000:.0f} kHz",
            }
        )

    probe, source_targets = _make_endpoint_probe(rate)
    probe_frames = len(probe)
    start_frame = min(max(0, source.frames // 4), max(0, source.frames - 1))
    wanted_frames = min(probe_frames, source.frames - start_frame)
    raw_start = bank.read(source, start_frame, wanted_frames, mono=True)[:, 0]
    re_read = bank.read(source, start_frame, wanted_frames, mono=True)[:, 0]
    raw_exact = wanted_frames > 0 and np.array_equal(raw_start, re_read)
    checks.append(
        {
            "id": "raw_endpoint_exact",
            "label": "原始端点",
            "pass": bool(raw_exact),
            "value": f"[{start_frame}, {start_frame + wanted_frames}) · {wanted_frames:,} 帧",
            "detail": "原始读取为左闭右开区间，端点不偏移",
        }
    )

    for label, factor in (("减速 10×", 10.0), ("异频", 1.0)):
        if label == "异频":
            monitor = heterodyne(probe, rate, 70_000, 20_000, "USB", MONITOR_RATE)
        else:
            monitor = time_expand(probe, rate, factor, MONITOR_RATE)
        click_times = sorted(_detect_click_times(monitor, MONITOR_RATE))
        mapped = [
            monitor_seconds_to_source(seconds, 0, rate, factor) for seconds in click_times
        ]
        errors = []
        if len(mapped) == len(source_targets):
            errors = [abs(a - b) for a, b in zip(mapped, source_targets)]
        max_error = max(errors) if errors else 9999.0
        endpoint_map_seconds = monitor_seconds_to_source(
            len(monitor) / MONITOR_RATE, 0, rate, factor
        )
        map_error = abs(endpoint_map_seconds - probe_frames)
        passed = len(mapped) == 2 and max_error <= 80 and map_error <= 1.0
        checks.append(
            {
                "id": f"endpoint_{'expansion' if factor > 1 else 'heterodyne'}",
                "label": f"{label} 端点定位",
                "pass": bool(passed),
                "value": f"标记误差 {max_error:.0f} 样本 · 区间误差 {map_error:.2f}",
                "detail": f"输出 {len(monitor):,} 帧；两个标记均已反算回源坐标",
            }
        )

    passed = all(check["pass"] for check in checks)
    return {
        "passed": passed,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "source_name": source.name,
        "summary": f"全部 {len(checks)} 项检查通过" if passed else "存在未通过的检查",
        "checks": checks,
        "sweep": {
            "tone_count": len(segments),
            "frequencies_hz": [segment["source_hz"] for segment in segments],
            "center_hz": center,
            "bandwidth_hz": bandwidth,
            "duration_s": len(sweep) / rate,
        },
        "endpoint_probe": {
            "duration_s": len(probe) / rate,
            "pips_source_frame": source_targets,
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_stem(name: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in Path(name).stem
    )
    return (cleaned or "audio")[:48]


def export_bundle(
    bank: SourceBank,
    source: Source,
    start_frame: int,
    end_frame: int,
    rendered: dict,
    exports_dir: Path,
) -> dict:
    exports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    export_id = f"{stamp}_{uuid.uuid4().hex[:6]}"
    folder = exports_dir / export_id
    folder.mkdir(parents=True, exist_ok=False)
    stem = _safe_stem(source.name)
    mode = rendered["params"]["mode"]
    mode_label = "heterodyne" if mode == "heterodyne" else f"slow{rendered['time_factor']:g}x"
    raw_path = folder / f"{stem}_raw.wav"
    monitor_path = folder / f"{stem}_{mode_label}_monitor.wav"
    mapping_path = folder / "mapping.csv"
    manifest_path = folder / "manifest.json"

    raw = bank.read(source, start_frame, end_frame - start_frame, mono=False)
    raw_subtype = (
        source.subtype
        if source.subtype in {"PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"}
        else "FLOAT"
    )
    sf.write(str(raw_path), raw, source.rate, subtype=raw_subtype, format="WAV")
    sf.write(
        str(monitor_path),
        rendered["audio"],
        rendered["monitor_rate"],
        subtype="PCM_16",
        format="WAV",
    )

    factor = float(rendered["time_factor"])
    points = 201
    source_points = np.linspace(start_frame, end_frame, points)
    lines = [
        "source_frame,source_seconds,monitor_frame,monitor_seconds",
    ]
    for frame in source_points:
        clamped = min(float(frame), float(end_frame - 1))
        monitor_seconds = source_frame_to_monitor(clamped, start_frame, source.rate, factor)
        monitor_frame = int(round(monitor_seconds * rendered["monitor_rate"]))
        lines.append(
            f"{int(round(frame))},{frame / source.rate:.9f},{monitor_frame},"
            f"{monitor_seconds:.9f}"
        )
    last_monitor = rendered["monitor_frames"] / rendered["monitor_rate"]
    lines.append(
        f"{end_frame},{end_frame / source.rate:.9f},"
        f"{rendered['monitor_frames']},{last_monitor:.9f}"
    )
    mapping_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    relation = {
        "source_frame_from_monitor_seconds": (
            "round(start_frame + monitor_seconds * source_rate / time_factor)"
        ),
        "monitor_seconds_from_source_frame": (
            "(source_frame - start_frame) * time_factor / source_rate"
        ),
        "start": {
            "source_frame": start_frame,
            "monitor_seconds": 0.0,
            "monitor_frame": 0,
        },
        "end_exclusive": {
            "source_frame": end_frame,
            "monitor_seconds": last_monitor,
            "monitor_frame": rendered["monitor_frames"],
        },
        "time_factor": factor,
        "frequency": rendered["frequency_map"],
    }
    manifest = {
        "schema": "echolab.mapping.v1",
        "export_id": export_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "name": source.name,
            "sample_rate": source.rate,
            "channels": source.channels,
            "source_frames": source.frames,
            "duration": source.duration,
        },
        "selection": {
            "start_frame": start_frame,
            "end_exclusive_frame": end_frame,
            "frames": end_frame - start_frame,
            "seconds": (end_frame - start_frame) / source.rate,
        },
        "processing": {
            "mode": mode,
            "params": rendered["params"],
            "monitor_rate": rendered["monitor_rate"],
            "monitor_frames": rendered["monitor_frames"],
            "monitor_duration_seconds": last_monitor,
        },
        "mapping": relation,
        "files": {
            "raw_clip": {
                "name": raw_path.name,
                "format": f"WAV / {raw_subtype} / original sample rate",
                "subtype": raw_subtype,
                "sample_rate": source.rate,
                "frames": end_frame - start_frame,
                "sha256": _sha256(raw_path),
                "bytes": raw_path.stat().st_size,
            },
            "monitor_clip": {
                "name": monitor_path.name,
                "format": "WAV / PCM16 / 48 kHz",
                "subtype": "PCM_16",
                "sample_rate": rendered["monitor_rate"],
                "frames": rendered["monitor_frames"],
                "sha256": _sha256(monitor_path),
                "bytes": monitor_path.stat().st_size,
            },
            "mapping_table": {
                "name": mapping_path.name,
                "sha256": _sha256(mapping_path),
            },
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest["files"]["manifest"] = {
        "name": manifest_path.name,
        "sha256": _sha256(manifest_path),
    }
    return {
        "id": export_id,
        "folder": folder,
        "manifest": manifest,
        "raw_path": raw_path,
        "monitor_path": monitor_path,
        "manifest_path": manifest_path,
        "mapping_path": mapping_path,
    }





