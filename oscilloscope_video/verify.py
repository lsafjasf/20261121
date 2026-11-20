"""Decode an output MP4 and verify cue timing and event first frames."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from array import array
from fractions import Fraction
from pathlib import Path
from typing import Any

from .audio import Cue, DEFAULT_CUE_AMPLITUDE, cue_tone
from .encode import require_ffmpeg


def verify_mp4(
    video_path: str | Path,
    manifest_path: str | Path,
    *,
    audio_tolerance_ms: float = 10.0,
    min_waveform_correlation: float = 0.62,
) -> dict[str, Any]:
    video = Path(video_path)
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    issues: list[str] = []
    probe = _probe(video, issues)
    frame_rate = Fraction(manifest["video"]["fps"], 1)
    stream_rate = probe.get("avg_frame_rate", "0/0")
    if Fraction(stream_rate) != frame_rate:
        issues.append(f"frame rate is {stream_rate}, expected {frame_rate}")
    if int(probe.get("nb_frames") or 0) != manifest["video"]["frame_count"]:
        issues.append(
            f"video has {probe.get('nb_frames')} frames, "
            f"expected {manifest['video']['frame_count']}"
        )
    if probe.get("codec_name") != "h264":
        issues.append(f"video codec is {probe.get('codec_name')}, expected h264")
    if probe.get("width") != manifest["video"]["width"]:
        issues.append("decoded width differs from manifest")
    if probe.get("height") != manifest["video"]["height"]:
        issues.append("decoded height differs from manifest")

    audio_rate = int(manifest["audio"]["sample_rate"])
    decoded_audio = _decode_audio(video, audio_rate)
    if not decoded_audio:
        issues.append("decoded audio is empty")

    event_results = []
    for event in manifest["events"]:
        cue = Cue(
            event_index=event["index"],
            video_time_s=event["video_time_s"],
            start_sample=event["audio_sample"],
            sample_count=event["audio_sample_count"],
            hz=event["cue_hz"],
            amplitude=DEFAULT_CUE_AMPLITUDE,
        )
        result = _verify_cue(
            decoded_audio,
            cue,
            audio_rate,
            audio_tolerance_ms,
            min_waveform_correlation,
        )
        result["label"] = event["label"]
        if not result["pass"]:
            issues.append(f"event {event['index'] + 1} cue timing/waveform failed")
        event_results.append(result)

    flash_flags, decoded_frames = _decode_flash_flags(
        video,
        manifest["video"]["width"],
        manifest["video"]["height"],
    )
    if decoded_frames != manifest["video"]["frame_count"]:
        issues.append(
            f"decoded {decoded_frames} flash-analysis frames, "
            f"expected {manifest['video']['frame_count']}"
        )
    flash_runs = _flash_runs(flash_flags)
    for event_result, event in zip(event_results, manifest["events"]):
        expected = int(event["first_frame"])
        tolerance = int(manifest["render"]["event_flash_frames"]) + 1
        decoded_first = _matching_run_start(flash_runs, expected, tolerance)
        event_result["expected_first_frame"] = expected
        event_result["decoded_first_frame"] = decoded_first
        event_result["first_frame_error"] = (
            None if decoded_first is None else decoded_first - expected
        )
        if decoded_first != expected:
            issues.append(
                f"event {event['index'] + 1} first red frame is "
                f"{decoded_first}, expected {expected}"
            )
            event_result["pass"] = False

    return {
        "passed": not issues,
        "issues": issues,
        "video": {
            "codec": probe.get("codec_name"),
            "width": probe.get("width"),
            "height": probe.get("height"),
            "avg_frame_rate": probe.get("avg_frame_rate"),
            "frames": decoded_frames,
            "flash_runs": flash_runs,
        },
        "audio": {
            "sample_rate": audio_rate,
            "decoded_samples": len(decoded_audio),
            "duration_s": round(len(decoded_audio) / audio_rate, 9) if decoded_audio else 0,
            "tolerance_ms": audio_tolerance_ms,
        },
        "events": event_results,
    }


def _probe(video: Path, issues: list[str]) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe was not found on PATH")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(video),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        issues.append(f"ffprobe failed: {result.stderr.strip()}")
        return {}
    streams = json.loads(result.stdout).get("streams", [])
    return streams[0] if streams else {}


def _decode_audio(video: Path, sample_rate: int) -> array[int]:
    ffmpeg = require_ffmpeg()
    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(video),
        "-map",
        "0:a:0",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    raw = subprocess.check_output(command)
    samples = array("h")
    samples.frombytes(raw[: len(raw) - (len(raw) % 2)])
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def _verify_cue(
    samples: array[int],
    cue: Cue,
    sample_rate: int,
    tolerance_ms: float,
    min_waveform_correlation: float,
) -> dict[str, Any]:
    reference = [value / 32768.0 for value in cue_tone(cue, sample_rate)]
    search = max(1, round(0.030 * sample_rate))
    expected = cue.start_sample
    base = expected - search
    result = {
        "index": cue.event_index,
        "label": "",
        "expected_cue_sample": expected,
        "cue_hz": cue.hz,
        "sample_count": cue.sample_count,
        "decoded_start_sample": None,
        "sample_error": None,
        "error_ms": None,
        "waveform_correlation": 0.0,
        "pass": False,
    }
    if not samples:
        return result
    envelope_window = max(2, round(0.0015 * sample_rate))
    envelope = _moving_average_abs(
        samples,
        base,
        len(reference) + 2 * search + 2 * envelope_window,
        envelope_window,
    )
    reference_envelope = _moving_average_abs(
        array("h", [round(value * 32768.0) for value in reference]),
        0,
        len(reference) + 2 * envelope_window,
        envelope_window,
    )
    reference_window = reference_envelope[: len(reference)]
    reference_energy = sum(value * value for value in reference_window)
    if reference_energy <= 0:
        return result

    best_lag = 0
    best_score = -1.0
    for lag in range(-search, search + 1, 4):
        score = _envelope_correlation(
            envelope,
            search + lag,
            reference_window,
            reference_energy,
        )
        if score > best_score:
            best_score = score
            best_lag = lag
    for lag in range(max(-search, best_lag - 5), min(search, best_lag + 5) + 1):
        score = _envelope_correlation(
            envelope,
            search + lag,
            reference_window,
            reference_energy,
        )
        if score > best_score:
            best_score = score
            best_lag = lag

    decoded_start = expected + best_lag
    start = decoded_start
    end = start + len(reference)
    if start >= 0 and end <= len(samples):
        waveform_correlation = _waveform_correlation(samples, start, reference)
    else:
        waveform_correlation = 0.0
    error_ms = best_lag / sample_rate * 1000.0
    result.update(
        {
            "decoded_start_sample": decoded_start,
            "sample_error": best_lag,
            "error_ms": round(error_ms, 4),
            "envelope_correlation": round(best_score, 5),
            "waveform_correlation": round(waveform_correlation, 5),
        }
    )
    result["pass"] = (
        abs(error_ms) <= tolerance_ms
        and waveform_correlation >= min_waveform_correlation
    )
    return result


def _moving_average_abs(
    samples: array[int],
    start: int,
    count: int,
    window: int | None = None,
) -> list[float]:
    window = max(2, round(0.0015 * 48000)) if window is None else window
    padded_start = start - window
    padded_end = start + count
    values = [0.0] * max(0, padded_end - padded_start)
    for index in range(max(0, padded_start), min(len(samples), padded_end)):
        values[index - padded_start] = abs(float(samples[index]))
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)
    output = []
    offset = start - padded_start
    for index in range(count):
        left = max(0, offset + index - window + 1)
        right = min(len(values) - 1, offset + index)
        total = prefix[right + 1] - prefix[left]
        output.append(total / max(1, right - left + 1))
    return output


def _envelope_correlation(
    decoded: list[float],
    start: int,
    reference: list[float],
    reference_energy: float,
) -> float:
    if start < 0 or start + len(reference) > len(decoded):
        return -1.0
    dot = 0.0
    energy = 0.0
    for offset, ref_value in enumerate(reference):
        value = decoded[start + offset]
        dot += value * ref_value
        energy += value * value
    if energy <= 0:
        return -1.0
    return dot / math.sqrt(energy * reference_energy)


def _waveform_correlation(samples: array[int], start: int, reference: list[float]) -> float:
    dot = 0.0
    energy = 0.0
    for offset, ref_value in enumerate(reference):
        value = samples[start + offset] / 32768.0
        dot += value * ref_value
        energy += value * value
    if energy <= 0:
        return 0.0
    return dot / math.sqrt(energy * sum(value * value for value in reference))


def _decode_flash_flags(
    video: Path,
    width: int,
    height: int,
) -> tuple[list[bool], int]:
    ffmpeg = require_ffmpeg()
    target_width = min(320, width)
    target_height = max(2, round(height * target_width / width))
    if target_height % 2:
        target_height += 1
    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-vf",
        f"scale={target_width}:{target_height}:flags=area",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    raw = subprocess.check_output(command)
    frame_bytes = target_width * target_height * 3
    frame_count = len(raw) // frame_bytes
    flags = []
    for frame_index in range(frame_count):
        frame = raw[frame_index * frame_bytes : (frame_index + 1) * frame_bytes]
        red_pixels = 0
        for index in range(0, len(frame), 3):
            red, green, blue = frame[index], frame[index + 1], frame[index + 2]
            if red > 190 and green < 90 and blue < 90 and red - green > 120 and red - blue > 120:
                red_pixels += 1
        flags.append(red_pixels >= 20)
    return flags, frame_count


def _flash_runs(flags: list[bool]) -> list[list[int]]:
    runs: list[list[int]] = []
    start: int | None = None
    for index, active in enumerate(flags):
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append([start, index - 1])
            start = None
    if start is not None:
        runs.append([start, len(flags) - 1])
    return runs


def _matching_run_start(
    runs: list[list[int]],
    expected: int,
    tolerance: int,
) -> int | None:
    matches = [start for start, _end in runs if abs(start - expected) <= tolerance]
    return min(matches, key=lambda value: abs(value - expected)) if matches else None
