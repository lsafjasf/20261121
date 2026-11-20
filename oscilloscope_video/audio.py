"""Sample-accurate cue-tone synthesis and WAV writing."""

from __future__ import annotations

import math
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .model import Event, Timeline

DEFAULT_CUE_HZ = 1320.0
DEFAULT_CUE_SECONDS = 0.085
DEFAULT_CUE_AMPLITUDE = 0.62


@dataclass(frozen=True)
class Cue:
    event_index: int
    video_time_s: float
    start_sample: int
    sample_count: int
    hz: float
    amplitude: float

    @property
    def end_sample(self) -> int:
        return self.start_sample + self.sample_count


def cues_for_events(
    events: Sequence[Event],
    timeline: Timeline,
    sample_rate: int,
    *,
    cue_seconds: float = DEFAULT_CUE_SECONDS,
) -> list[Cue]:
    sample_count = max(2, round(cue_seconds * sample_rate))
    cues: list[Cue] = []
    for index, event in enumerate(events):
        video_time = timeline.video_time(event.time_s)
        cues.append(
            Cue(
                event_index=index,
                video_time_s=video_time,
                start_sample=round(video_time * sample_rate),
                sample_count=sample_count,
                hz=float(event.cue_hz or DEFAULT_CUE_HZ),
                amplitude=DEFAULT_CUE_AMPLITUDE,
            )
        )
    return cues


def build_cue_audio(
    duration_s: float,
    cues: Sequence[Cue],
    sample_rate: int,
) -> array[int]:
    """Mix all cue tones into mono signed 16-bit PCM."""

    sample_total = max(1, round(duration_s * sample_rate))
    samples = array("h", [0]) * sample_total
    for cue in cues:
        tone = cue_tone(cue, sample_rate)
        for offset, value in enumerate(tone):
            index = cue.start_sample + offset
            if 0 <= index < sample_total:
                mixed = samples[index] + value
                samples[index] = max(-32768, min(32767, mixed))
    return samples


def cue_tone(cue: Cue, sample_rate: int) -> list[int]:
    """Return the deterministic PCM waveform used by an event cue.

    The Hann window starts and ends at zero.  Its exact sample placement is
    retained in the manifest so verification can compare decoded audio with
    the same waveform.
    """

    values: list[int] = []
    count = cue.sample_count
    angular = 2.0 * math.pi * cue.hz / sample_rate
    for index in range(count):
        phase = (index + 0.5) / count
        envelope = math.sin(math.pi * phase) ** 2
        value = cue.amplitude * envelope * math.sin(angular * index)
        values.append(round(value * 32767.0))
    return values


def float_reference(cue: Cue, sample_rate: int) -> list[float]:
    return [value / 32768.0 for value in cue_tone(cue, sample_rate)]


def write_wav(path: str | Path, samples: array[int], sample_rate: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())
