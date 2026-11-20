"""Synthetic four-speaker meeting used to validate the full pipeline.

The scenario deliberately includes every tricky case from the brief:

* two sample-rate versions (16 kHz for A/C, 24 kHz for B/D);
* deliberate cross-talk (B interrupts A, C talks over A);
* a weakly pronounced final syllable (very low amplitude, still labelled);
* neighbouring short utterances that must merge, plus a gap large enough
  that merging must not happen.

Tones stand in for speech; silence is exact zero so cut placement stays
deterministic.  Tracks are quantised to 16-bit and decoded again, making the
later clip write a lossless, sample-exact round trip.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from relisten.audio import Track, read_wav, write_wav
from relisten.model import Utterance


SPEAKER_RATES = {"A": 16000, "B": 24000, "C": 16000, "D": 24000}
SPEAKER_TONES = {"A": 220.0, "B": 330.0, "C": 440.0, "D": 550.0}
MEETING_DURATION = 15.0
FADE_SECONDS = 0.008

# (speaker, start, end, weak-tail end or None)
LABELS = [
    ("A", 1.0, 2.0, None),
    ("A", 2.6, 3.4, None),       # 0.6 s gap -> merges with A1
    ("A", 6.4, 9.0, 9.0),        # weak tail 8.6-9.0; C talks over it
    ("B", 3.0, 3.8, None),       # crosses A2 on 3.0-3.4
    ("B", 4.2, 4.9, None),       # 0.4 s gap -> merges with B1
    ("B", 10.0, 10.5, None),     # isolated
    ("C", 5.0, 5.5, None),
    ("C", 5.6, 6.0, None),       # 0.1 s gap -> merges with C1
    ("C", 8.0, 8.4, None),       # crosses A3
    ("D", 9.4, 9.8, None),       # isolated, no overlap with A3
    ("D", 12.0, 12.6, None),
    ("D", 14.0, 14.4, None),     # 1.4 s gap -> must stay separate
]

WEAK_TAIL_START = 8.6
WEAK_TAIL_AMPLITUDE = 0.02
SPEECH_AMPLITUDE = 0.5

# speaker_id -> expected number of output groups
EXPECTED_GROUPS = {"A": 2, "B": 2, "C": 2, "D": 3}


def _tone_track(
    speaker_id: str,
    labels: list[tuple[str, float, float, float | None]],
) -> list[float]:
    rate = SPEAKER_RATES[speaker_id]
    samples = [0.0] * round(MEETING_DURATION * rate)
    tone = SPEAKER_TONES[speaker_id]
    fade = round(FADE_SECONDS * rate)

    for speaker, start, end, weak_end in labels:
        if speaker != speaker_id:
            continue
        start_index = round(start * rate)
        end_index = round(end * rate)
        weak_index = (
            round(WEAK_TAIL_START * rate)
            if weak_end is not None
            else end_index
        )
        span = end_index - start_index
        for index in range(start_index, end_index):
            amplitude = (
                WEAK_TAIL_AMPLITUDE
                if index >= weak_index
                else SPEECH_AMPLITUDE
            )
            position = index - start_index
            fade_scale = min(1.0, position / fade, (span - position) / fade)
            samples[index] = (
                amplitude
                * fade_scale
                * math.sin(2 * math.pi * tone * index / rate)
            )
    return samples


def build_meeting(base_dir: str | Path) -> dict:
    base = Path(base_dir)
    tracks_dir = base / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    tracks: dict[str, Track] = {}
    for speaker_id, rate in SPEAKER_RATES.items():
        path = tracks_dir / f"speaker_{speaker_id}.wav"
        write_wav(path, _tone_track(speaker_id, LABELS), rate)
        tracks[speaker_id] = read_wav(path, speaker_id)

    utterances = [
        Utterance(speaker_id=speaker, start=start, end=end)
        for speaker, start, end, _ in LABELS
    ]
    annotations_path = base / "annotations.json"
    with annotations_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "meeting_id": "synthetic-four-party",
                "utterances": [
                    {
                        "speaker_id": speaker,
                        "start": start,
                        "end": end,
                    }
                    for speaker, start, end, _ in LABELS
                ],
            },
            handle,
            indent=2,
        )

    return {
        "tracks": tracks,
        "utterances": utterances,
        "annotations_path": annotations_path,
        "base": base,
    }
