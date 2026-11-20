"""Load manually labelled speech intervals from JSON.

Accepted schema (seconds, float)::

    {
      "utterances": [
        {"speaker_id": "A", "start": 1.00, "end": 3.20},
        ...
      ]
    }

A top-level list is accepted as well.  Unknown keys are ignored so that the
same document may also carry meeting metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

from .model import Utterance


def load_annotations(path: str | Path) -> list[Utterance]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        raw = payload.get("utterances")
        if raw is None:
            raise ValueError(
                f"{path}: annotation JSON must contain an 'utterances' list"
            )
    elif isinstance(payload, list):
        raw = payload
    else:
        raise ValueError(f"{path}: unsupported annotation document")

    utterances: list[Utterance] = []
    for index, item in enumerate(raw):
        try:
            utterances.append(
                Utterance(
                    speaker_id=str(item["speaker_id"]),
                    start=float(item["start"]),
                    end=float(item["end"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}: invalid utterance at index {index}: {item!r}"
            ) from exc

    if not utterances:
        raise ValueError(f"{path}: no utterances found")
    return utterances
