from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import TimestampSummary


def extract_decoder_timestamps(path: str | Path) -> list[float | None]:
    """Return presentation timestamps from decoded AVFrames, in milliseconds.

    PyAV decodes and reorders frames exactly as the decoder presents them. The
    PTS values are therefore decoder-layer presentation times, not wall-clock
    capture times.
    """

    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required for decoder timestamps") from exc

    values: list[float | None] = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"no video stream: {path}")
        stream = container.streams.video[0]
        time_base = stream.time_base
        for frame in container.decode(stream):
            if frame.pts is not None and time_base is not None:
                values.append(float(frame.pts * time_base) * 1000.0)
            elif frame.time is not None:
                values.append(float(frame.time) * 1000.0)
            else:
                values.append(None)
    return values


def summarize_timestamps(
    pts_ms: list[float | None],
    frame_count: int,
    tolerance_ms: float,
    tolerance_ratio: float,
) -> TimestampSummary:
    available = [value for value in pts_ms if value is not None]
    missing = max(frame_count - len(available), len(pts_ms) - len(available))
    if len(available) < 2:
        return TimestampSummary(
            frame_count=frame_count,
            available_count=len(available),
            missing_count=missing,
            nominal_delta_ms=None,
            continuity="unavailable",
            regular_ratio=0.0,
            gap_count=0,
            backstep_count=0,
            duplicate_pts_count=0,
            first_ms=available[0] if available else None,
            last_ms=available[-1] if available else None,
            median_delta_ms=None,
        )

    deltas = np.diff(np.asarray(available, dtype=np.float64))
    positive = deltas[deltas > 1e-6]
    nominal = float(np.median(positive)) if positive.size else None
    if nominal is None or nominal <= 0:
        return TimestampSummary(
            frame_count=frame_count,
            available_count=len(available),
            missing_count=missing,
            nominal_delta_ms=None,
            continuity="discontinuous",
            regular_ratio=0.0,
            gap_count=0,
            backstep_count=int(np.count_nonzero(deltas <= 1e-6)),
            duplicate_pts_count=int(np.count_nonzero(np.abs(deltas) <= 1e-6)),
            first_ms=float(available[0]),
            last_ms=float(available[-1]),
            median_delta_ms=float(np.median(deltas)),
        )

    tolerance = max(tolerance_ms, nominal * tolerance_ratio)
    regular = np.abs(deltas - nominal) <= tolerance
    gaps = deltas > nominal + tolerance
    backsteps = deltas <= 1e-6
    duplicate_pts = np.abs(deltas) <= 1e-6
    regular_ratio = float(np.mean(regular))
    continuity = (
        "continuous"
        if missing == 0 and regular_ratio >= 0.98 and not gaps.any() and not backsteps.any()
        else "discontinuous"
    )
    if missing > 0 and len(available) > 0:
        continuity = "partial"
    return TimestampSummary(
        frame_count=frame_count,
        available_count=len(available),
        missing_count=missing,
        nominal_delta_ms=nominal,
        continuity=continuity,
        regular_ratio=regular_ratio,
        gap_count=int(np.count_nonzero(gaps)),
        backstep_count=int(np.count_nonzero(backsteps)),
        duplicate_pts_count=int(np.count_nonzero(duplicate_pts)),
        first_ms=float(available[0]),
        last_ms=float(available[-1]),
        median_delta_ms=float(np.median(deltas)),
    )
