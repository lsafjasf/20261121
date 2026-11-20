"""Compute backend.

Numpy is optional.  All heavy routines below have a standard-library
implementation, so the toolkit keeps working on stripped-down machines
(archive workstations often are).  When numpy is present we route through it.
"""

from __future__ import annotations

try:  # pragma: no cover - exercised implicitly in the two environments
    import numpy as _np

    HAVE_NUMPY = True
except Exception:  # pragma: no cover
    _np = None
    HAVE_NUMPY = False


def backend_name() -> str:
    return "numpy" if HAVE_NUMPY else "stdlib"


def as_array(buf, width: int, height: int):
    """Wrap a byte/bytearray buffer as a 2-D uint8 array or raw buffer."""
    if HAVE_NUMPY:
        return _np.frombuffer(bytes(buf), dtype=_np.uint8).reshape(height, width)
    return bytes(buf)


def sad(a, b) -> int:
    """Sum of absolute differences for two equally sized grayscale images."""
    if HAVE_NUMPY:
        return int(_np.abs(a.astype(_np.int16) - b.astype(_np.int16)).sum())
    return sum(abs(x - y) for x, y in zip(a, b))


def mean_abs(a, b) -> float:
    """Mean absolute pixel difference (SAD normalised by pixel count)."""
    if HAVE_NUMPY:
        return float(
            _np.abs(a.astype(_np.int16) - b.astype(_np.int16)).mean()
        )
    if not a:
        return 0.0
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def mean_abs_subset(a, b, max_samples: int = 4096) -> float:
    """Cheap MAD on an even stride sample; keeps stdlib mode fast."""
    n = len(a)
    if n <= max_samples:
        return mean_abs(a, b)
    step = max(1, n // max_samples)
    if HAVE_NUMPY:
        sa = a.reshape(-1)[::step]
        sb = b.reshape(-1)[::step]
        return float(_np.abs(sa.astype(_np.int16) - sb.astype(_np.int16)).mean())
    total = 0
    count = 0
    for i in range(0, n, step):
        total += abs(a[i] - b[i])
        count += 1
    return total / count
