"""Rolling-shutter calibration from LED flicker video.

Estimates per-row readout time (line time), per-row exposure time and
readout scan direction of a rolling-shutter camera from footage of an LED
flickering at a known frequency.
"""

from .model import (
    exposure_fraction,
    led_harmonic,
    exposure_transfer,
    normalized_sinc,
)

__version__ = "0.1.0"

__all__ = [
    "exposure_fraction",
    "led_harmonic",
    "exposure_transfer",
    "normalized_sinc",
    "__version__",
]
