from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class LabConfig:
    width: int = 480
    height: int = 360
    fps: int = 24
    seed: int = 21
    noise_sigma: float = 0.0032
    instability_gain: float = 6.5
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    # True panel response used by the synthetic recording.  A channel's
    # response time is intentionally different from the other two.
    tau_rise: tuple[float, float, float] = (1.45, 3.10, 6.80)
    tau_fall: tuple[float, float, float] = (0.85, 1.75, 4.10)

    # Small, deterministic directional bias in the calibration recording.
    # Restoration uses only the globally estimated response.
    direction_tau_bias: dict[str, tuple[float, float, float]] = field(
        default_factory=lambda: {
            "right": (1.08, 1.03, 1.04),
            "left": (0.94, 0.97, 0.96),
            "down": (1.04, 0.99, 1.02),
            "up": (0.96, 1.02, 0.98),
        }
    )

    @property
    def channels(self) -> tuple[str, str, str]:
        return ("R", "G", "B")

    @property
    def alpha_rise(self) -> np.ndarray:
        return 1.0 - np.exp(-1.0 / np.asarray(self.tau_rise))

    @property
    def alpha_fall(self) -> np.ndarray:
        return 1.0 - np.exp(-1.0 / np.asarray(self.tau_fall))
