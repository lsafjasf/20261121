"""PI controller that steers slave-device buffer occupancy back to target.

The output is the resampling *consume ratio*: slave samples consumed per
master output frame.  A full buffer means the slave clock is running fast,
so the ratio rises to drain it; an empty buffer lowers the ratio so the
level recovers.  The control bandwidth stays well below 1 Hz, which is
fast enough to absorb crystal drift yet slow enough not to add audible
pitch wobble or track the audio waveform.
"""

from __future__ import annotations


class FillController:
    def __init__(
        self,
        target_fill: int,
        kp: float,
        ki: float,
        ratio_min: float = 0.98,
        ratio_max: float = 1.02,
        max_slew: float = 1e-3,
    ) -> None:
        self.target_fill = int(target_fill)
        self.kp = float(kp)
        self.ki = float(ki)
        self.ratio_min = float(ratio_min)
        self.ratio_max = float(ratio_max)
        self.max_slew = float(max_slew)
        self._integral = 0.0
        self._ratio = 1.0

    def reset(self) -> None:
        self._integral = 0.0
        self._ratio = 1.0

    @property
    def ratio(self) -> float:
        return self._ratio

    def update(self, fill: int) -> float:
        """One step per master block; positive error = buffer too full."""
        error = fill - self.target_fill

        tentative = self._integral + self.ki * error
        candidate = self._clamp(1.0 + self.kp * error + tentative)
        # Conditional integration (anti-windup): freeze the accumulator when
        # the actuator is already saturated in the error's direction.
        saturated_hi = candidate >= self.ratio_max and error > 0
        saturated_lo = candidate <= self.ratio_min and error < 0
        if not (saturated_hi or saturated_lo):
            self._integral = tentative

        target_ratio = self._clamp(
            1.0 + self.kp * error + self._integral
        )
        # Slew-rate limit keeps every ratio change sample-continuous.
        delta = target_ratio - self._ratio
        delta = max(-self.max_slew, min(self.max_slew, delta))
        self._ratio += delta
        return self._ratio

    def _clamp(self, value: float) -> float:
        return max(self.ratio_min, min(self.ratio_max, value))
