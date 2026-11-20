"""Priority-based mixing console with sidechain ducking.

Three channels run simultaneously:

    ambient (priority 0)  <  narration (priority 1)  <  alert (priority 2)

Whenever a higher-priority channel is active, every lower-priority channel
is ducked toward a configured floor.  Gain moves with independent attack /
release time constants (one-pole smoother), so a re-triggered alert while a
previous one is still releasing simply re-engages the attack.

The narration channel *pauses its playhead* while fully ducked by an alert,
so when the alert ends the narration resumes at exactly the sample it was
about to play - no skipped syllables, no time-stretching.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field


@dataclass
class Channel:
    name: str
    priority: int
    signal: list[float]
    base_gain: float = 1.0
    duck_floor: float = 1.0          # gain floor while a higher prio speaks
    attack_ms: float = 20.0
    release_ms: float = 250.0
    active_threshold: float = 0.005
    pause_when_ducked: bool = False
    always_active: bool = False
    position: int = 0
    gain: float = 1.0
    paused: bool = False
    has_content: bool = True
    # Per-block snapshots filled in by the mixer, for the gain-curve log.
    history: list[tuple[float, float, float]] = field(default_factory=list)

    def is_active(self) -> bool:
        if self.always_active:
            return True
        if not self.has_content or self.position >= len(self.signal):
            return False
        window = self.signal[self.position:self.position + 256]
        return any(abs(s) > self.active_threshold for s in window)

    def reset(self) -> None:
        self.position = 0
        self.gain = self.base_gain
        self.paused = False
        self.has_content = True
        self.history.clear()


class Mixer:
    """Block-wise mixer with scheduled one-shot events and gain logging."""

    def __init__(self, sample_rate: int = 48000, block_size: int = 256,
                 soft_clip: float = 0.97):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.soft_clip = soft_clip
        self.channels: list[Channel] = []
        # Heap of (sample_index, seq, callable) scheduled events.
        self._events: list[tuple[int, int, object]] = []
        self._event_seq = 0
        # Output diagnostics: (time_s, raw_peak, clipped_peak).
        self.output_log: list[tuple[float, float, float]] = []

    # ------------------------------------------------------------------ setup
    def add_channel(self, channel: Channel) -> Channel:
        self.channels.append(channel)
        self.channels.sort(key=lambda c: c.priority)
        return channel

    def channel(self, name: str) -> Channel:
        for ch in self.channels:
            if ch.name == name:
                return ch
        raise KeyError(name)

    def schedule(self, sample_index: int, fn: object) -> None:
        """Run ``fn()`` once when the render reaches ``sample_index``."""
        heapq.heappush(self._events, (sample_index, self._event_seq, fn))
        self._event_seq += 1

    def schedule_seconds(self, seconds: float, fn: object) -> None:
        self.schedule(int(seconds * self.sample_rate), fn)

    def play_on(self, channel_name: str, signal: list[float],
                at_seconds: float) -> None:
        """Start ``signal`` on a channel at the scheduled time.

        A later event preempts whatever is playing: the new clip replaces
        the buffer from sample zero, which models a higher-priority
        announcement cutting into the current one.
        """
        ch = self.channel(channel_name)

        def _fire() -> None:
            ch.signal = signal
            ch.position = 0
            ch.paused = False
            ch.has_content = True

        self.schedule_seconds(at_seconds, _fire)

    # ------------------------------------------------------------- sidechain
    def _duck_target(self, ch: Channel) -> float:
        """Target gain for ``ch`` given currently speaking higher prios."""
        someone_higher = any(
            other.priority > ch.priority and other.is_active()
            for other in self.channels)
        return ch.duck_floor if someone_higher else ch.base_gain

    @staticmethod
    def _coeff(time_constant_ms: float, sample_rate: int,
               block_size: int) -> float:
        """One-pole coefficient across one whole block."""
        tau_samples = max(1.0, time_constant_ms / 1000.0 * sample_rate)
        per_sample = 1.0 - math.exp(-1.0 / tau_samples)
        return 1.0 - (1.0 - per_sample) ** block_size

    def _smooth(self, ch: Channel, target: float) -> None:
        if target < ch.gain:
            coeff = self._coeff(ch.attack_ms, self.sample_rate,
                                self.block_size)
        else:
            coeff = self._coeff(ch.release_ms, self.sample_rate,
                                self.block_size)
        ch.gain += coeff * (target - ch.gain)
        # Fully ducked narration freezes its playhead; anything above the
        # floor (attacking or releasing) lets the playhead move again.
        ch.paused = (ch.pause_when_ducked
                     and target <= ch.duck_floor + 1e-9
                     and (ch.gain - ch.duck_floor)
                     < 0.15 * (ch.base_gain - ch.duck_floor))

    # ---------------------------------------------------------------- render
    def render(self, total_samples: int) -> list[float]:
        output = [0.0] * total_samples
        block = self.block_size
        for start in range(0, total_samples, block):
            self._fire_events(start)
            stop = min(start + block, total_samples)
            size = stop - start
            block_mix = [0.0] * size
            for ch in self.channels:
                target = self._duck_target(ch)
                self._smooth(ch, target)
                gain = ch.gain
                if not ch.paused:
                    for j in range(size):
                        if ch.position < len(ch.signal):
                            block_mix[j] += ch.signal[ch.position] * gain
                            ch.position += 1
                ch.history.append(((start + size / 2) / self.sample_rate,
                                   gain, 1.0 if ch.paused else 0.0))
            raw_peak = 0.0
            for j, sample in enumerate(block_mix):
                raw_peak = max(raw_peak, abs(sample))
                # Soft knee: tanh keeps peaks bounded without hard clicks.
                sample = math.tanh(sample / self.soft_clip) * self.soft_clip
                block_mix[j] = sample
                output[start + j] = sample
            clip_peak = max(abs(s) for s in block_mix)
            self.output_log.append(
                ((start + size / 2) / self.sample_rate, raw_peak, clip_peak))
        return output

    def _fire_events(self, sample_index: int) -> None:
        while self._events and self._events[0][0] <= sample_index:
            _, _, fn = heapq.heappop(self._events)
            fn()

    def reset(self) -> None:
        for ch in self.channels:
            ch.reset()
        self._events.clear()
        self.output_log.clear()
