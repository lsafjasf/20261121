from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
import sys
from pathlib import Path
import wave

from .config import SweepConfig


def next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def fft(values: list[complex], inverse: bool = False) -> list[complex]:
    n = len(values)
    if n == 0 or n & (n - 1):
        raise ValueError("FFT length must be a power of two")
    output = list(values)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            output[i], output[j] = output[j], output[i]
    length = 2
    sign = 1.0 if inverse else -1.0
    while length <= n:
        angle = sign * 2.0 * math.pi / length
        root = complex(math.cos(angle), math.sin(angle))
        half = length // 2
        for start in range(0, n, length):
            twiddle = 1.0 + 0.0j
            for offset in range(half):
                even = output[start + offset]
                odd = output[start + offset + half] * twiddle
                output[start + offset] = even + odd
                output[start + offset + half] = even - odd
                twiddle *= root
        length <<= 1
    if inverse:
        scale = 1.0 / n
        for index in range(n):
            output[index] *= scale
    return output


def ifft(values: list[complex]) -> list[complex]:
    return fft(values, inverse=True)


def real_fft(values: list[float], n: int) -> list[complex]:
    return fft([complex(value, 0.0) for value in values] + [0j] * (n - len(values)))


def exponential_sweep(config: SweepConfig, sample_rate: int) -> list[float]:
    count = int(round(config.duration_s * sample_rate))
    fade = min(int(round(config.fade_s * sample_rate)), count // 4)
    ratio = config.f_stop / config.f_start
    rate = math.log(ratio)
    output: list[float] = []
    for index in range(count):
        time = index / sample_rate
        phase = (
            2.0
            * math.pi
            * config.f_start
            * config.duration_s
            / rate
            * (math.exp(time / config.duration_s * rate) - 1.0)
        )
        value = math.sin(phase)
        if fade:
            if index < fade:
                value *= 0.5 - 0.5 * math.cos(math.pi * index / fade)
            remaining = count - 1 - index
            if remaining < fade:
                value *= 0.5 - 0.5 * math.cos(math.pi * remaining / fade)
        output.append(value)
    return output


def deconvolve_sweep(
    recording: list[float],
    sweep: list[float],
    sample_rate: int,
    f_start: float,
    f_stop: float,
    regularization: float = 2e-8,
) -> list[float]:
    size = next_power_of_two(len(recording) + len(sweep) - 1)
    y_spectrum = real_fft(recording, size)
    x_spectrum = real_fft(sweep, size)
    peak_power = max(abs(value) ** 2 for value in x_spectrum)
    floor = max(peak_power * regularization, 1e-30)
    output_spectrum: list[complex] = []
    nyquist = size // 2
    for index in range(size):
        frequency = index * sample_rate / size
        in_band = f_start * 0.9 <= frequency <= min(f_stop * 1.04, sample_rate * 0.49)
        if not in_band:
            output_spectrum.append(0j)
            continue
        x_value = x_spectrum[index]
        denominator = abs(x_value) ** 2 + floor
        output_spectrum.append(y_spectrum[index] * x_value.conjugate() / denominator)
    output_spectrum[0] = 0j
    output_spectrum[nyquist] = 0j
    ir = ifft(output_spectrum)
    return [value.real for value in ir]


def analytic_envelope(signal: list[float]) -> list[float]:
    n = next_power_of_two(len(signal))
    spectrum = real_fft(signal, n)
    for index in range(1, n // 2):
        spectrum[index] *= 2.0
    for index in range(n // 2 + 1, n):
        spectrum[index] = 0j
    analytic = ifft(spectrum)
    return [abs(value) for value in analytic[: len(signal)]]


@dataclass(frozen=True)
class Peak:
    index: int
    delay_s: float
    signed_amplitude: float
    envelope: float
    energy: float

    @property
    def polarity(self) -> float:
        return 1.0 if self.signed_amplitude >= 0 else -1.0


def detect_peaks(
    signal: list[float],
    sample_rate: int,
    max_delay_s: float,
    min_separation_s: float = 0.0009,
    count: int = 6,
    relative_threshold: float = 0.02,
) -> list[Peak]:
    search_stop = min(len(signal), int(round(max_delay_s * sample_rate)))
    if search_stop < 5:
        return []
    envelope = analytic_envelope(signal[:search_stop])
    global_max = max(envelope, default=0.0)
    if global_max <= 0:
        return []
    floor_values = sorted(envelope[: max(5, int(0.002 * sample_rate))])
    noise_floor = floor_values[len(floor_values) // 2] if floor_values else 0.0
    threshold = max(global_max * relative_threshold, noise_floor * 6.0)
    min_separation = max(2, int(round(min_separation_s * sample_rate)))
    local: list[tuple[float, int]] = []
    for index in range(2, len(envelope) - 2):
        value = envelope[index]
        if value < threshold:
            continue
        if value >= envelope[index - 1] and value >= envelope[index + 1]:
            local.append((value, index))
    local.sort(reverse=True)
    selected: list[tuple[float, int]] = []
    for value, index in local:
        if all(abs(index - other) >= min_separation for _, other in selected):
            selected.append((value, index))
            if len(selected) >= count:
                break
    selected.sort(key=lambda item: item[1])
    peaks: list[Peak] = []
    for value, index in selected:
        left = envelope[index - 1]
        center = envelope[index]
        right = envelope[index + 1]
        denominator = left - 2.0 * center + right
        offset = (
            0.5 * (left - right) / denominator
            if abs(denominator) > 1e-12
            else 0.0
        )
        offset = max(-0.5, min(0.5, offset))
        energy_radius = max(2, int(round(0.0011 * sample_rate)))
        energy = math.sqrt(
            sum(
                signal[position] * signal[position]
                for position in range(
                    max(0, index - energy_radius),
                    min(len(signal), index + energy_radius + 1),
                )
            )
        )
        peaks.append(
            Peak(
                index=index,
                delay_s=(index + offset) / sample_rate,
                signed_amplitude=signal[index],
                envelope=value,
                energy=energy,
            )
        )
    return peaks


def read_wav(path: str | Path) -> tuple[list[list[float]], int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder != "little":
        samples.byteswap()
    output = [[0.0] * (len(samples) // channels) for _ in range(channels)]
    for index, value in enumerate(samples):
        output[index % channels][index // channels] = value / 32768.0
    return output, sample_rate


def write_wav(
    path: str | Path,
    channels: list[list[float]],
    sample_rate: int,
) -> None:
    if not channels or len({len(channel) for channel in channels}) != 1:
        raise ValueError("all WAV channels must have the same non-zero length")
    interleaved = array("h")
    for frame in range(len(channels[0])):
        for channel in channels:
            value = max(-1.0, min(1.0, channel[frame]))
            interleaved.append(int(round(value * 32767.0)))
    if sys.byteorder != "little":
        interleaved.byteswap()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(len(channels))
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(interleaved.tobytes())
