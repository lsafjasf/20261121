"""Dependency-free FFT convolution and normalized cross-correlation."""

from __future__ import annotations

from dataclasses import dataclass
import math


def next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def fft(values: list[complex] | list[float], inverse: bool = False) -> list[complex]:
    """Return the radix-2 Cooley-Tukey FFT of a power-of-two length input."""

    size = len(values)
    if size == 0 or size & (size - 1):
        raise ValueError("FFT length must be a positive power of two")
    result = [complex(value) for value in values]
    bit = 0
    for index in range(1, size):
        target = size >> 1
        while bit & target:
            bit ^= target
            target >>= 1
        bit ^= target
        if index < bit:
            result[index], result[bit] = result[bit], result[index]

    length = 2
    direction = 1.0 if inverse else -1.0
    while length <= size:
        angle = direction * 2.0 * math.pi / length
        step = complex(math.cos(angle), math.sin(angle))
        half = length >> 1
        for start in range(0, size, length):
            twiddle = 1.0 + 0.0j
            stop = start + half
            for left in range(start, stop):
                right = left + half
                even = result[left]
                odd = result[right] * twiddle
                result[left] = even + odd
                result[right] = even - odd
                twiddle *= step
        length <<= 1

    if inverse:
        scale = 1.0 / size
        return [value * scale for value in result]
    return result


def _remove_mean(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    return [value - mean for value in values]


def _energy(values: list[float]) -> float:
    return sum(value * value for value in values)


def fft_convolve(signal: list[float], impulse_response: list[float]) -> list[float]:
    """Convolve two real sequences without external numerical libraries."""

    if not signal or not impulse_response:
        return []
    output_length = len(signal) + len(impulse_response) - 1
    fft_length = next_power_of_two(output_length)
    left = fft(signal + [0.0] * (fft_length - len(signal)))
    right = fft(impulse_response + [0.0] * (fft_length - len(impulse_response)))
    product = [a * b for a, b in zip(left, right)]
    transformed = fft(product, inverse=True)
    return [value.real for value in transformed[:output_length]]


@dataclass(frozen=True)
class CorrelationPeak:
    signed_score: float
    lag_samples: int
    lag_seconds: float
    overlap_ratio: float
    secondary_score: float
    peak_to_secondary_db: float
    noise_floor: float
    peak_to_noise: float
    polarity: int


def _valid_bounds(nx: int, ny: int, lag: int) -> tuple[int, int] | None:
    if lag >= 0:
        start = 0
        stop = min(nx, ny - lag)
    else:
        start = -lag
        stop = min(nx, ny - lag)
    if stop <= start:
        return None
    return start, stop


def normalized_xcorr_peak(
    reference: list[float],
    recording: list[float],
    sample_rate: int,
    min_overlap: float = 0.25,
    max_lag_seconds: float | None = None,
    exclusion_seconds: float = 0.002,
) -> CorrelationPeak:
    """Find the strongest normalized cross-correlation peak and its side lobe."""

    if not reference or not recording:
        raise ValueError("correlation inputs must not be empty")
    x = _remove_mean(reference)
    y = _remove_mean(recording)
    nx, ny = len(x), len(y)
    norm = math.sqrt(_energy(x) * _energy(y))
    if norm <= 1e-20:
        return CorrelationPeak(0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    fft_length = next_power_of_two(nx + ny - 1)
    x_spectrum = fft(x + [0.0] * (fft_length - nx))
    y_spectrum = fft(y + [0.0] * (fft_length - ny))
    product = [value.conjugate() * other for value, other in zip(x_spectrum, y_spectrum)]
    curve = fft(product, inverse=True)

    minimum_overlap = max(1, int(math.ceil(min_overlap * min(nx, ny))))
    maximum_lag = (
        int(round(max_lag_seconds * sample_rate))
        if max_lag_seconds is not None
        else max(nx, ny)
    )
    maximum_lag = min(maximum_lag, max(nx - 1, ny - 1))
    best: tuple[float, float, int, float] | None = None
    valid_count = 0
    for lag in range(-maximum_lag, maximum_lag + 1):
        bounds = _valid_bounds(nx, ny, lag)
        if bounds is None:
            continue
        overlap_length = bounds[1] - bounds[0]
        if overlap_length < minimum_overlap:
            continue
        index = lag if lag >= 0 else fft_length + lag
        normalized = curve[index].real / norm
        candidate = (
            abs(normalized),
            normalized,
            lag,
            overlap_length / min(nx, ny),
        )
        valid_count += 1
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return CorrelationPeak(0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    best_abs, best_signed, best_lag, best_overlap = best
    exclusion = max(1, int(round(exclusion_seconds * sample_rate)))
    second_abs = 0.0
    stride = max(1, valid_count // 4096)
    sampled: list[float] = []
    seen = 0
    for lag in range(-maximum_lag, maximum_lag + 1):
        bounds = _valid_bounds(nx, ny, lag)
        if bounds is None:
            continue
        if bounds[1] - bounds[0] < minimum_overlap:
            continue
        index = lag if lag >= 0 else fft_length + lag
        magnitude = abs(curve[index].real / norm)
        if abs(lag - best_lag) > exclusion and magnitude > second_abs:
            second_abs = magnitude
        seen += 1
        if seen % stride == 0:
            sampled.append(magnitude)
    sampled.sort()
    median = sampled[len(sampled) // 2] if sampled else 0.0
    noise_floor = max(median, 1e-12)
    ratio = best_abs / max(second_abs, 1e-12)
    peak_to_secondary_db = 20.0 * math.log10(max(ratio, 1.0))
    return CorrelationPeak(
        signed_score=best_signed,
        lag_samples=best_lag,
        lag_seconds=best_lag / sample_rate,
        overlap_ratio=best_overlap,
        secondary_score=second_abs,
        peak_to_secondary_db=peak_to_secondary_db,
        noise_floor=noise_floor,
        peak_to_noise=best_abs / noise_floor,
        polarity=1 if best_signed >= 0.0 else -1,
    )
