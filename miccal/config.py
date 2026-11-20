from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


class SceneError(ValueError):
    pass


Vec3 = tuple[float, float, float]


def _vec3(value: Any, name: str) -> Vec3:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise SceneError(f"{name} must be a three-element position")
    try:
        result = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise SceneError(f"{name} must contain numbers") from exc
    return result  # type: ignore[return-value]


def _floats(value: Any, count: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise SceneError(f"{name} must contain exactly {count} numbers")
    try:
        return tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise SceneError(f"{name} must contain numbers") from exc


def _ints(value: Any, count: int, name: str) -> tuple[int, ...]:
    values = _floats(value, count, name)
    result = tuple(int(v) for v in values)
    if any(v != f for v, f in zip(result, values)):
        raise SceneError(f"{name} must contain integers")
    return result


@dataclass(frozen=True)
class SweepConfig:
    duration_s: float = 0.26
    f_start: float = 250.0
    f_stop: float = 6500.0
    fade_s: float = 0.004
    noise_rms: float = 2e-6

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SweepConfig":
        data = data or {}
        cfg = cls(
            duration_s=float(data.get("duration_s", cls.duration_s)),
            f_start=float(data.get("f_start", cls.f_start)),
            f_stop=float(data.get("f_stop", cls.f_stop)),
            fade_s=float(data.get("fade_s", cls.fade_s)),
            noise_rms=float(data.get("noise_rms", cls.noise_rms)),
        )
        if cfg.duration_s <= 0 or cfg.f_stop <= cfg.f_start:
            raise SceneError("invalid sweep range or duration")
        return cfg


@dataclass(frozen=True)
class Wall:
    point: Vec3
    normal: Vec3
    reflection_amplitudes: tuple[float, float, float, float]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Wall":
        return cls(
            point=_vec3(data.get("point"), "wall.point"),
            normal=_vec3(data.get("normal"), "wall.normal"),
            reflection_amplitudes=_floats(
                data.get("reflection_amplitudes"),
                4,
                "wall.reflection_amplitudes",
            ),
        )


@dataclass(frozen=True)
class Source:
    name: str
    position: Vec3

    @classmethod
    def from_dict(cls, index: int, data: dict[str, Any]) -> "Source":
        return cls(
            name=str(data.get("name", f"source_{index + 1}")),
            position=_vec3(data.get("position"), f"sources[{index}].position"),
        )


@dataclass(frozen=True)
class Installation:
    microphones: tuple[Vec3, ...] | None = None
    channel_order: tuple[int, int, int, int] = (0, 1, 2, 3)
    gain: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    polarity: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Installation":
        data = data or {}
        mics = data.get("microphones")
        parsed_mics = (
            tuple(_vec3(m, f"installation.microphones[{i}]") for i, m in enumerate(mics))
            if mics is not None
            else None
        )
        order = _ints(data.get("channel_order", [0, 1, 2, 3]), 4, "installation.channel_order")
        if sorted(order) != [0, 1, 2, 3]:
            raise SceneError("installation.channel_order must be a permutation of 0..3")
        gain = _floats(data.get("gain", [1, 1, 1, 1]), 4, "installation.gain")
        polarity = _floats(data.get("polarity", [1, 1, 1, 1]), 4, "installation.polarity")
        if any(g <= 0 for g in gain):
            raise SceneError("installation.gain values must be positive")
        if any(p not in (-1.0, 1.0) for p in polarity):
            raise SceneError("installation.polarity values must be -1 or 1")
        return cls(parsed_mics, order, gain, polarity)


@dataclass(frozen=True)
class AnalysisConfig:
    max_delay_s: float = 0.015
    candidate_count: int = 6
    gain_warn_db: float = 1.0
    delay_warn_mm: float = 10.0
    map_tolerance_samples: float = 0.12
    peak_rel_threshold: float = 0.02
    beam_frequencies: tuple[float, ...] = (1200.0, 1800.0, 2500.0, 3500.0, 4800.0)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AnalysisConfig":
        data = data or {}
        freqs = tuple(float(f) for f in data.get("beam_frequencies", cls.beam_frequencies))
        if not freqs or any(f <= 0 for f in freqs):
            raise SceneError("analysis.beam_frequencies must contain positive values")
        return cls(
            max_delay_s=float(data.get("max_delay_s", cls.max_delay_s)),
            candidate_count=int(data.get("candidate_count", cls.candidate_count)),
            gain_warn_db=float(data.get("gain_warn_db", cls.gain_warn_db)),
            delay_warn_mm=float(data.get("delay_warn_mm", cls.delay_warn_mm)),
            map_tolerance_samples=float(
                data.get("map_tolerance_samples", cls.map_tolerance_samples)
            ),
            peak_rel_threshold=float(
                data.get("peak_rel_threshold", cls.peak_rel_threshold)
            ),
            beam_frequencies=freqs,
        )


@dataclass(frozen=True)
class Scene:
    sample_rate: int
    microphones: tuple[Vec3, ...]
    sources: tuple[Source, ...]
    sound_speed: float = 343.0
    reference_channel: int = 0
    sweep: SweepConfig = field(default_factory=SweepConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    wall: Wall | None = None
    installation: Installation = field(default_factory=Installation)
    recording_seconds: float = 0.58
    name: str = "scene"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scene":
        try:
            mics = tuple(
                _vec3(m, f"microphones[{i}]")
                for i, m in enumerate(data.get("microphones", []))
            )
            sources = tuple(
                Source.from_dict(i, s)
                for i, s in enumerate(data.get("sources", []))
            )
            scene = cls(
                sample_rate=int(data.get("sample_rate", 16000)),
                microphones=mics,
                sources=sources,
                sound_speed=float(data.get("sound_speed", 343.0)),
                reference_channel=int(data.get("reference_channel", 0)),
                sweep=SweepConfig.from_dict(data.get("sweep")),
                analysis=AnalysisConfig.from_dict(data.get("analysis")),
                wall=Wall.from_dict(data["wall"]) if data.get("wall") else None,
                installation=Installation.from_dict(data.get("installation")),
                recording_seconds=float(data.get("recording_seconds", 0.58)),
                name=str(data.get("name", "scene")),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, SceneError):
                raise
            raise SceneError(str(exc)) from exc
        scene.validate()
        return scene

    def validate(self) -> None:
        if len(self.microphones) != 4:
            raise SceneError("exactly four microphone positions are required")
        if not self.sources:
            raise SceneError("at least one source position is required")
        if self.sample_rate < 8000:
            raise SceneError("sample_rate must be at least 8000 Hz")
        if not 0 <= self.reference_channel < 4:
            raise SceneError("reference_channel must be in 0..3")
        if self.recording_seconds <= self.sweep.duration_s:
            raise SceneError("recording_seconds must exceed sweep.duration_s")
        if self.installation.microphones is not None:
            if len(self.installation.microphones) != 4:
                raise SceneError("installation.microphones must contain four positions")
        if self.wall is not None:
            nx, ny, nz = self.wall.normal
            if nx * nx + ny * ny + nz * nz < 1e-12:
                raise SceneError("wall.normal cannot be zero")
        for source in self.sources:
            for mic in self.microphones:
                if _distance(source.position, mic) < 1e-6:
                    raise SceneError("source and microphone positions must differ")

    @property
    def recording_samples(self) -> int:
        return int(round(self.recording_seconds * self.sample_rate))

    def installed_microphones(self) -> tuple[Vec3, ...]:
        return self.installation.microphones or self.microphones

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sample_rate": self.sample_rate,
            "sound_speed": self.sound_speed,
            "recording_seconds": self.recording_seconds,
            "reference_channel": self.reference_channel,
            "microphones": [list(m) for m in self.microphones],
            "sources": [
                {"name": source.name, "position": list(source.position)}
                for source in self.sources
            ],
            "sweep": {
                "duration_s": self.sweep.duration_s,
                "f_start": self.sweep.f_start,
                "f_stop": self.sweep.f_stop,
                "fade_s": self.sweep.fade_s,
                "noise_rms": self.sweep.noise_rms,
            },
            "analysis": {
                "max_delay_s": self.analysis.max_delay_s,
                "candidate_count": self.analysis.candidate_count,
                "gain_warn_db": self.analysis.gain_warn_db,
                "delay_warn_mm": self.analysis.delay_warn_mm,
                "map_tolerance_samples": self.analysis.map_tolerance_samples,
                "peak_rel_threshold": self.analysis.peak_rel_threshold,
                "beam_frequencies": list(self.analysis.beam_frequencies),
            },
            "wall": (
                {
                    "point": list(self.wall.point),
                    "normal": list(self.wall.normal),
                    "reflection_amplitudes": list(self.wall.reflection_amplitudes),
                }
                if self.wall
                else None
            ),
            "installation": {
                "microphones": (
                    [list(m) for m in self.installation.microphones]
                    if self.installation.microphones
                    else None
                ),
                "channel_order": list(self.installation.channel_order),
                "gain": list(self.installation.gain),
                "polarity": list(self.installation.polarity),
            },
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.as_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "Scene":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SceneError(f"cannot load scene {path}: {exc}") from exc
        return cls.from_dict(data)


def _distance(a: Vec3, b: Vec3) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
