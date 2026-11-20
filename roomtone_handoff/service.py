"""Render all configured handoff versions and their manifests."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

from .errors import ConfigError
from .manifest import build_manifest, manifest_csv, manifest_json
from .model import HandoffConfig
from .render import RenderResult, render_version
from .wav import read_wav


@dataclass(frozen=True)
class VersionOutput:
    version: str
    wav_path: Path
    manifest_path: Path
    csv_path: Path
    result: RenderResult


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _check_output_paths(config: HandoffConfig) -> None:
    protected = {
        config.source_path.resolve(),
        config.room_tone_path.resolve(),
    }
    for version in config.versions:
        wav_path = (config.output_dir / f"{version.name}.wav").resolve()
        if wav_path in protected:
            raise ConfigError(
                f"version {version.name!r} would overwrite an input WAV; "
                "choose another output directory or version name"
            )


def render_all(config: HandoffConfig) -> tuple[VersionOutput, ...]:
    source = read_wav(config.source_path)
    room_tone = read_wav(config.room_tone_path)
    try:
        config.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigError(f"cannot create output directory {config.output_dir}: {exc}") from exc
    if not config.output_dir.is_dir():
        raise ConfigError(f"output path is not a directory: {config.output_dir}")
    _check_output_paths(config)
    outputs: list[VersionOutput] = []
    for version in config.versions:
        result = render_version(source, room_tone, version)
        document = build_manifest(
            result, config.output_dir, config.source_path, config.room_tone_path
        )
        output_paths = {
            "wav": config.output_dir / f"{version.name}.wav",
            "manifest": config.output_dir / f"{version.name}.manifest.json",
            "csv": config.output_dir / f"{version.name}.manifest.csv",
        }
        _atomic_write(output_paths["wav"], result.output_bytes)
        _atomic_write(
            output_paths["manifest"], manifest_json(document).encode("utf-8")
        )
        _atomic_write(
            output_paths["csv"], manifest_csv(document).encode("utf-8")
        )
        outputs.append(
            VersionOutput(
                version=version.name,
                wav_path=output_paths["wav"],
                manifest_path=output_paths["manifest"],
                csv_path=output_paths["csv"],
                result=result,
            )
        )
    return tuple(outputs)
