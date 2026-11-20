"""FFmpeg-backed lossy encode/decode configurations."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodecConfig:
    key: str
    label: str
    encoder: str
    extension: str
    args: tuple[str, ...]
    target: str


CODEC_CONFIGS: tuple[CodecConfig, ...] = (
    CodecConfig(
        key="mp3",
        label="MP3 / LAME",
        encoder="libmp3lame",
        extension=".mp3",
        args=("-b:a", "128k"),
        target="128 kbps CBR",
    ),
    CodecConfig(
        key="vorbis",
        label="Vorbis",
        encoder="libvorbis",
        extension=".ogg",
        args=("-q:a", "4"),
        target="q4 / VBR",
    ),
    CodecConfig(
        key="opus",
        label="Opus",
        encoder="libopus",
        extension=".ogg",
        args=("-b:a", "96k", "-vbr", "on"),
        target="96 kbps VBR",
    ),
)


def require_ffmpeg() -> str:
    binary = shutil.which("ffmpeg")
    if not binary:
        raise RuntimeError("ffmpeg was not found on PATH")
    return binary


def _run(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{detail}"
        )


def _ffmpeg(*args: str) -> None:
    _run(
        [
            require_ffmpeg(),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            *args,
        ]
    )


def available_encoders() -> set[str]:
    binary = require_ffmpeg()
    completed = subprocess.run(
        [binary, "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    text = completed.stdout.decode("utf-8", errors="replace")
    encoders: set[str] = set()
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and len(fields[0]) == 6:
            encoders.add(fields[1])
    return encoders


def verify_codecs(codecs: tuple[CodecConfig, ...] = CODEC_CONFIGS) -> None:
    encoders = available_encoders()
    missing = [config.encoder for config in codecs if config.encoder not in encoders]
    if missing:
        raise RuntimeError(
            "FFmpeg is missing required lossy encoders: " + ", ".join(missing)
        )


def normalize_reference(
    source: str | Path, destination: str | Path, sample_rate: int
) -> None:
    _ffmpeg(
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(destination),
    )


def encode_lossy(
    reference_wav: str | Path,
    destination: str | Path,
    config: CodecConfig,
) -> None:
    _ffmpeg(
        "-i",
        str(reference_wav),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-c:a",
        config.encoder,
        *config.args,
        str(destination),
    )


def decode_lossy(
    encoded: str | Path,
    destination: str | Path,
    sample_rate: int,
) -> None:
    _ffmpeg(
        "-i",
        str(encoded),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(destination),
    )
