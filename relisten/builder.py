"""Build the per-speaker relisten package and its mapping manifest."""

from __future__ import annotations
import json
from pathlib import Path

from .audio import Track, write_wav
from .cutting import resolve_cut_points
from .merge import find_overlaps, merge_utterances
from .model import BuildConfig, Clip, Utterance


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_package(
    tracks: dict[str, Track],
    utterances: list[Utterance],
    output_dir: str | Path,
    config: BuildConfig | None = None,
) -> dict:
    config = config or BuildConfig()
    config.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = merge_utterances(utterances, config)
    clips: list[Clip] = []

    for speaker_id, groups in grouped.items():
        if speaker_id not in tracks:
            raise KeyError(
                f"no audio track provided for speaker {speaker_id!r}"
            )
        track = tracks[speaker_id]
        sample_rate = track.sample_rate
        cuts = resolve_cut_points(
            track.samples, sample_rate, groups, config
        )
        speaker_dir = output_dir / speaker_id

        for group_index, (group, (start_sample, end_sample)) in enumerate(
            zip(groups, cuts)
        ):
            clip_id = f"{speaker_id}_{group_index + 1:03d}"
            output_path = speaker_dir / f"{clip_id}.wav"
            write_wav(
                output_path,
                track.samples[start_sample:end_sample],
                sample_rate,
            )
            clip = Clip(
                clip_id=clip_id,
                speaker_id=speaker_id,
                group_index=group_index,
                source_path=str(track.path),
                sample_rate=sample_rate,
                source_start_sample=start_sample,
                source_end_sample=end_sample,
                clip_start_sample=0,
                clip_end_sample=end_sample - start_sample,
                start=start_sample / sample_rate,
                end=end_sample / sample_rate,
                utterance_indices=list(group.indices),
                merged_intervals=[
                    {"start": start, "end": end}
                    for start, end in group.intervals
                ],
                overlaps=find_overlaps(group, utterances),
                output_path=_relative(output_path, output_dir),
            )
            clips.append(clip)

    manifest = {
        "schema_version": 1,
        "config": config.__dict__,
        "speakers": sorted(grouped.keys()),
        "tracks": [
            {
                "speaker_id": speaker_id,
                "path": str(track.path),
                "sample_rate": track.sample_rate,
                "duration": track.duration,
                "samples": len(track.samples),
            }
            for speaker_id, track in sorted(tracks.items())
        ],
        "clips": [clip.to_dict() for clip in clips],
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest
