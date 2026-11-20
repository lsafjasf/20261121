"""Build the demo OTIO timeline and the missing-handle counter example."""

from __future__ import annotations

import argparse
from pathlib import Path

import opentimelineio as otio
from opentimelineio.opentime import RationalTime, TimeRange

TIMELINE_RATE = 30.0
WIDTH = 320
HEIGHT = 180


def _time(value: float, rate: float) -> RationalTime:
    return RationalTime(round(value * rate), rate)


def _reference(media_dir: Path, name: str, fps: float, frames: int):
    ref = otio.schema.ExternalReference(
        target_url=str((media_dir / f"{name}.mkv").resolve())
    )
    ref.available_range = TimeRange(
        start_time=RationalTime(0, fps), duration=RationalTime(frames, fps)
    )
    ref.metadata["local_render"] = {"width": WIDTH, "height": HEIGHT}
    return ref


def _clip(name, ref, start_seconds, duration_seconds):
    clip = otio.schema.Clip(name=name, media_reference=ref)
    clip.source_range = TimeRange(
        start_time=_time(start_seconds, TIMELINE_RATE),
        duration=_time(duration_seconds, TIMELINE_RATE),
    )
    return clip


def _dissolve(in_frames: int, out_frames: int):
    transition = otio.schema.Transition(
        name=f"dissolve_{in_frames + out_frames}f",
        transition_type=otio.schema.TransitionTypes.SMPTE_Dissolve,
    )
    transition.in_offset = RationalTime(in_frames, TIMELINE_RATE)
    transition.out_offset = RationalTime(out_frames, TIMELINE_RATE)
    return transition


def _build(media_dir: Path, missing_handle: bool = False):
    red_ref = _reference(media_dir, "red", 24.0, 120)
    blue_ref = _reference(media_dir, "blue", 24.0, 72)
    green_ref = _reference(media_dir, "green", 30.0, 60)

    dissolve_frames = 30
    out_frames = 30 if missing_handle else 0
    red_visible = _clip("red", red_ref, 0.25, 2.5)
    blue_start = 0.0
    blue_visible = _clip("blue", blue_ref, blue_start, 1.0)
    green_clip = _clip("green", green_ref, -0.5, 2.5)

    timeline = otio.schema.Timeline(name="demo", global_start_time=RationalTime(0, TIMELINE_RATE))
    v1 = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    v2 = otio.schema.Track(name="V2", kind=otio.schema.TrackKind.Video)
    a1 = otio.schema.Track(name="A1", kind=otio.schema.TrackKind.Audio)
    a2 = otio.schema.Track(name="A2", kind=otio.schema.TrackKind.Audio)
    v1.append(red_visible)
    v1.append(_dissolve(dissolve_frames, out_frames))
    v1.append(blue_visible)
    v2.append(green_clip)
    a1.append(_clip("red", red_ref, 0.25, 2.5))
    a1.append(_dissolve(dissolve_frames, out_frames))
    a1.append(_clip("blue", blue_ref, blue_start, 1.0))
    a2.append(_clip("green", green_ref, -0.5, 2.5))
    timeline.tracks.extend([v1, v2, a1, a2])
    return timeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_dir", help="directory for .otio files")
    parser.add_argument("media_dir", help="directory holding generated media")
    args = parser.parse_args()
    project_dir = Path(args.project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    media_dir = Path(args.media_dir)
    demo = _build(media_dir)
    demo_path = project_dir / "demo.otio"
    otio.adapters.write_to_file(demo, str(demo_path))
    bad = _build(media_dir, missing_handle=True)
    bad.name = "missing_handle"
    bad_path = project_dir / "missing_handle.otio"
    otio.adapters.write_to_file(bad, str(bad_path))
    print(f"wrote {demo_path} and {bad_path}")


if __name__ == "__main__":
    main()
