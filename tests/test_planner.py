from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_render.errors import MissingHandleError
from local_render.planner import build_plan

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def project_dir(tmp_path_factory):
    base = tmp_path_factory.mktemp("project")
    media = base / "media"
    project = base / "project"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/make_media.py"), str(media)],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/make_timeline.py"),
         str(project), str(media)],
        check=True,
    )
    return project


def test_demo_plan_timing(project_dir):
    plan = build_plan(project_dir / "demo.otio")
    assert plan.timeline_rate == 30.0
    assert plan.total_frames == 105
    assert [t.name for t in plan.video_tracks] == ["V1", "V2"]
    assert [t.name for t in plan.audio_tracks] == ["A1", "A2"]


def test_negative_source_start_becomes_black_prefix(project_dir):
    plan = build_plan(project_dir / "demo.otio")
    green = next(
        clip for track in plan.video_tracks for clip in track.clips
        if clip.name == "green"
    )
    assert green.black_prefix == 15
    assert green.media_start == 15
    assert green.timeline_start == 0


def test_mixed_source_rates_are_recorded(project_dir):
    plan = build_plan(project_dir / "demo.otio")
    rates = {
        clip.name: clip.source_rate
        for track in plan.video_tracks for clip in track.clips
    }
    assert rates["red"] == 24.0
    assert rates["green"] == 30.0


def test_dissolve_requires_real_handles(project_dir):
    plan = build_plan(project_dir / "demo.otio")
    transition = plan.video_tracks[0].transitions[0]
    assert transition.duration_frames == 30
    assert transition.start_frame == 45
    assert transition.cut_frame == 75


def test_missing_handle_is_rejected(project_dir):
    with pytest.raises(MissingHandleError) as excinfo:
        build_plan(project_dir / "missing_handle.otio")
    assert excinfo.value.details
    assert any("blue" in detail for detail in excinfo.value.details)


def test_manifest_weights_at_cut(project_dir):
    plan = build_plan(project_dir / "demo.otio")
    frame60 = next(
        entry for entry in plan.manifest["per_frame"]
        if entry["frame"] == 60
    )
    v1 = frame60["tracks"][0]["contributors"]
    weights = {item["source"]: item["weight"] for item in v1}
    assert weights == pytest.approx({"red": 0.5, "blue": 0.5}, abs=0.001)
