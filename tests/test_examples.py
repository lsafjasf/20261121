from itertools import chain

import pytest

from examples.generate_examples import (
    black_frames,
    encode,
    normal_lowlight_frames,
    repeated_frozen_frames,
)
from freeze_diag import DiagnosticConfig, analyze_video


@pytest.fixture(scope="module")
def videos(tmp_path_factory):
    root = tmp_path_factory.mktemp("clips")
    cases = {
        "normal": (normal_lowlight_frames(75), "400k"),
        "repeat": (repeated_frozen_frames(75), "45k"),
        "black": (black_frames(75), "20k"),
        "mixed": (
            chain(
                normal_lowlight_frames(50),
                repeated_frozen_frames(75),
                black_frames(50),
            ),
            "400k",
        ),
    }
    paths = {}
    for name, (frames, bitrate) in cases.items():
        path = root / f"{name}.mp4"
        encode(frames, path, 25, bitrate, name)
        paths[name] = path
    return paths


def test_normal_lowlight_is_not_a_freeze(videos):
    report = analyze_video(videos["normal"])
    assert report["verdict"] == "no_freeze_detected"
    assert report["freeze_intervals"] == []
    assert report["metrics"]["noise_above_floor_fraction"] > 0.9


def test_repeated_low_bitrate_frame_is_a_freeze(videos):
    report = analyze_video(videos["repeat"])
    assert report["verdict"] == "frozen_repeat_detected"
    assert len(report["freeze_intervals"]) == 1
    interval = report["freeze_intervals"][0]
    assert interval["evidence"]["duplicate_ratio"] == 1.0
    assert interval["evidence"]["noise_below_floor"] is True
    assert interval["evidence"]["timestamp_continuity"] == "continuous"


def test_all_black_is_indeterminate(videos):
    report = analyze_video(videos["black"])
    assert report["verdict"] == "indeterminate_blackout"
    assert report["freeze_intervals"] == []
    assert len(report["black_intervals"]) == 1
    assert report["black_intervals"][0]["evidence"]["determinable"] is False


def test_mixed_video_splits_freeze_and_black_intervals(videos):
    config = DiagnosticConfig(min_freeze_ms=500)
    report = analyze_video(videos["mixed"], config)
    assert report["verdict"] == "mixed_frozen_and_indeterminate_black"
    assert len(report["freeze_intervals"]) == 1
    assert len(report["black_intervals"]) == 1
    freeze = report["freeze_intervals"][0]
    black = report["black_intervals"][0]
    assert freeze["end_frame"] < black["start_frame"]
    assert freeze["evidence"]["timestamp_continuity"] == "continuous"
