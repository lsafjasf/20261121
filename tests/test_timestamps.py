from freeze_diag.timestamps import summarize_timestamps


def test_continuous_timestamps():
    pts = [float(value) for value in range(0, 400, 40)]
    summary = summarize_timestamps(pts, len(pts), 3.0, 0.2)
    assert summary.continuity == "continuous"
    assert summary.regular_ratio == 1.0
    assert summary.gap_count == 0
    assert summary.backstep_count == 0
    assert summary.nominal_delta_ms == 40.0


def test_gap_backstep_and_duplicate_pts():
    pts = [0.0, 40.0, 80.0, 200.0, 240.0, 240.0, 280.0]
    summary = summarize_timestamps(pts, len(pts), 3.0, 0.2)
    assert summary.continuity == "discontinuous"
    assert summary.gap_count == 1
    assert summary.backstep_count == 1
    assert summary.duplicate_pts_count == 1


def test_missing_timestamps_are_partial():
    pts = [0.0, 40.0, None, 120.0]
    summary = summarize_timestamps(pts, len(pts), 3.0, 0.2)
    assert summary.continuity == "partial"
    assert summary.missing_count == 1
