"""End-to-end checks for beatswap. Run: python3 test_beatswap.py"""

import json
import os
import subprocess
import sys

import numpy as np

from beatswap import audio, cli, features

OUT = "out_test"


def setup_module():
    subprocess.run([sys.executable, "make_test_track.py"], check=True)


def run_cli():
    if os.path.isdir(OUT):
        import shutil
        shutil.rmtree(OUT)
    return cli.run("testdata/A.wav", "testdata/B.wav", OUT, bars=8, top_k=5)


def test_tempo_and_beats():
    xa, sr = audio.load_wav("testdata/A.wav")
    xb, sr = audio.load_wav("testdata/B.wav")
    xa, _ = audio.normalize_loudness(xa)
    xb, _ = audio.normalize_loudness(xb)
    ea, fps = features.onset_envelope(xa, sr)
    eb, _ = features.onset_envelope(xb, sr)
    bpm_a = features.estimate_tempo(ea, fps)
    bpm_b = features.estimate_tempo(eb, fps)
    assert abs(bpm_a - 120.0) < 2.0, bpm_a
    assert abs(bpm_b - 122.4) < 2.0, bpm_b


def test_candidates_and_warnings():
    payload = run_cli()
    cands = payload["candidates"]
    # multiple equivalent loops -> multiple candidates
    assert len(cands) >= 3, f"expected several candidates, got {len(cands)}"
    clean = [c for c in cands if not c["warnings"]]
    flagged = [c for c in cands if c["warnings"]]
    assert len(clean) >= 2, "expected multiple clean equivalent loops"
    assert len(flagged) >= 1, "inserted extra beat was not flagged"
    # the flagged window must cover B's inserted beat (bar 20, ~39.2s in B)
    b_extra_s = 20 * 4 * 60.0 / 122.4
    covers = [c for c in flagged
              if c["b_entry_s"] - 1.0 < b_extra_s < c["b_exit_s"] + 1.0]
    assert covers, "flagged window does not cover the inserted beat"
    assert any("beat_count_mismatch" in w for w in covers[0]["warnings"])
    # clean candidates must score clearly higher than the flagged one
    assert min(c["score"] for c in clean) > max(c["score"] for c in flagged)


def test_previews_spliced_correctly():
    payload = run_cli()
    sr = audio.TARGET_SR
    xa, _ = audio.load_wav("testdata/A.wav")
    xb, _ = audio.load_wav("testdata/B.wav")
    xa, _ = audio.normalize_loudness(xa)
    xb, _ = audio.normalize_loudness(xb)
    cand = next(c for c in payload["candidates"] if not c["warnings"])
    rank = payload["candidates"].index(cand) + 1
    prev = os.path.join(OUT, f"preview{rank}_A_with_B_middle.wav")
    assert os.path.exists(prev)
    p, _ = audio.load_wav(prev)
    p, _ = audio.normalize_loudness(p)

    def seg(x, t0, t1):
        return x[int(t0 * sr):int(t1 * sr)]

    def ncc_lag(u, v, max_lag=3000):
        n = min(len(u), len(v)) - max_lag - 1
        best = -2.0
        for lag in range(-max_lag, max_lag, 7):
            a = u[max_lag:max_lag + n]
            b = v[max_lag + lag:max_lag + lag + n]
            m = min(len(a), len(b))
            best = max(best, float(np.corrcoef(a[:m], b[:m])[0, 1]))
        return best

    e0, e1 = cand["a_entry_s"], cand["a_exit_s"]
    b0, b1 = cand["b_entry_s"], cand["b_exit_s"]
    mid_dur = b1 - b0
    assert ncc_lag(seg(p, 0.5, e0 - 0.5), seg(xa, 0.5, e0 - 0.5)) > 0.8
    assert ncc_lag(seg(p, e0 - 0.1, e0 + mid_dur - 0.5),
                   seg(xb, b0, b1 - 0.4)) > 0.8
    assert ncc_lag(seg(p, e0 + mid_dur + 0.5, e0 + mid_dur + 5),
                   seg(xa, e1 + 0.5, e1 + 5)) > 0.8
    clip = prev.replace(".wav", "_joint_entry.wav")
    x, _ = audio.load_wav(clip)
    assert 3.0 < len(x) / sr < 5.0


def test_evidence_files():
    run_cli()
    import xml.etree.ElementTree as ET
    for f in ("evidence_envelopes.svg", "evidence_dtw.svg",
              "evidence_scores.svg"):
        ET.parse(os.path.join(OUT, f))
    report = json.load(open(os.path.join(OUT, "report.json")))
    assert report["diagnostics"]["tempo_ratio_a_over_b"] > 1.0


if __name__ == "__main__":
    setup_module()
    test_tempo_and_beats()
    print("ok: tempo and beats")
    test_candidates_and_warnings()
    print("ok: candidates and mis-match warnings")
    test_previews_spliced_correctly()
    print("ok: previews spliced correctly")
    test_evidence_files()
    print("ok: evidence files")
    print("ALL TESTS PASSED")
