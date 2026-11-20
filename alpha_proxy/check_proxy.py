"""Verify a transparent proxy against its straight-alpha master.

Decodes the proxy, composites it over BLACK and WHITE backgrounds under
both interpretations (straight / premultiplied), compares against the
master composited the same way, and measures edge-band error.

The interpretation giving the lower error is reported as the proxy's
actual encoding; if it disagrees with the declared interpretation
(sidecar .json from make_proxy.py or --declared), a mismatch is flagged.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from common import (composite, edge_mask, ffmpeg_decode_rgba,
                    load_png_sequence, to_float, to_u8)

INTERPRETATIONS = ("straight", "premultiplied")
PASS_THRESHOLD = 2.0 / 255.0  # mean abs error in the edge band


def _save_png(rgb_u8, path):
    h, w, _ = rgb_u8.shape
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-i", "-", "-frames:v", "1", str(path)]
    proc = subprocess.run(cmd, input=rgb_u8.tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode())


def _comp_errors(proxy_f, master_f, bg_value, mask):
    """Mean/max abs error of proxy comps vs the ground-truth composite.

    The master is straight by definition, so the true composite is
    master-as-straight; the proxy is composited under each interpretation
    and compared against that single truth.
    """
    bg = np.full(proxy_f.shape[:-1] + (3,), bg_value, dtype=np.float32)
    truth = composite(master_f, bg, "straight")
    results = {}
    diffs = {}
    for interp in INTERPRETATIONS:
        comp_p = composite(proxy_f, bg, interp)
        diff = np.abs(comp_p - truth)
        diffs[interp] = diff
        band = diff[:, mask, :]  # (n, n_edge_px, 3)
        results[interp] = {
            "edge_mae": float(band.mean()),
            "edge_max": float(band.max()),
            "full_mae": float(diff.mean()),
        }
    return results, diffs


def check_proxy(proxy_path, master_dir, declared=None, report_dir=None):
    proxy_path = Path(proxy_path)
    proxy_f = to_float(ffmpeg_decode_rgba(proxy_path))
    master_f = to_float(load_png_sequence(master_dir))
    n = min(proxy_f.shape[0], master_f.shape[0])
    proxy_f, master_f = proxy_f[:n], master_f[:n]

    mask = edge_mask(master_f[..., 3].mean(axis=0))
    edge_fraction = float(mask.mean())
    if not mask.any():
        mask = np.ones(mask.shape, dtype=bool)

    black, black_diffs = _comp_errors(proxy_f, master_f, 0.0, mask)
    white, white_diffs = _comp_errors(proxy_f, master_f, 1.0, mask)

    score = {i: (black[i]["edge_mae"] + white[i]["edge_mae"]) / 2.0
             for i in INTERPRETATIONS}
    actual = min(score, key=score.get)
    other = "premultiplied" if actual == "straight" else "straight"
    ratio = score[other] / max(score[actual], 1e-9)

    # Hidden-RGB survival where the master is fully transparent. When alpha
    # is binary (no semi-transparent edge pixels), compositing cannot tell
    # the interpretations apart; the hidden RGB can: a straight proxy keeps
    # it, a premultiplied proxy has it multiplied down to (near) zero.
    transparent = master_f[..., 3].mean(axis=0) < 1e-4
    hidden_rgb_mae = None
    hidden_vote = None
    if transparent.any():
        hidden_rgb_mae = float(
            np.abs(proxy_f[..., :3] - master_f[..., :3])[:, transparent, :].mean())
        master_hidden = float(master_f[..., :3][:, transparent, :].mean())
        if master_hidden > 4.0 / 255.0:
            err_straight = hidden_rgb_mae
            err_premult = float(
                np.abs(proxy_f[..., :3])[:, transparent, :].mean())
            hidden_vote = ("straight" if err_straight <= err_premult
                           else "premultiplied")

    method = "composite"
    if hidden_vote is not None and edge_fraction < 0.001:
        actual = hidden_vote
        method = "hidden_rgb"
        ratio = float("inf")

    if declared is None:
        sidecar = proxy_path.with_suffix(".json")
        if sidecar.exists():
            declared = json.loads(sidecar.read_text()).get("interpretation")

    report = {
        "proxy": str(proxy_path),
        "master": str(master_dir),
        "frames_compared": int(n),
        "edge_pixels": int(mask.sum()),
        "declared_interpretation": declared,
        "detected_interpretation": actual,
        "detection_method": method,
        "detection_confidence_ratio": (round(ratio, 2)
                                       if ratio != float("inf") else "inf"),
        "edge_error_black_bg": black,
        "edge_error_white_bg": white,
        "hidden_rgb_mae_fully_transparent": hidden_rgb_mae,
        "mismatch": declared is not None and declared != actual,
        "pass": score[actual] < PASS_THRESHOLD,
    }

    if report_dir:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "report.json").write_text(json.dumps(report, indent=2))
        for interp in INTERPRETATIONS:
            worst = np.maximum(black_diffs[interp], white_diffs[interp])
            heat = to_u8(np.clip(worst.max(axis=0) * 4.0, 0, 1))
            _save_png(heat, report_dir / f"edge_error_{interp}.png")
    return report


def _fmt(metrics):
    return (f"edge MAE {metrics['edge_mae']*255:6.2f}/255  "
            f"edge max {metrics['edge_max']*255:6.1f}/255  "
            f"full MAE {metrics['full_mae']*255:6.2f}/255")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proxy", required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--declared", choices=list(INTERPRETATIONS), default=None,
                    help="override sidecar .json declaration")
    ap.add_argument("--report-dir", default=None,
                    help="write report.json and edge-error heatmaps here")
    args = ap.parse_args()
    r = check_proxy(args.proxy, args.master, args.declared, args.report_dir)

    print(f"proxy:    {r['proxy']}")
    print(f"master:   {r['master']}  ({r['frames_compared']} frames, "
          f"{r['edge_pixels']} edge px)")
    for interp in INTERPRETATIONS:
        print(f"  as {interp:13s} black bg: {_fmt(r['edge_error_black_bg'][interp])}")
        print(f"  as {interp:13s} white bg: {_fmt(r['edge_error_white_bg'][interp])}")
    if r["hidden_rgb_mae_fully_transparent"] is not None:
        print(f"  hidden RGB @ alpha=0 MAE: "
              f"{r['hidden_rgb_mae_fully_transparent']*255:.2f}/255")
    print(f"detected interpretation: {r['detected_interpretation']} "
          f"(method: {r['detection_method']}, "
          f"confidence x{r['detection_confidence_ratio']})")
    if r["declared_interpretation"]:
        if r["mismatch"]:
            print(f"MISMATCH: declared '{r['declared_interpretation']}' but the "
                  f"proxy decodes as '{r['detected_interpretation']}' - "
                  f"wrong interpretation!")
        else:
            print(f"declared interpretation '{r['declared_interpretation']}' "
                  f"matches the decode - OK")
    ok = r["pass"] and not r["mismatch"]
    if not r["pass"]:
        print("verdict: FAIL (edge error above threshold even with the "
              "best interpretation)")
    elif r["mismatch"]:
        print("verdict: FAIL (interpretation mismatch)")
    else:
        print("verdict: PASS")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
