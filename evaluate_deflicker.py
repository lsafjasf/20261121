"""Run the deflicker on the test clips and report metrics.

Checks:
  1. Static-region brightness variance drops after correction.
  2. Motion edges of the moving foreground are preserved (gradient
     correlation with the clean ground truth stays high).
  3. A genuine global dimming ramp keeps its trend after correction.

Writes corrected videos and a metrics report into artifacts/.
"""

import os

import cv2
import numpy as np

from deflicker import DeflickerConfig, DeflickerProcessor

ART = os.path.join(os.path.dirname(__file__), "artifacts")
FPS = 30


def read_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    return frames


def write_video(path, frames):
    h, w = frames[0].shape
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h), False)
    for f in frames:
        writer.write(f)
    writer.release()


def static_series(frames, patch=(100, 125, 100, 300)):
    y0, y1, x0, x1 = patch
    return np.array([f[y0:y1, x0:x1].mean() for f in frames])


def edge_correlation(frames_a, frames_b, strip=(120, 200)):
    """Mean correlation of Sobel magnitude inside the mover strip."""
    y0, y1 = strip
    corrs = []
    for fa, fb in zip(frames_a, frames_b):
        ga = cv2.Sobel(fa[y0:y1], cv2.CV_32F, 1, 0, ksize=3)
        gb = cv2.Sobel(fb[y0:y1], cv2.CV_32F, 1, 0, ksize=3)
        a, b = ga.ravel() - ga.mean(), gb.ravel() - gb.mean()
        corrs.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
    return float(np.mean(corrs))


def trend_stats(frames):
    series = np.array([f.mean() for f in frames])
    t = np.arange(len(series), dtype=np.float32)
    slope, intercept = np.polyfit(t, series, 1)
    resid = series - (slope * t + intercept)
    return slope, float(resid.std()), series


def main():
    clean = read_video(os.path.join(ART, "flicker_clean.mp4"))
    flicker = read_video(os.path.join(ART, "flicker_input.mp4"))
    dimming = read_video(os.path.join(ART, "dimming_input.mp4"))

    proc = DeflickerProcessor(DeflickerConfig())
    fixed = proc.process(flicker)
    write_video(os.path.join(ART, "flicker_fixed.mp4"), fixed)
    roi = proc.diagnostics["ref_roi"]

    dim_proc = DeflickerProcessor(DeflickerConfig())
    dim_fixed = dim_proc.process(dimming)
    write_video(os.path.join(ART, "dimming_fixed.mp4"), dim_fixed)

    # 1. Static-region brightness variance
    var_in = static_series(flicker).var()
    var_out = static_series(fixed).var()
    var_clean = static_series(clean).var()

    # 2. Motion-edge preservation vs clean ground truth
    ec_in = edge_correlation(flicker, clean)
    ec_out = edge_correlation(fixed, clean)
    mae_in = np.mean([np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()
                      for a, b in zip(flicker, clean)])
    mae_out = np.mean([np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()
                       for a, b in zip(fixed, clean)])

    # 3. Dimming trend preservation
    slope_in, resid_in, _ = trend_stats(dimming)
    slope_out, resid_out, series_out = trend_stats(dim_fixed)

    report = [
        f"reference roi (auto): {roi}",
        "",
        "[1] static-region brightness variance",
        f"    input (flickered) : {var_in:10.4f}",
        f"    corrected         : {var_out:10.4f}  ({var_in / max(var_out, 1e-9):.1f}x reduction)",
        f"    clean reference   : {var_clean:10.4f}",
        "",
        "[2] motion-edge correlation with clean ground truth",
        f"    input vs clean    : {ec_in:.4f}",
        f"    corrected vs clean: {ec_out:.4f}",
        f"    MAE vs clean      : {mae_in:.3f} -> {mae_out:.3f}",
        "",
        "[3] global dimming trend preservation",
        f"    input slope / frame   : {slope_in:.4f}",
        f"    corrected slope /frame: {slope_out:.4f}  (ratio {slope_out / slope_in:.3f})",
        f"    detrended residual std: {resid_in:.4f} -> {resid_out:.4f}",
    ]
    text = "\n".join(report)
    print(text)
    with open(os.path.join(ART, "report.txt"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
