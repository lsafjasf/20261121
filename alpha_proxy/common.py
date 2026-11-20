"""Shared helpers: ffmpeg rawvideo IO, checkerboard, compositing, metrics."""
import json
import subprocess
from pathlib import Path

import numpy as np

FPS = 24


def ffmpeg_encode_rgba(frames, out_path, codec="prores"):
    """frames: uint8 (n,h,w,4) RGBA, encoded as-is (alpha preserved)."""
    n, h, w, _ = frames.shape
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if codec == "prores":
        codec_args = ["-c:v", "prores_ks", "-profile:v", "4444",
                      "-pix_fmt", "yuva444p10le", "-vendor", "apl0"]
    elif codec == "qtrle":
        codec_args = ["-c:v", "qtrle", "-pix_fmt", "argb"]
    elif codec == "png":
        codec_args = ["-c:v", "png", "-pix_fmt", "rgba"]
    else:
        raise ValueError(codec)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
           "-r", str(FPS), "-i", "-",
           *codec_args, "-frames:v", str(n), str(out_path)]
    proc = subprocess.run(cmd, input=frames.tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode())


def ffmpeg_decode_rgba(path):
    """Decode any video file to uint8 (n,h,w,4) RGBA."""
    path = str(path)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", path],
        capture_output=True, check=True)
    info = json.loads(probe.stdout)["streams"][0]
    w, h = info["width"], info["height"]
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
         "-f", "rawvideo", "-pix_fmt", "rgba", "-"],
        capture_output=True, check=True)
    buf = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = buf.size // (w * h * 4)
    return buf[: n * w * h * 4].reshape(n, h, w, 4).copy()


def ffmpeg_encode_rgb(frames_rgb, out_path, codec="h264"):
    """frames_rgb: uint8 (n,h,w,3). Preview encode without alpha."""
    n, h, w, _ = frames_rgb.shape
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if codec == "h264":
        codec_args = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"]
    else:
        raise ValueError(codec)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(FPS), "-i", "-",
           *codec_args, "-frames:v", str(n), str(out_path)]
    proc = subprocess.run(cmd, input=frames_rgb.tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode())


def save_png_sequence(frames_rgba, seq_dir):
    """Write uint8 (n,h,w,4) frames as frame_0001.png ... via ffmpeg."""
    seq_dir = Path(seq_dir)
    seq_dir.mkdir(parents=True, exist_ok=True)
    n, h, w, _ = frames_rgba.shape
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
           "-r", str(FPS), "-i", "-",
           str(seq_dir / "frame_%04d.png")]
    proc = subprocess.run(cmd, input=frames_rgba.tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode())


def load_png_sequence(seq_dir):
    seq_dir = Path(seq_dir)
    files = sorted(seq_dir.glob("frame_*.png"))
    if not files:
        raise FileNotFoundError(f"no frames in {seq_dir}")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=width,height",
         "-of", "json", str(files[0])], capture_output=True, check=True)
    info = json.loads(probe.stdout)["streams"][0]
    w, h = info["width"], info["height"]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-i", str(seq_dir / "frame_%04d.png"),
           "-f", "rawvideo", "-pix_fmt", "rgba", "-"]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    buf = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = buf.size // (w * h * 4)
    return buf[: n * w * h * 4].reshape(n, h, w, 4).copy()


def checkerboard(h, w, cell=16, lo=0.30, hi=0.62):
    """float32 (h,w,3) checkerboard in [0,1]."""
    yy, xx = np.mgrid[0:h, 0:w]
    cells = ((yy // cell) + (xx // cell)) % 2
    board = np.where(cells, hi, lo).astype(np.float32)
    return np.stack([board, board, board], axis=-1)


def composite(rgba_f, bg, interpretation):
    """rgba_f: float32 (...,4) in [0,1]; bg: float32 (...,3).

    straight:       out = rgb * a + bg * (1 - a)
    premultiplied:  out = rgb     + bg * (1 - a)
    """
    rgb, a = rgba_f[..., :3], rgba_f[..., 3:4]
    if interpretation == "straight":
        return rgb * a + bg * (1.0 - a)
    if interpretation == "premultiplied":
        return rgb + bg * (1.0 - a)
    raise ValueError(interpretation)


def edge_mask(alpha_f, dilate=2):
    """Boolean mask of semi-transparent pixels, dilated by `dilate` px."""
    m = (alpha_f > 1e-4) & (alpha_f < 1.0 - 1e-4)
    out = m.copy()
    for _ in range(dilate):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def to_float(frames_u8):
    return frames_u8.astype(np.float32) / 255.0


def to_u8(frames_f):
    return np.clip(frames_f * 255.0 + 0.5, 0, 255).astype(np.uint8)
