"""Build a transparent proxy from a straight-alpha master sequence.

The interpretation flag is explicit:
  straight       - encode RGB as-is; alpha separate
  premultiplied  - multiply RGB by alpha before encoding

Also writes a checkerboard preview (no alpha) rendered from the *encoded*
proxy using the declared interpretation, so what you see is what you got.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common import (FPS, checkerboard, composite, ffmpeg_decode_rgba,
                    ffmpeg_encode_rgb, ffmpeg_encode_rgba,
                    load_png_sequence, to_float, to_u8)


def build_proxy(master_dir, interpretation, codec, proxy_out, preview_out):
    master_dir = Path(master_dir)
    frames_u8 = load_png_sequence(master_dir)
    frames = to_float(frames_u8)
    if interpretation == "premultiplied":
        frames = frames.copy()
        frames[..., :3] *= frames[..., 3:4]
    elif interpretation != "straight":
        raise ValueError(interpretation)

    ffmpeg_encode_rgba(to_u8(frames), proxy_out, codec=codec)

    # Preview: decode the actual proxy file and comp over checkerboard.
    decoded = to_float(ffmpeg_decode_rgba(proxy_out))
    n, h, w, _ = decoded.shape
    board = checkerboard(h, w)
    bg = np.broadcast_to(board, (n, h, w, 3))
    preview = composite(decoded, bg, interpretation)
    ffmpeg_encode_rgb(to_u8(preview), preview_out)

    meta = {
        "master": str(master_dir),
        "interpretation": interpretation,
        "codec": codec,
        "fps": FPS,
        "frames": int(n),
        "width": int(w),
        "height": int(h),
        "proxy": str(proxy_out),
        "preview": str(preview_out),
    }
    meta_path = Path(proxy_out).with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[proxy] {interpretation:13s} {codec:7s} -> {proxy_out}")
    print(f"[preview] checkerboard comp   -> {preview_out}")
    print(f"[meta]                      -> {meta_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master", required=True,
                    help="master PNG sequence dir (from make_masters.py)")
    ap.add_argument("--interpretation", required=True,
                    choices=["straight", "premultiplied"],
                    help="how RGB is encoded relative to alpha")
    ap.add_argument("--codec", default="prores",
                    choices=["prores", "qtrle", "png"])
    ap.add_argument("--out", required=True, help="proxy movie path (.mov)")
    ap.add_argument("--preview", default=None,
                    help="checkerboard preview path (default: <out>.checker.mp4)")
    args = ap.parse_args()
    preview = args.preview or str(Path(args.out).with_suffix("")) + ".checker.mp4"
    build_proxy(args.master, args.interpretation, args.codec,
                args.out, preview)


if __name__ == "__main__":
    main()
