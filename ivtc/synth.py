"""Synthetic mixed-cadence sample generator.

Builds a deterministic clip with three recognisably different sections:

1. a 24fps "film" scene (moving vertical bar + moving disc + frame code),
   converted to 29.97i with classic 3:2 pulldown,
2. a hard cut to a second scene, mid cadence cycle,
3. native 60i video (every field carries new, faster motion),
4. a second cut back into a pulldown scene at a shifted phase,

so the detector has to re-lock instead of seeing one global rhythm.
Ground-truth provenance (source frame / time / scene) is attached to every
generated field, which lets the test suite verify every reconstructed frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .imageio import VideoFrame

WIDTH = 160
HEIGHT = 128


def _clip(value: float) -> int:
    return 0 if value < 0 else 255 if value > 255 else int(value)


def render_progressive_frame(index: int, scene: str, fps: float = 24.0) -> bytes:
    """One clean progressive film frame with strong moving edges."""
    buf = bytearray(WIDTH * HEIGHT)
    t = index / fps
    if scene == "A":
        bg, bar, disc = 32, 220, 160
        bar_x = int((20 + 90 * t) % (WIDTH - 24))
        disc_y = int(HEIGHT * 0.5 + 26 * math.sin(t * 1.7))
    else:
        bg, bar, disc = 200, 40, 90
        bar_x = int((WIDTH - 44 - 80 * t) % (WIDTH - 24))
        disc_y = int(HEIGHT * 0.5 + 30 * math.cos(t * 2.1))

    for y in range(HEIGHT):
        row = y * WIDTH
        for x in range(WIDTH):
            v = bg
            if bar_x <= x < bar_x + 12:
                v = bar
            dx = x - (WIDTH // 2)
            if dx * dx + (y - disc_y) ** 2 <= 18 * 18:
                v = disc
            buf[row + x] = _clip(v)

    # binary frame code in the top-left 6x8 area: unique per source frame,
    # used by tests to confirm which film frame a weave actually recovered
    for bit in range(6):
        if index & (1 << bit):
            for y in range(1, 7):
                buf[y * WIDTH + 1 + bit] = 255
    return bytes(buf)


def render_native_field(field_index: int, scene: str) -> bytes:
    """One 60i field: genuinely new content every field (fast panning ramp)."""
    fh = HEIGHT // 2
    buf = bytearray(WIDTH * fh)
    base = 60 if scene == "C" else 120
    shift = (field_index * 3) % WIDTH
    for y in range(fh):
        row = y * WIDTH
        for x in range(WIDTH):
            stripe = 180 if ((x + shift) // 16) % 2 == 0 else 40
            buf[row + x] = _clip(0.5 * base + 0.5 * stripe)
    return bytes(buf)


# classic 3:2: five output fields per four film frames, one repeat
TELECINE_PATTERN = (
    (0, "T"), (0, "B"), (1, "T"), (1, "B"), (1, "T"),
    (2, "B"), (2, "T"), (3, "B"), (3, "T"), (3, "B"),
)


def _field_from_progressive(pixels: bytes, parity: str) -> bytes:
    fh = HEIGHT // 2
    out = bytearray(WIDTH * fh)
    start = 0 if parity == "T" else 1
    for r in range(fh):
        src = (start + 2 * r) * WIDTH
        dst = r * WIDTH
        out[dst:dst + WIDTH] = pixels[src:src + WIDTH]
    return bytes(out)


@dataclass
class GroundTruth:
    kind: str
    scene: str
    film_frame: int | None
    telecine_step: int | None


def telecine_fields(film_count: int, scene: str,
                    phase_shift: int = 0) -> list:
    """Emit 3:2 pulldown fields for ``film_count`` progressive frames.

    Returns ``(Field-like records, truth)``; records carry raw field buffers
    plus parity and full provenance.  ``phase_shift`` rotates the pattern so
    a cut can re-enter cadence at a non-zero phase.
    """
    film = [render_progressive_frame(i, scene) for i in range(film_count)]
    # A phase change means entering the infinite 3:2 periodic stream at a
    # different offset; it must NOT rotate the T/B labels, otherwise a bottom
    # field would land in a global top slot (real capture always alternates
    # top/bottom by field index).  We emit the unrotated periodic sequence
    # and skip ``phase_shift`` leading fields instead.
    pattern = TELECINE_PATTERN
    records = []
    truth = []
    step = phase_shift
    full_blocks = film_count // 4
    while step < phase_shift + full_blocks * 10:
        film_idx, parity = pattern[step % 10]
        block = step // 10
        frame_idx = block * 4 + film_idx
        if frame_idx >= film_count:
            break
        records.append({
            "parity": "top" if parity == "T" else "bottom",
            "buf": _field_from_progressive(film[frame_idx], parity),
            "source": {
                "scene": scene, "film_frame": frame_idx,
                "telecine_step": step, "origin": "telecine",
            },
        })
        truth.append(GroundTruth("pulldown", scene, frame_idx, step))
        step += 1
    return records, truth


def _weave_record_pair(top_rec: dict, bot_rec: dict, frame_index: int) -> VideoFrame:
    fh = HEIGHT // 2
    buf = bytearray(WIDTH * HEIGHT)
    for r in range(fh):
        off = r * WIDTH
        buf[(2 * r) * WIDTH:(2 * r) * WIDTH + WIDTH] = top_rec["buf"][off:off + WIDTH]
        buf[(2 * r + 1) * WIDTH:(2 * r + 1) * WIDTH + WIDTH] = \
            bot_rec["buf"][off:off + WIDTH]
    source = {
        "top": dict(top_rec["source"]),
        "bottom": dict(bot_rec["source"]),
    }
    return VideoFrame(frame_index, WIDTH, HEIGHT, bytes(buf), source)


def _native_records(count_fields: int, scene: str, start_index: int) -> list:
    records = []
    for k in range(count_fields):
        records.append({
            "parity": "top" if (start_index + k) % 2 == 0 else "bottom",
            "buf": render_native_field(k, scene),
            "source": {
                "scene": scene, "native_field": k,
                "origin": "native60i",
            },
        })
    return records


def build_mixed_clip() -> tuple[list[VideoFrame], list[GroundTruth]]:
    """Assemble the canonical demonstration clip.

    Timeline (field units, 60 fields = 30 interlaced frames):

    * fields  0..29 : scene A, 3:2 pulldown, phase 0
    * cut at field 30 (frame 15), scene change A -> C
    * fields 30..45 : scene C, native 60i
    * cut at field 46 (frame 23), scene change C -> B
    * fields 46..59 : scene B, 3:2 pulldown, phase shifted by 2
    """
    records: list[dict] = []
    truth: list[GroundTruth] = []

    rec_a, truth_a = telecine_fields(12, "A", phase_shift=0)
    records.extend(rec_a[:30])
    truth.extend(truth_a[:30])

    cut1 = len(records)
    native = _native_records(16, "C", start_index=cut1)
    records.extend(native)
    truth.extend(GroundTruth("native", "C", None, None) for _ in native)

    cut2 = len(records)
    rec_b, truth_b = telecine_fields(12, "B", phase_shift=2)
    records.extend(rec_b[:14])
    truth.extend(truth_b[:14])

    if len(records) % 2:
        raise RuntimeError("clip must contain an even number of fields")
    expected_first = "top"
    frames = []
    for fi in range(len(records) // 2):
        top_rec, bot_rec = records[2 * fi], records[2 * fi + 1]
        if top_rec["parity"] != expected_first:
            raise RuntimeError(
                f"parity assembly error at frame {fi}: "
                f"{top_rec['parity']}/{bot_rec['parity']}")
        frames.append(_weave_record_pair(top_rec, bot_rec, fi))
    return frames, truth[:len(records)]


def timeline_truth() -> dict:
    """Hand-verifiable ground-truth boundaries used by the test suite."""
    return {
        "width": WIDTH,
        "height": HEIGHT,
        "field_count": 60,
        "frame_count": 30,
        "segments": [
            {"kind": "pulldown", "scene": "A",
             "field_start": 0, "field_end": 30, "phase": 0},
            {"kind": "native", "scene": "C",
             "field_start": 30, "field_end": 46},
            {"kind": "pulldown", "scene": "B",
             "field_start": 46, "field_end": 60, "phase": 2},
        ],
        "cuts": [
            {"field_index": 30, "frame_index": 15, "scene_from": "A",
             "scene_to": "C"},
            {"field_index": 46, "frame_index": 23, "scene_from": "C",
             "scene_to": "B"},
        ],
    }
