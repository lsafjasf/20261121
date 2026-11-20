"""Shared helpers for stereo projector material generation and checking.

Pure-python RGB frame buffers (bytearray of W*H*3), piped through ffmpeg.
Frame layout (W x H):
  - top strip   : 16-cell binary barcode of the source frame index;
                  cell0 is a guard bit (always 1), cells 1..15 hold the value
  - top-right   : eye marker, two boxes; left eye = left box lit,
                  right eye = right box lit
  - scene       : moving geometric shapes; the right eye view is shifted
                  horizontally by PARALLAX px to mimic normal disparity
  - bottom-left : human readable eye tag + source index
"""

W = 320
H = 180
FPS = 30
PARALLAX = 6
BAR_X = 8
BAR_Y = 4
BAR_CELL_W = 14
BAR_CELL_H = 12
BAR_BITS = 16
EYE_MARK_X = W - 44
EYE_MARK_Y = 4
EYE_BOX = 16
EYE_GAP = 4

FONT = {
    "0": ["111", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "111"],
    "2": ["111", "001", "111", "100", "111"],
    "3": ["111", "001", "111", "001", "111"],
    "4": ["101", "101", "111", "001", "001"],
    "5": ["111", "100", "111", "001", "111"],
    "6": ["111", "100", "111", "101", "111"],
    "7": ["111", "001", "001", "010", "010"],
    "8": ["111", "101", "111", "101", "111"],
    "9": ["111", "101", "111", "001", "111"],
    "A": ["010", "101", "111", "101", "101"],
    "B": ["110", "101", "110", "101", "110"],
    "C": ["011", "100", "100", "100", "011"],
    "D": ["110", "101", "101", "101", "110"],
    "E": ["111", "100", "110", "100", "111"],
    "F": ["111", "100", "110", "100", "100"],
    "G": ["011", "100", "101", "101", "011"],
    "H": ["101", "101", "111", "101", "101"],
    "I": ["111", "010", "010", "010", "111"],
    "J": ["001", "001", "001", "101", "010"],
    "K": ["101", "101", "110", "101", "101"],
    "L": ["100", "100", "100", "100", "111"],
    "M": ["101", "111", "111", "101", "101"],
    "N": ["101", "111", "111", "111", "101"],
    "O": ["010", "101", "101", "101", "010"],
    "P": ["111", "101", "111", "100", "100"],
    "Q": ["010", "101", "101", "010", "001"],
    "R": ["110", "101", "110", "101", "101"],
    "S": ["011", "100", "010", "001", "110"],
    "T": ["111", "010", "010", "010", "010"],
    "U": ["101", "101", "101", "101", "111"],
    "V": ["101", "101", "101", "101", "010"],
    "W": ["101", "101", "111", "111", "101"],
    "X": ["101", "101", "010", "101", "101"],
    "Y": ["101", "101", "010", "010", "010"],
    "Z": ["111", "001", "010", "100", "111"],
    "-": ["000", "000", "111", "000", "000"],
    "+": ["000", "010", "111", "010", "000"],
    ".": ["000", "000", "000", "000", "010"],
    ":": ["000", "010", "000", "010", "000"],
    "=": ["000", "111", "000", "111", "000"],
    "#": ["101", "111", "101", "111", "101"],
    " ": ["000", "000", "000", "000", "000"],
}


def new_frame(color=(24, 24, 32)):
    buf = bytearray(W * H * 3)
    r, g, b = color
    for y in range(H):
        shade = (y * 24) // H
        row = bytes((r + shade, g + shade, b + shade)) * W
        buf[y * W * 3:(y + 1) * W * 3] = row
    return buf


def set_px(buf, x, y, color):
    if 0 <= x < W and 0 <= y < H:
        i = (y * W + x) * 3
        buf[i:i + 3] = bytes(color)


def fill_rect(buf, x0, y0, w, h, color):
    row = bytes(color) * max(0, w)
    for y in range(max(0, y0), min(H, y0 + h)):
        xa = max(0, x0)
        xb = min(W, x0 + w)
        if xa >= xb:
            continue
        i = (y * W + xa) * 3
        buf[i:i + (xb - xa) * 3] = row[:(xb - xa) * 3]


def fill_circle(buf, cx, cy, rad, color):
    r2 = rad * rad
    for y in range(cy - rad, cy + rad + 1):
        for x in range(cx - rad, cx + rad + 1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                set_px(buf, x, y, color)


def fill_triangle(buf, cx, cy, size, color):
    for dy in range(size):
        half = size - dy
        for dx in range(-half, half + 1):
            set_px(buf, cx + dx, cy + dy, color)


def draw_text(buf, x, y, text, color, scale=2):
    cx = x
    for ch in text.upper():
        glyph = FONT.get(ch, FONT[" "])
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "1":
                    fill_rect(buf, cx + gx * scale, y + gy * scale,
                              scale, scale, color)
        cx += 4 * scale


def draw_barcode(buf, index):
    bits = [1] + [(index >> (14 - i)) & 1 for i in range(15)]
    for i, bit in enumerate(bits):
        color = (255, 255, 255) if bit else (0, 0, 0)
        fill_rect(buf, BAR_X + i * BAR_CELL_W + 1, BAR_Y + 1,
                  BAR_CELL_W - 2, BAR_CELL_H - 2, color)


def draw_eye_marker(buf, eye):
    left_on = (eye == "L")
    fill_rect(buf, EYE_MARK_X, EYE_MARK_Y, EYE_BOX, EYE_BOX,
              (255, 255, 255) if left_on else (0, 0, 0))
    fill_rect(buf, EYE_MARK_X + EYE_BOX + EYE_GAP, EYE_MARK_Y,
              EYE_BOX, EYE_BOX, (0, 0, 0) if left_on else (255, 255, 255))


def _region_mean(buf, x0, y0, x1, y1):
    total = 0
    n = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            j = (y * W + x) * 3
            total += (buf[j] + buf[j + 1] + buf[j + 2]) // 3
            n += 1
    return total / max(1, n)


def read_barcode(buf):
    bits = []
    for i in range(BAR_BITS):
        mean = _region_mean(buf, BAR_X + i * BAR_CELL_W + 3, BAR_Y + 3,
                            BAR_X + i * BAR_CELL_W + BAR_CELL_W - 3,
                            BAR_Y + BAR_CELL_H - 3)
        bits.append(1 if mean > 127 else 0)
    if bits[0] != 1:
        return None
    value = 0
    for b in bits[1:]:
        value = (value << 1) | b
    return value


def read_eye_marker(buf):
    m0 = _region_mean(buf, EYE_MARK_X + 3, EYE_MARK_Y + 3,
                      EYE_MARK_X + EYE_BOX - 3, EYE_MARK_Y + EYE_BOX - 3)
    x1 = EYE_MARK_X + EYE_BOX + EYE_GAP
    m1 = _region_mean(buf, x1 + 3, EYE_MARK_Y + 3,
                      x1 + EYE_BOX - 3, EYE_MARK_Y + EYE_BOX - 3)
    if abs(m0 - m1) < 60:
        return "?"
    return "L" if m0 > m1 else "R"


def render_source_frame(index, eye):
    """Deterministic geometric scene for a source frame index and eye.

    The right eye applies a horizontal PARALLAX shift only, which is the
    normal binocular disparity and must not be treated as a sync error.
    """
    buf = new_frame()
    dx = 0 if eye == "L" else PARALLAX
    fill_rect(buf, (index * 7) % W, 24, 3, H - 48, (70, 70, 90))
    cx = 30 + (index * 5) % (W - 60)
    cy = 70 + ((index * 13) % 40)
    fill_circle(buf, cx + dx, cy, 14, (230, 120, 40))
    rx = W - 50 - (index * 3) % (W - 100)
    fill_rect(buf, rx + dx, 100 + ((index * 7) % 30), 26, 18, (60, 180, 120))
    tx = 40 + (index * 11) % (W - 80)
    fill_triangle(buf, tx + dx, 130 - ((index * 5) % 25), 12, (90, 120, 230))
    draw_barcode(buf, index)
    draw_eye_marker(buf, eye)
    draw_text(buf, 8, H - 18, eye, (255, 220, 80), scale=3)
    draw_text(buf, 40, H - 16, "SRC " + str(index), (220, 220, 220), scale=2)
    return buf


def encode_video(frames, path, fps=FPS, crf=18):
    import subprocess
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", "%dx%d" % (W, H), "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), path],
        stdin=subprocess.PIPE)
    for buf in frames:
        proc.stdin.write(bytes(buf))
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg encode failed for " + path)


def decode_video(path):
    import subprocess
    proc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-i", path,
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE)
    frames = []
    size = W * H * 3
    while True:
        data = proc.stdout.read(size)
        if len(data) < size:
            break
        frames.append(bytearray(data))
    proc.wait()
    return frames
