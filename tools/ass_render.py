"""Render the computed cue model to real video frames via ASS + ffmpeg/libass.

Both the source TTML and the migrated IMSC document are parsed into the same
computed model (ttml_model) and rendered through this single backend, so any
rendered-frame difference is attributable to the migration itself.

Known renderer approximations (identical for source and migrated docs):
  - tts:textOutline is approximated by the ASS opaque-box outline
  - tts:lineHeight uses libass default line spacing
  - vertical writingMode is rendered horizontally
  - region background is drawn per active cue (not merged per region)
"""
import subprocess

NAMED_COLORS = {
    "transparent": (0, 0, 0, 0), "black": (0, 0, 0, 255),
    "silver": (192, 192, 192, 255), "gray": (128, 128, 128, 255),
    "white": (255, 255, 255, 255), "maroon": (128, 0, 0, 255),
    "red": (255, 0, 0, 255), "purple": (128, 0, 128, 255),
    "fuchsia": (255, 0, 255, 255), "magenta": (255, 0, 255, 255),
    "green": (0, 128, 0, 255), "lime": (0, 255, 0, 255),
    "olive": (128, 128, 0, 255), "yellow": (255, 255, 0, 255),
    "navy": (0, 0, 128, 255), "blue": (0, 0, 255, 255),
    "teal": (0, 128, 128, 255), "aqua": (0, 255, 255, 255),
    "cyan": (0, 255, 255, 255),
}

GENERIC_FONTS = {
    "default": "DejaVu Sans", "sansSerif": "DejaVu Sans",
    "proportionalSansSerif": "DejaVu Sans", "serif": "DejaVu Serif",
    "proportionalSerif": "DejaVu Serif", "monospace": "DejaVu Sans Mono",
    "monospaceSansSerif": "DejaVu Sans Mono", "monospaceSerif": "DejaVu Sans Mono",
    "cursive": "DejaVu Sans", "fantasy": "DejaVu Sans",
}


def parse_color(val):
    """TTML color -> (r, g, b, a)."""
    val = val.strip().lower()
    if val in NAMED_COLORS:
        return NAMED_COLORS[val]
    if val.startswith("#"):
        h = val[1:]
        try:
            if len(h) == 3:
                r, g, b = (int(c * 2, 16) for c in h)
                return (r, g, b, 255)
            if len(h) == 4:
                r, g, b, a = (int(c * 2, 16) for c in h)
                return (r, g, b, a)
            if len(h) == 6:
                return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)
            if len(h) == 8:
                return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16),
                        int(h[6:8], 16))
        except ValueError:
            pass
    return (255, 255, 255, 255)


def ass_color(rgba):
    r, g, b, a = rgba
    return "&H%02X%02X%02X%02X" % (255 - a, b, g, r)  # ASS &HAABBGGRR


def ass_alpha(rgba):
    return "&H%02X&" % (255 - rgba[3])


def resolve_length(val, pct_base, cell, font_px):
    """TTML length -> pixels (None for auto/normal)."""
    val = val.strip()
    if val.endswith("px"):
        return float(val[:-2])
    if val.endswith("%"):
        return pct_base * float(val[:-1]) / 100.0
    if val.endswith("c"):
        return cell * float(val[:-1])
    if val.endswith("em"):
        return font_px * float(val[:-2])
    if val in ("auto", "normal"):
        return None
    try:
        return float(val)
    except ValueError:
        return None


def cell_h_of(tp):
    return int(tp.root_extent[1]) / tp.cell_rows


def font_size_px(style, parent_px, tp):
    v = style.get("fontSize", "1c").split()
    v = v[-1]  # two-component: height governs
    base = parent_px if parent_px else cell_h_of(tp)
    px = resolve_length(v, base, cell_h_of(tp), base)
    return px if px else cell_h_of(tp)


def region_rect(region, tp):
    w, h = int(tp.root_extent[0]), int(tp.root_extent[1])
    cell_w, cell_h = w / tp.cell_cols, h / tp.cell_rows
    st = region.style
    ox, oy = (st.get("origin") or "0% 0%").split()[:2]
    ex, ey = (st.get("extent") or "100% 100%").split()[:2]
    x = resolve_length(ox, w, cell_w, 0) or 0
    y = resolve_length(oy, h, cell_h, 0) or 0
    rw = resolve_length(ex, w, cell_w, 0)
    rh = resolve_length(ey, h, cell_h, 0)
    return (int(x), int(y), int(rw if rw is not None else w - x),
            int(rh if rh is not None else h - y))


# ---------------------------------------------------------- ASS output ----

def _ass_time(t):
    """Fraction seconds -> ASS H:MM:SS.cc"""
    cs = int(round(float(t) * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return "%d:%02d:%02d.%02d" % (h, m, s, c)


_AN = {("before", "start"): 7, ("before", "center"): 8, ("before", "end"): 9,
       ("center", "start"): 4, ("center", "center"): 5, ("center", "end"): 6,
       ("after", "start"): 1, ("after", "center"): 2, ("after", "end"): 3}


def _align_key(cue_style, region_style):
    # displayAlign is a region property; textAlign inherits into content
    da = region_style.get("displayAlign", "before")
    ta = cue_style.get("textAlign", "center")
    ta = {"left": "start", "right": "end"}.get(ta, ta)
    if ta not in ("start", "center", "end"):
        ta = "center"
    if da not in ("before", "center", "after"):
        da = "before"
    return da, ta


def _run_override(run):
    st = run.style
    tags = []
    tags.append("\\fs%d" % max(1, round(st["_fontSizePx"])))
    color = parse_color(st.get("color", "white"))
    tags.append("\\c%s" % ass_color(color))
    bg = parse_color(st["backgroundColor"]) if st.get("backgroundColor") else None
    if bg and bg[3] > 0:
        tags.append("\\3c%s" % ass_color(bg))
        tags.append("\\3a%s" % ass_alpha(bg))
    else:
        tags.append("\\3a&HFF&")  # invisible opaque-box outline
    if st.get("fontStyle") == "italic":
        tags.append("\\i1")
    if st.get("fontWeight") == "bold":
        tags.append("\\b1")
    deco = st.get("textDecoration", "none")
    if "underline" in deco:
        tags.append("\\u1")
    if "lineThrough" in deco:
        tags.append("\\s1")
    fam = (st.get("fontFamily") or "default").split()[0]
    tags.append("\\fn%s" % GENERIC_FONTS.get(fam, fam))
    return "{%s}" % "".join(tags)


def _escape(text):
    return text.replace("\\", "\\h").replace("{", "(").replace("}", ")")


def cue_events(doc, cue):
    """ASS event lines for one cue (region background + text segments)."""
    tp = doc.params
    region = doc.regions.get(cue.region) or next(iter(doc.regions.values()))
    x, y, w, h = region_rect(region, tp)
    W, H = tp.root_extent
    begin, end = cue.begin, cue.end
    if end is None:
        end = begin + 5
    events = []

    rbg = region.style.get("backgroundColor")
    if rbg:
        rgba = parse_color(rbg)
        if rgba[3] > 0:
            draw = ("{\\an7\\pos(%d,%d)\\p1\\c%s\\1a%s}"
                    "m 0 0 l %d 0 l %d %d l 0 %d"
                    % (x, y, ass_color(rgba), ass_alpha(rgba), w, w, h, h))
            events.append((0, begin, end, "", 0, 0, 0, draw))

    da, ta = _align_key(cue.style, region.style)
    an = _AN[(da, ta)]
    ax = {"start": x, "center": x + w // 2, "end": x + w}[ta]
    ay = {"before": y, "center": y + h // 2, "after": y + h}[da]
    ml, mr = x, W - (x + w)

    # split into segments at timed-span boundaries
    bounds = {begin, end}
    for line in cue.lines:
        for r in line:
            if r.timing:
                bounds.add(max(begin, r.timing[0]))
                if r.timing[1] is not None:
                    bounds.add(min(end, r.timing[1]))
    bounds = sorted(bounds)
    segments = [(a, b) for a, b in zip(bounds, bounds[1:]) if a < b] \
        or [(begin, end)]

    agent = doc.agents.get(cue.agent, cue.agent or "")
    for seg_b, seg_e in segments:
        mid = (seg_b + seg_e) / 2
        parts = []
        for li, line in enumerate(cue.lines):
            if li:
                parts.append("\\N")
            for r in line:
                if r.timing and not (r.timing[0] <= mid and
                                     (r.timing[1] is None or mid < r.timing[1])):
                    continue
                parts.append(_run_override(r) + _escape(r.text))
        text = "{\\an%d\\pos(%d,%d)}" % (an, ax, ay) + "".join(parts)
        events.append((1, seg_b, seg_e, agent, ml, mr, 0, text))
    return events


def build_ass(doc, only_cue=None):
    """Full ASS script for the document (or a single cue when only_cue set)."""
    W, H = doc.params.root_extent
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: %d" % W,
        "PlayResY: %d" % H,
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: T,DejaVu Sans,20,&H00FFFFFF,&H00FFFFFF,&H00000000,"
        "&H00000000,0,0,0,0,100,100,0,0,3,1,0,2,0,0,0,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text",
    ]
    lines = list(header)
    for cue in doc.cues:
        if only_cue is not None and cue is not only_cue:
            continue
        if cue.style.get("display") == "none" or \
           cue.style.get("visibility") == "hidden":
            continue
        for layer, b, e, name, ml, mr, mv, text in cue_events(doc, cue):
            lines.append("Dialogue: %d,%s,%s,T,%s,%d,%d,%d,,%s"
                         % (layer, _ass_time(b), _ass_time(e),
                            name.replace(",", " "), ml, mr, mv, text))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------- ffmpeg ----------

def fps_string(tp):
    fr = tp.frame_rate
    if fr.denominator == 1:
        return str(fr.numerator)
    return "%d/%d" % (fr.numerator, fr.denominator)


def render_gray(ass_path, tp, n_frames, out_path, scale=1.0):
    """Render the ASS over a dark background; output rawvideo gray frames."""
    W, H = tp.root_extent
    vf = "ass='%s'" % ass_path
    if scale != 1.0:
        vf += ",scale=%d:%d" % (int(W * scale), int(H * scale))
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "lavfi", "-i",
           "color=c=0x181818:s=%dx%d:r=%s" % (W, H, fps_string(tp)),
           "-vf", vf, "-frames:v", str(n_frames),
           "-f", "rawvideo", "-pix_fmt", "gray", out_path]
    subprocess.run(cmd, check=True)


def render_frame_png(ass_path, tp, frame_n, out_png):
    """Render composited frame number frame_n to PNG."""
    W, H = tp.root_extent
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "lavfi", "-i",
           "color=c=0x181818:s=%dx%d:r=%s" % (W, H, fps_string(tp)),
           "-vf", "ass='%s',select='eq(n\\,%d)'" % (ass_path, frame_n),
           "-frames:v", "1", out_png]
    subprocess.run(cmd, check=True)


def read_gray_frames(path, width, height):
    """Iterate rawvideo gray frames as bytes objects."""
    size = width * height
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(size)
            if len(buf) < size:
                return
            yield buf
