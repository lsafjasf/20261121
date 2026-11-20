"""TTML parsing: style cascade, inheritance, time containers -> computed cue model.

The computed model is format-agnostic and is consumed by both the IMSC
emitter (imsc_emit.py) and the ASS renderer (ass_render.py), so source TTML
and migrated IMSC are rendered through the exact same layout path.
"""
import re
import xml.etree.ElementTree as ET
from fractions import Fraction

NS = {
    "tt": "http://www.w3.org/ns/ttml",
    "tts": "http://www.w3.org/ns/ttml#styling",
    "ttp": "http://www.w3.org/ns/ttml#parameter",
    "ttm": "http://www.w3.org/ns/ttml#metadata",
    "ebutts": "urn:ebu:tt:style",
    "smpte": "http://www.smpte-ra.org/ns/2052-1:2010/smpte-tt",
    "xml": "http://www.w3.org/XML/1998/namespace",
}
for _p, _u in NS.items():
    ET.register_namespace("" if _p == "tt" else _p, _u)

TTS = "{%s}" % NS["tts"]
TTP = "{%s}" % NS["ttp"]
TTM = "{%s}" % NS["ttm"]
XML_ID = "{%s}id" % NS["xml"]


def qname(tag):
    """{uri}local -> prefix:local for readable reports."""
    if tag.startswith("{"):
        uri, local = tag[1:].split("}", 1)
        for p, u in NS.items():
            if u == uri:
                return "%s:%s" % (p, local)
        return local
    return tag


# ---------------------------------------------------------------- timing ----

class TimingParams:
    def __init__(self, root):
        self.frame_rate = int(root.get(TTP + "frameRate", "30"))
        mult = root.get(TTP + "frameRateMultiplier", "").split()
        if len(mult) == 2:
            self.frame_rate = Fraction(self.frame_rate) * Fraction(int(mult[0]), int(mult[1]))
        else:
            self.frame_rate = Fraction(self.frame_rate)
        self.sub_frame_rate = int(root.get(TTP + "subFrameRate", "1"))
        self.tick_rate = int(root.get(TTP + "tickRate", "0")) or int(self.frame_rate * self.sub_frame_rate)
        self.time_base = root.get(TTP + "timeBase", "media")
        cell = root.get(TTP + "cellResolution", "32 15").split()
        self.cell_cols, self.cell_rows = int(cell[0]), int(cell[1])
        ext = root.get(TTP + "extent", "")
        if ext:
            raw = parse_extent(ext)
            self.root_extent = tuple(
                int(float(v[:-2])) if v.endswith("px") else int(float(v))
                for v in raw)
        else:
            self.root_extent = (640, 360)


_OFFSET_RE = re.compile(r"^([+-]?\d+(?:\.\d+)?)(h|m|s|ms|f|t|%)$")
_CLOCK_RE = re.compile(r"^(\d+):(\d{2}):(\d{2})(?:\.(\d+))?$")
_CLOCK_F_RE = re.compile(r"^(\d+):(\d{2}):(\d{2}):(\d+)(?:\.(\d+))?$")


def parse_time(expr, tp, parent_dur=None):
    """Parse a TTML time expression to Fraction seconds (or Fraction for %).

    Returns None when a percentage cannot be resolved (unknown parent duration).
    """
    expr = expr.strip()
    m = _CLOCK_F_RE.match(expr)
    if m:
        h, mi, s, f, sub = m.groups()
        frames = Fraction(int(f))
        if sub:
            frames += Fraction(int(sub), tp.sub_frame_rate)
        return Fraction(int(h) * 3600 + int(mi) * 60 + int(s)) + frames / tp.frame_rate
    m = _CLOCK_RE.match(expr)
    if m:
        h, mi, s, frac = m.groups()
        sec = Fraction(int(h) * 3600 + int(mi) * 60 + int(s))
        if frac:
            sec += Fraction(int(frac), 10 ** len(frac))
        return sec
    m = _OFFSET_RE.match(expr)
    if not m:
        raise ValueError("bad time expression: %r" % expr)
    val, unit = m.groups()
    val = Fraction(val)
    if unit == "h":
        return val * 3600
    if unit == "m":
        return val * 60
    if unit == "s":
        return val
    if unit == "ms":
        return val / 1000
    if unit == "f":
        return val / tp.frame_rate
    if unit == "t":
        return val / tp.tick_rate
    if unit == "%":
        if parent_dur is None:
            return None
        return parent_dur * val / 100
    raise ValueError("bad time unit: %r" % expr)


def parse_extent(expr):
    """'640px 360px' | '640 360' | '80% 20%' -> (w, h) as strings kept raw."""
    parts = expr.split()
    return (parts[0], parts[1]) if len(parts) == 2 else (expr, expr)


def fmt_time(t):
    """Fraction seconds -> TTML clock time, ms-exact when possible."""
    ms = t * 1000
    if ms.denominator == 1:
        total = int(ms)
        h, rem = divmod(total, 3600000)
        m, rem = divmod(rem, 60000)
        s, msec = divmod(rem, 1000)
        return "%02d:%02d:%02d.%03d" % (h, m, s, msec), True
    return None, False


# ----------------------------------------------------------------- style ----

# Properties that inherit down the content tree (TTML1 8.2.x "Inherited: yes").
INHERITED_PROPS = {
    "color", "direction", "fontFamily", "fontSize", "fontStyle", "fontWeight",
    "lineHeight", "opacity", "textAlign", "textDecoration", "textOutline",
    "unicodeBidi", "visibility", "wrapOption", "writingMode", "letterSpacing",
    "textShadow", "ruby", "rubyAlign", "rubyPosition", "textCombine",
}

INITIAL_STYLE = {
    "color": "white",
    "direction": "ltr",
    "display": "auto",
    "displayAlign": "before",
    "fontFamily": "default",
    "fontSize": "1c",
    "fontStyle": "normal",
    "fontWeight": "normal",
    "lineHeight": "normal",
    "opacity": "1",
    "showBackground": "always",
    "textAlign": "center",
    "textDecoration": "none",
    "unicodeBidi": "normal",
    "visibility": "visible",
    "wrapOption": "wrap",
    "writingMode": "lrtb",
    "zIndex": "auto",
}


def local_style_attrs(el):
    """Style properties specified directly on an element (tts:* and known foreign)."""
    out = {}
    for k, v in el.attrib.items():
        if k.startswith(TTS):
            out[k[len(TTS):]] = v
        elif k.startswith("{%s}" % NS["ebutts"]):
            out["ebutts:" + k.split("}", 1)[1]] = v
        elif k.startswith("{%s}" % NS["smpte"]):
            out["smpte:" + k.split("}", 1)[1]] = v
    return out


class StyleSheet:
    """<styling> definitions with chained style references."""

    def __init__(self, head):
        self.styles = {}
        styling = head.find("tt:styling", NS) if head is not None else None
        if styling is None:
            return
        for st in styling.findall("tt:style", NS):
            sid = st.get(XML_ID)
            if sid:
                self.styles[sid] = st

    def resolve_ref(self, sid, _seen=None):
        """Flatten one style element incl. its own style="" chain."""
        _seen = _seen or set()
        if sid in _seen or sid not in self.styles:
            return {}
        _seen.add(sid)
        el = self.styles[sid]
        merged = {}
        for ref in (el.get("style") or "").split():
            merged.update(self.resolve_ref(ref, _seen))
        merged.update(local_style_attrs(el))
        return merged

    def resolve_refs(self, refs):
        merged = {}
        for ref in refs.split():
            merged.update(self.resolve_ref(ref))
        return merged


# ----------------------------------------------------------------- model ----

class Region:
    def __init__(self, rid, style, specified=None):
        self.id = rid
        self.style = style  # computed region style (origin/extent/displayAlign/...)
        self.specified = specified or {}  # only explicitly specified props


class Run:
    """A leaf text run with fully computed style."""
    __slots__ = ("text", "style", "agent", "path", "timing")

    def __init__(self, text, style, agent, path, timing=None):
        self.text = text
        self.style = style
        self.agent = agent
        self.path = path
        self.timing = timing  # (begin, end) absolute Fractions, for timed spans


class Cue:
    """A <p> resolved to absolute time with computed styles and lines of runs."""
    def __init__(self, cid, path):
        self.id = cid
        self.path = path
        self.region = None
        self.begin = None   # Fraction seconds
        self.end = None
        self.lines = []     # list of list[Run]
        self.style = {}     # computed style of the <p> itself
        self.agent = None
        self.degraded = []  # degradation notes attached to this cue
        self.sets = []      # raw <set>/<animate> elements found inside


class Document:
    def __init__(self, src_path):
        self.src_path = src_path
        self.params = None
        self.regions = {}
        self.agents = {}    # id -> display name
        self.cues = []
        self.degradations = []  # dicts: element/attribute/value/reason/suggestion/severity

    def degrade(self, element, attribute, value, reason, suggestion, severity="warning"):
        self.degradations.append({
            "element": element, "attribute": attribute, "value": value,
            "reason": reason, "suggestion": suggestion, "severity": severity,
        })


# ---------------------------------------------------------------- parsing ---

def _path(parent_path, tag, index):
    return "%s/%s[%d]" % (parent_path, tag, index)


def _collapse_ws(text):
    return re.sub(r"\s+", " ", text)


def parse_document(src_path, root=None):
    """Parse a TTML/IMSC document into the computed model."""
    if root is None:
        tree = ET.parse(src_path)
        root = tree.getroot()
    doc = Document(src_path)
    tp = TimingParams(root)
    doc.params = tp
    if tp.time_base == "smpte":
        doc.degrade("/tt", "ttp:timeBase", "smpte",
                    "SMPTE time base is not permitted in IMSC",
                    "re-express times in media time base (clock or offset expressions)")

    head = root.find("tt:head", NS)
    sheet = StyleSheet(head)

    # agents
    if head is not None:
        for ag in head.iter("{%s}agent" % NS["ttm"]):
            aid = ag.get(XML_ID)
            name_el = ag.find("ttm:name", NS)
            doc.agents[aid] = (name_el.text or aid) if name_el is not None else aid

    # regions: computed style = style refs + local attrs
    layout = head.find("tt:layout", NS) if head is not None else None
    if layout is not None:
        for reg in layout.findall("tt:region", NS):
            rid = reg.get(XML_ID)
            specified = sheet.resolve_refs(reg.get("style") or "")
            specified.update(local_style_attrs(reg))
            style = dict(INITIAL_STYLE)
            style.update(specified)
            doc.regions[rid] = Region(rid, style, specified)
    if not doc.regions:
        default = dict(INITIAL_STYLE)
        default.update({"origin": "10% 75%", "extent": "80% 20%",
                        "displayAlign": "after", "textAlign": "center"})
        doc.regions[""] = Region("", default)

    body = root.find("tt:body", NS)
    if body is None:
        return doc

    body_style = dict(INITIAL_STYLE)
    body_style.update(sheet.resolve_refs(body.get("style") or ""))
    body_style.update(local_style_attrs(body))
    _walk_block(doc, sheet, body, body_style, None, None,
                Fraction(0), None, "par", "/tt/body", None)
    doc.cues.sort(key=lambda c: (c.begin, c.end if c.end is not None else Fraction(10**9)))
    return doc


def _font_px(font_size, parent_px, tp):
    """Resolve tts:fontSize to pixels (height component governs)."""
    v = (font_size or "1c").split()[-1]
    cell = tp.root_extent[1] / tp.cell_rows
    base = parent_px if parent_px else cell
    v = v.strip()
    try:
        if v.endswith("%"):
            return base * float(v[:-1]) / 100.0
        if v.endswith("c"):
            return cell * float(v[:-1])
        if v.endswith("px"):
            return float(v[:-2])
        if v.endswith("em"):
            return base * float(v[:-2])
    except ValueError:
        pass
    return cell


def _element_computed(sheet, el, inh, tp):
    """computed = INITIAL + inherited + style refs (in order) + local attrs"""
    computed = dict(INITIAL_STYLE)
    computed.update({k: v for k, v in inh.items() if k in INHERITED_PROPS})
    computed.update(sheet.resolve_refs(el.get("style") or ""))
    computed.update(local_style_attrs(el))
    computed["_fontSizePx"] = _font_px(computed.get("fontSize"),
                                       inh.get("_fontSizePx"), tp)
    # fontSize computes to an absolute length; children inherit that value
    computed["fontSize"] = "%.2fpx" % computed["_fontSizePx"]
    return computed


def _walk_block(doc, sheet, el, inh, region_id, agent,
                p_begin, p_end, parent_tc, path, seq_base):
    """Resolve one body/div/p element. Returns (begin, end) absolute Fractions."""
    tp = doc.params
    reg = el.get("region")
    if reg is not None:
        region_id = reg
        if reg in doc.regions:
            # region is the inheritance parent of content flowed into it
            inh = dict(inh)
            inh.update({k: v for k, v in doc.regions[reg].specified.items()
                        if k in INHERITED_PROPS})
        else:
            doc.degrade(path, "region", reg, "reference to undefined region",
                        "define the region in <layout> or remove the reference")
    computed = _element_computed(sheet, el, inh, tp)
    if el.get(TTM + "agent"):
        agent = el.get(TTM + "agent")

    parent_dur = (p_end - p_begin) if p_end is not None else None
    base = seq_base if seq_base is not None else p_begin
    b_attr, e_attr, d_attr = el.get("begin"), el.get("end"), el.get("dur")
    begin = base + (parse_time(b_attr, tp, parent_dur) or Fraction(0)) if b_attr else base
    end = None
    if e_attr:
        end = base + (parse_time(e_attr, tp, parent_dur) or Fraction(0))
    elif d_attr:
        dur = parse_time(d_attr, tp, parent_dur)
        end = begin + dur if dur is not None else None
    else:
        end = p_end
    if begin < p_begin:
        begin = p_begin
    if end is not None and p_end is not None and end > p_end:
        end = p_end

    tag = el.tag.split("}")[-1]
    if tag == "p":
        cue = Cue(el.get(XML_ID) or path.rsplit("/", 1)[-1], path)
        cue.region = region_id if region_id is not None else ""
        cue.begin, cue.end = begin, end
        cue.style = computed
        cue.agent = agent
        _walk_inline(doc, sheet, el, computed, agent, cue, begin, path)
        if cue.lines:
            doc.cues.append(cue)
        return begin, end

    # body or div: iterate timed children
    tc = el.get("timeContainer", parent_tc if tag == "body" else "par")
    if tag == "body":
        tc = el.get("timeContainer", "par")
    offset = begin
    counts = {}
    for child in el:
        ctag = child.tag.split("}")[-1]
        if ctag not in ("div", "p"):
            continue
        counts[ctag] = counts.get(ctag, 0) + 1
        cpath = _path(path, ctag, counts[ctag])
        c_begin, c_end = _walk_block(
            doc, sheet, child, computed, region_id, agent,
            begin, end, tc, cpath, offset if tc == "seq" else None)
        if tc == "seq":
            if c_end is None:
                doc.degrade(cpath, "dur", None,
                            "child of seq container has indefinite duration",
                            "give every child of a seq container an explicit dur or end",
                            severity="error")
            else:
                offset = c_end
    if tc == "seq" and end is None:
        end = offset
    return begin, end


def _walk_inline(doc, sheet, el, computed, agent, cue, p_begin, path):
    """Collect runs/lines from a <p>; handles span, br, set, metadata."""
    tp = doc.params
    lines = [[]]

    def add_text(text, style, ag, rpath, timing=None):
        if text:
            t = _collapse_ws(text)
            if t:
                # coalesce with previous run when presentation is identical,
                # so whitespace collapses consistently across node boundaries
                line = lines[-1]
                if line and line[-1].style == style and line[-1].agent == ag \
                        and line[-1].timing == timing:
                    line[-1].text = _collapse_ws(line[-1].text + t)
                    return
                lines[-1].append(Run(t, style, ag, rpath, timing))

    def recurse(node, style, ag, npath):
        add_text(node.text, style, ag, npath)
        counts = {}
        for child in node:
            ctag = child.tag.split("}")[-1]
            counts[ctag] = counts.get(ctag, 0) + 1
            cpath = _path(npath, ctag, counts[ctag])
            if ctag == "br":
                lines.append([])
            elif ctag == "span":
                cstyle = _element_computed(sheet, child, style, tp)
                cagent = child.get(TTM + "agent") or ag
                timing = None
                if child.get("begin") or child.get("end"):
                    cb = p_begin + (parse_time(child.get("begin"), tp) or Fraction(0)) \
                        if child.get("begin") else p_begin
                    ce = p_begin + parse_time(child.get("end"), tp) \
                        if child.get("end") else None
                    timing = (cb, ce)
                recurse(child, cstyle, cagent, cpath)
            elif ctag in ("set", "animate"):
                cue.sets.append((cpath, dict(child.attrib)))
            elif ctag == "metadata":
                cue.sets.append((cpath, {"__metadata__": "1"}))
            add_text(child.tail, style, ag, npath)

    recurse(el, computed, agent, path)
    # trim whitespace at line edges, drop empty runs
    cleaned = []
    for line in lines:
        if line:
            line[0].text = line[0].text.lstrip()
            line[-1].text = line[-1].text.rstrip()
        line = [r for r in line if r.text]
        if line:
            cleaned.append(line)
    cue.lines = cleaned
