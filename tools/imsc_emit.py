"""Migrate a TTML document to the supported IMSC 1.1 text-profile subset.

Strategy: transform the source XML tree (keeping referential styling intact),
flatten seq/par time containers to absolute document time on <p>, filter every
attribute/element against the supported subset, and record a human-readable
degradation entry (element position + suggestion) for anything that cannot be
migrated equivalently.
"""
import xml.etree.ElementTree as ET
from fractions import Fraction

import ttml_model as M

IMSC_PROFILE = "http://www.w3.org/ns/ttml/profile/imsc1.1/text"

# Style properties kept as-is in the IMSC 1.1 text-profile subset we target.
SUPPORTED_STYLE = {
    "backgroundColor", "color", "direction", "display", "displayAlign",
    "extent", "fontFamily", "fontSize", "fontStyle", "fontWeight",
    "lineHeight", "opacity", "origin", "overflow", "padding",
    "showBackground", "textAlign", "textDecoration", "textOutline",
    "unicodeBidi", "visibility", "wrapOption", "writingMode", "zIndex",
    "letterSpacing", "ruby", "rubyAlign", "rubyPosition", "textCombine",
    "textEmphasis",
}
SUPPORTED_FOREIGN_STYLE = {"ebutts:linePadding", "itts:fillLineGap"}

# Non-style attributes allowed on content elements.
SUPPORTED_CONTENT_ATTRS = {"begin", "end", "dur", "region", "style"}

WRITING_MODE_ALIASES = {"lr": "lrtb", "rl": "rltb", "tb": "tbrl"}


class Migrator:
    def __init__(self, src_path):
        self.src_path = src_path
        self.tree = ET.parse(src_path)
        self.root = self.tree.getroot()
        self.model = M.parse_document(src_path, root=self.root)
        self.degradations = list(self.model.degradations)
        self.cue_by_path = {c.path: c for c in self.model.cues}

    def degrade(self, element, attribute, value, reason, suggestion,
                severity="warning"):
        self.degradations.append({
            "element": element, "attribute": attribute, "value": value,
            "reason": reason, "suggestion": suggestion, "severity": severity,
        })

    # --------------------------------------------------------- timing ----
    def _fmt_abs_time(self, t, path):
        """Absolute Fraction seconds -> best IMSC time expression."""
        clock, ok = M.fmt_time(t)
        if ok:
            return clock
        frames = t * self.model.params.frame_rate
        if frames.denominator == 1:
            return "%df" % frames.numerator
        sub = self.model.params.sub_frame_rate
        if (frames * sub).denominator == 1 or frames.denominator in (2, 4, 10, 100):
            text = ("%f" % float(frames)).rstrip("0").rstrip(".")
            return text + "f"
        # last resort: round to milliseconds
        rounded = Fraction(round(t * 1000), 1000)
        clock, _ = M.fmt_time(rounded)
        self.degrade(path, "begin/end", "%.6fs" % float(t),
                     "time is not representable exactly at the document frame rate",
                     "quantise cue times to whole frames or milliseconds")
        return clock

    # ---------------------------------------------------------- styles ----
    def _filter_style_attrs(self, el, path):
        for attr in list(el.attrib):
            if not attr.startswith(M.TTS):
                continue
            prop = attr[len(M.TTS):]
            val = el.get(attr)
            if prop not in SUPPORTED_STYLE:
                self._unsupported_style(prop, val, path)
                del el.attrib[attr]
                continue
            if prop == "fontSize" and len(val.split()) == 2:
                w, h = val.split()
                el.set(attr, h)
                self.degrade(path, "tts:fontSize", val,
                             "two-component fontSize (width height) is not in IMSC",
                             "kept the height component (%s); use a single length" % h)
            elif prop == "writingMode" and val in WRITING_MODE_ALIASES:
                el.set(attr, WRITING_MODE_ALIASES[val])
                self.degrade(path, "tts:writingMode", val,
                             "TTML1 writing-mode alias is not in IMSC",
                             "rewritten to canonical '%s'" % WRITING_MODE_ALIASES[val],
                             severity="info")

    def _unsupported_style(self, prop, val, path):
        suggestions = {
            "textShadow": "approximate with tts:textOutline (supported in IMSC 1.1), "
                          "e.g. a small dark outline, or pre-render the effect",
            "position": "position the region with tts:origin/tts:extent instead",
            "fontKerning": "accept default kerning or bake it into the font",
        }
        self.degrade(path, "tts:" + prop, val,
                     "tts:%s is not in the IMSC text profile" % prop,
                     suggestions.get(prop, "remove the property or approximate it "
                                           "with a supported IMSC feature"))

    def _filter_foreign_style_attrs(self, el, path):
        for attr in list(el.attrib):
            if attr.startswith("{%s}" % M.NS["ebutts"]):
                local = attr.split("}", 1)[1]
                val = el.get(attr)
                if "ebutts:" + local == "ebutts:multiRowAlign":
                    if el.get(M.TTS + "textAlign") is None:
                        el.set(M.TTS + "textAlign", val)
                    del el.attrib[attr]
                    self.degrade(path, "ebutts:multiRowAlign", val,
                                 "ebu multiRowAlign is not in IMSC",
                                 "mapped to tts:textAlign='%s' (per-line alignment "
                                 "of ragged lines is lost)" % val)
                elif "ebutts:" + local not in SUPPORTED_FOREIGN_STYLE:
                    del el.attrib[attr]
                    self.degrade(path, "ebutts:" + local, val,
                                 "ebu style extension is not in IMSC",
                                 "remove or approximate with an IMSC feature")
            elif attr.startswith("{%s}" % M.NS["smpte"]):
                local = attr.split("}", 1)[1]
                val = el.get(attr)
                del el.attrib[attr]
                self.degrade(path, "smpte:" + local, val,
                             "SMPTE-TT image/attribute is not in the IMSC text profile",
                             "use the IMSC image profile for background images, or "
                             "replace the image with text")

    # ---------------------------------------------------------- tree ------
    def _filter_element(self, el, path, allow_timing=True):
        self._filter_style_attrs(el, path)
        self._filter_foreign_style_attrs(el, path)
        for attr in list(el.attrib):
            if attr.startswith(M.TTS) or attr.startswith("{"):
                if attr.startswith("{%s}" % M.NS["xml"]) or \
                   attr.startswith("{%s}" % M.NS["ttm"]):
                    continue
                if attr.startswith("{%s}" % M.NS["ebutts"]) or \
                   attr.startswith("{%s}" % M.NS["smpte"]):
                    continue  # already handled
                if attr.startswith(M.TTS):
                    continue
                # foreign attribute in an unknown namespace
                self.degrade(path, M.qname(attr), el.get(attr),
                             "foreign-namespace attribute is not in IMSC",
                             "move the information into ttm metadata or drop it",
                             severity="info")
                del el.attrib[attr]
                continue
            if attr in SUPPORTED_CONTENT_ATTRS:
                if not allow_timing and attr in ("begin", "end", "dur"):
                    continue  # span timing handled in _walk_content
                continue
            if attr == "timeContainer":
                continue  # handled separately
            self.degrade(path, attr, el.get(attr),
                         "attribute is not in the IMSC subset",
                         "remove it or express the intent with supported features",
                         severity="info")
            del el.attrib[attr]

    def _rewrite_p_timing(self, el, path):
        cue = self.cue_by_path.get(path)
        for a in ("begin", "end", "dur"):
            el.attrib.pop(a, None)
        if cue is None:
            return
        el.set("begin", self._fmt_abs_time(cue.begin, path))
        if cue.end is not None:
            el.set("end", self._fmt_abs_time(cue.end, path))

    def _walk_content(self, el, path):
        tag = el.tag.split("}")[-1]
        self._filter_element(el, path, allow_timing=(tag != "span"))
        if tag == "span":
            timed = [a for a in ("begin", "end", "dur") if el.get(a) is not None]
            if timed:
                vals = ", ".join("%s=%r" % (a, el.get(a)) for a in timed)
                for a in timed:
                    del el.attrib[a]
                self.degrade(
                    path, "/".join(timed), vals,
                    "timing on <span> is not permitted in IMSC",
                    "split the text into separate <p> elements carrying the "
                    "timing, or move the timing up to the parent <p>")
        if tag == "p":
            self._rewrite_p_timing(el, path)
        elif tag in ("body", "div"):
            el.attrib.pop("timeContainer", None)
            for a in ("begin", "end", "dur"):
                el.attrib.pop(a, None)
        counts = {}
        for child in list(el):
            ctag = child.tag.split("}")[-1] if isinstance(child.tag, str) else ""
            counts[ctag] = counts.get(ctag, 0) + 1
            cpath = M._path(path, ctag, counts[ctag])
            if ctag in ("set", "animate"):
                props = [M.qname(a) for a in child.attrib if a.startswith(M.TTS)]
                self.degrade(
                    cpath, ctag, child.get("begin") or "",
                     "discrete animation <%s> is not in the IMSC text profile" % ctag,
                     "split the caption into consecutive timed <p> elements that "
                     "change %s at the same offsets" % (", ".join(props) or "style"))
                self._remove_preserving_tail(el, child)
                continue
            if ctag == "metadata":
                known = ("{%s}" % M.NS["tt"], "{%s}" % M.NS["ttm"])
                foreign = [c for c in child.iter()
                           if isinstance(c.tag, str)
                           and not c.tag.startswith(known)]
                if foreign:
                    self.degrade(cpath, "metadata", M.qname(foreign[0].tag),
                                 "foreign metadata vocabulary is not in IMSC",
                                 "keep it in the sidecar workflow or convert to "
                                 "ttm:desc", severity="info")
                    self._remove_preserving_tail(el, child)
                continue
            if ctag in ("div", "p", "span", "br"):
                self._walk_content(child, cpath)

    @staticmethod
    def _remove_preserving_tail(parent, child):
        """Remove child but keep its tail text in the content flow."""
        if child.tail:
            siblings = list(parent)
            idx = siblings.index(child)
            if idx > 0:
                prev = siblings[idx - 1]
                prev.tail = (prev.tail or "") + child.tail
            else:
                parent.text = (parent.text or "") + child.tail
        parent.remove(child)

    def run(self):
        root = self.root
        # root parameters
        root.set(M.TTP + "contentProfiles", IMSC_PROFILE)
        if root.get(M.TTP + "timeBase") == "smpte":
            del root.attrib[M.TTP + "timeBase"]
        head = root.find("tt:head", M.NS)
        if head is not None:
            styling = head.find("tt:styling", M.NS)
            if styling is not None:
                for i, st in enumerate(styling.findall("tt:style", M.NS), 1):
                    self._filter_element(st, "/tt/head/styling/style[%d]" % i)
            layout = head.find("tt:layout", M.NS)
            if layout is not None:
                for i, reg in enumerate(layout.findall("tt:region", M.NS), 1):
                    self._filter_element(reg, "/tt/head/layout/region[%d]" % i)
        body = root.find("tt:body", M.NS)
        if body is not None:
            if any(d.get("timeContainer") == "seq"
                   for d in [body] + body.findall("tt:div", M.NS)):
                self.degrade("/tt/body", "timeContainer", "seq",
                             "IMSC does not allow seq time containers",
                             "flattened to absolute par timing on each <p>",
                             severity="info")
            self._walk_content(body, "/tt/body")
        return self

    def write(self, dst_path):
        ET.indent(self.tree, space="  ")
        self.tree.write(dst_path, encoding="UTF-8", xml_declaration=True)


def migrate_file(src_path, dst_path):
    mig = Migrator(src_path).run()
    mig.write(dst_path)
    return mig.degradations
