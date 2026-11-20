"""Field-ordering dumps around cuts and output-frame provenance tables."""

from __future__ import annotations

from .detector import Analysis


def dump_cut_neighbourhood(analysis: Analysis, context: int = 4) -> str:
    """Show parity/provenance/cadence residue of fields around every cut."""
    fields = analysis.fields
    residues = analysis.residues
    lines = []
    for cut in analysis.cuts:
        c = cut.field_index
        lo = max(0, c - context)
        hi = min(len(fields), c + context + 1)
        lines.append(
            f"CUT at field {c} (interlaced frame {cut.frame_index}), "
            f"frame_diff={cut.frame_diff:.2f}")
        lines.append("  field parity  residue  scene origin            film/native")
        for i in range(lo, hi):
            f = fields[i]
            r = residues[i]
            r_txt = "  -   " if r is None else f"{r:6.2f}"
            origin = f.source.get("origin", f.source.get("input", "?"))
            scene = f.source.get("scene", "-")
            detail = (f"film={f.source['film_frame']}"
                      if "film_frame" in f.source
                      else f"native={f.source.get('native_field', '-')}")
            marker = "  <-- CUT" if i == c else ""
            lines.append(
                f"  {i:>5} {f.parity:<6} {r_txt}  "
                f"{scene:<3} {origin:<16} {detail}{marker}")
        lines.append("")
    return "\n".join(lines)


def dump_output_provenance(reconstructed) -> str:
    lines = ["output_frame  top_field  bottom_field  anchor  confidence  "
             "source_interlaced_frames"]
    for item in reconstructed:
        src_frames = sorted({item.top_field // 2, item.bottom_field // 2})
        lines.append(
            f"  {item.frame.index:>6}       {item.top_field:>6}      "
            f"{item.bottom_field:>6}       {item.anchor:>5}    "
            f"{item.confidence:.2f}        {src_frames}")
    return "\n".join(lines)
