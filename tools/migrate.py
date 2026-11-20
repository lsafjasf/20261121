"""Batch driver: migrate TTML files to the IMSC subset and verify the result
with rendered-frame comparison.

Usage: python3 migrate.py [-o OUTDIR] [--no-verify] FILE_OR_DIR...
"""
import argparse
import glob
import json
import os
import sys

import imsc_emit
import verify as V


def collect_inputs(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "*.ttml"))))
        else:
            files.append(p)
    return [f for f in files if not f.endswith(".imsc.ttml")]


def md_escape(text):
    return str(text).replace("|", "&#124;").replace("\n", " ")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", default="out")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    summary = []
    for src in collect_inputs(args.inputs):
        name = os.path.splitext(os.path.basename(src))[0]
        imsc_path = os.path.join(args.out, name + ".imsc.ttml")
        degradations = imsc_emit.migrate_file(src, imsc_path)
        entry = {"source": src, "imsc": imsc_path,
                 "degradations": degradations, "verify": None}
        if not args.no_verify:
            vdir = os.path.join(args.out, "verify", name)
            entry["verify"] = V.verify_pair(src, imsc_path, vdir)
        summary.append(entry)
        n_warn = sum(1 for d in degradations if d["severity"] != "info")
        print("%s -> %s  (%d degradations, %d non-info)"
              % (src, imsc_path, len(degradations), n_warn))

    json_path = os.path.join(args.out, "report.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    md_path = os.path.join(args.out, "report.md")
    with open(md_path, "w") as fh:
        fh.write(render_markdown(summary))
    print("reports: %s, %s" % (json_path, md_path))
    return 0


def render_markdown(summary):
    lines = ["# TTML -> IMSC migration report", ""]
    for entry in summary:
        lines.append("## `%s`" % os.path.basename(entry["source"]))
        lines.append("")
        lines.append("Migrated document: `%s`" % entry["imsc"])
        lines.append("")
        degs = entry["degradations"]
        if degs:
            lines.append("### Degradations (%d)" % len(degs))
            lines.append("")
            lines.append("| Severity | Element | Attribute | Value | Reason | Suggestion |")
            lines.append("|---|---|---|---|---|---|")
            for d in degs:
                lines.append("| %s | `%s` | `%s` | `%s` | %s | %s |" % (
                    d["severity"], d["element"], d["attribute"],
                    md_escape(d["value"])[:60],
                    md_escape(d["reason"]), md_escape(d["suggestion"])))
        else:
            lines.append("No degradations: every effect maps to the IMSC subset.")
        lines.append("")
        v = entry.get("verify")
        if v:
            lines.append("### Rendered-frame verification")
            lines.append("")
            chk = v["checks"]
            lines.append("- frame diff (source vs migrated): **%s**"
                         % chk["frame_diff"]["changed_frames"])
            lines.append("- overlap incident groups: **%d**"
                         % chk["overlap"]["incident_groups"])
            frac = chk.get("fractional", [])
            passed = sum(1 for c in frac if c["ok"])
            lines.append("- fractional in-points: **%d/%d passed**"
                         % (passed, len(frac)))
            flash = chk["flicker"]["single_frame_cues"]
            lines.append("- flash-frame risks: **%s**"
                         % (", ".join(flash) if flash else "none"))
            errs = [f for f in v["findings"] if f["severity"] == "error"]
            if errs:
                lines.append("")
                lines.append("Errors:")
                for f in errs:
                    lines.append("- %s" % f["message"])
            if v["keyframes"]:
                lines.append("")
                lines.append("Key frames: `%s`"
                             % os.path.dirname(v["keyframes"][0]))
            lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
