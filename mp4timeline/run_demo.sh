#!/usr/bin/env bash
# Build the reference fixture, run the three-layer investigation, then self-test.
set -euo pipefail

cd "$(dirname "$0")"
OUT="${1:-output}"

echo "==> 1/3 building fixture (pre-roll, paused gaps, late audio, B-frames)"
python3 -m tools.make_fixture "$OUT"

echo "==> 2/3 investigating timing and exporting tables + frame sheets"
python3 -m tools.investigate "$OUT/fixture.mp4" "$OUT/investigation"

echo "==> 3/3 running self-test (cross-check vs ffprobe)"
python3 -m tools.selftest "$OUT/fixture.mp4"

echo
echo "Done. Key artifacts under $OUT/investigation:"
echo "  tables/                       CSV/JSON/Markdown for third-party diffing"
echo "  contact_decode_order.jpg      decode (DTS) order with reorder tags"
echo "  contact_display_order.jpg     display (PTS) order with movie timestamps"
echo "  frames_labeled/               per-frame labeled thumbnails"
