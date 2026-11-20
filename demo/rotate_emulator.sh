#!/usr/bin/env bash
# Emulator demo driver: rotates the emulated device through all four orientations
# while the app is recording, captures a screenshot at each step, then pulls the
# recorded mp4 + sidecar metadata for verification.
#
# Usage:
#   1. Install and launch the app on a running emulator, start recording.
#   2. ./demo/rotate_emulator.sh
#   3. Inspect demo/emulator_out/ and verify (see demo/README.md).
set -euo pipefail

OUT="${1:-demo/emulator_out}"
ADB="${ADB:-adb}"
mkdir -p "$OUT"

echo "== disabling auto-rotate, forcing user_rotation"
"$ADB" shell settings put system accelerometer_rotation 0

for ROT in 0 1 2 3; do
    DEG=$((ROT * 90))
    echo "== rotating to ${DEG} deg (user_rotation=$ROT)"
    "$ADB" shell settings put system user_rotation "$ROT"
    sleep 3
    "$ADB" exec-out screencap -p > "$OUT/screen_R${DEG}.png"
    echo "   screenshot: $OUT/screen_R${DEG}.png"
done

"$ADB" shell settings put system accelerometer_rotation 1

echo "== pulling recordings and sidecar metadata"
"$ADB" pull /sdcard/Android/data/com.example.rotatex.app/files/Movies/ "$OUT/" || true

echo
echo "Verify:"
echo "  - screenshots screen_R*.png: preview arrow stays upright, HUD shows MATCH"
echo "  - ffprobe -show_streams $OUT/Movies/rotatex_*.mp4 | grep -i rotate"
echo "    -> constant rotation for the whole file (locked target orientation)"
echo "  - ffprobe -show_streams $OUT/Movies/rotatex_*.mp4 | grep -c codec_type=audio"
echo "    -> 1 audio stream, continuous"
echo "  - cat $OUT/Movies/rotatex_*.json -> cropsMatch=true on every event"
