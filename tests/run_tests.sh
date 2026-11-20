#!/usr/bin/env bash
# End-to-end checks for footbatch.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$ROOT/out/tests"
BIN="$ROOT/build/footbatch"
FAILURES=0

rm -rf "$WORK"
mkdir -p "$WORK"

pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1"; FAILURES=$((FAILURES + 1)); }

# Resolve the library root from the strategy into the test work dir.
STRATEGY="$WORK/strategy.json"
sed "s|\"out/library\"|\"$WORK/library\"|" "$ROOT/examples/strategy.json" > "$STRATEGY"

"$BIN" genlib --strategy "$STRATEGY" >/dev/null || { echo "genlib failed"; exit 1; }

# --- Test 1: one-shot render -------------------------------------------------
"$BIN" render --strategy "$STRATEGY" --events "$ROOT/examples/events_walk.json" \
    --clip-id walk --out "$WORK/oneshot" >/dev/null \
    && pass "one-shot render" || fail "one-shot render"

[ -s "$WORK/oneshot/walk.wav" ] && pass "wav produced" || fail "wav produced"
[ -s "$WORK/oneshot/walk.manifest.json" ] && pass "manifest produced" \
    || fail "manifest produced"

# --- Test 2: resumed render must be bit identical ----------------------------
# Simulate a crash after the final block: keep the checkpoint + raw/selection
# sidecars but remove the finished wav, then resume and rebuild from disk.
rm -rf "$WORK/resumed"
"$BIN" render --strategy "$STRATEGY" --events "$ROOT/examples/events_walk.json" \
    --clip-id walk --out "$WORK/resumed" >/dev/null \
    && pass "checkpoint + sidecars written" || fail "checkpoint + sidecars written"
rm -f "$WORK/resumed/walk.wav" "$WORK/resumed/walk.manifest.json"

"$BIN" render --strategy "$STRATEGY" --events "$ROOT/examples/events_walk.json" \
    --clip-id walk --out "$WORK/resumed" --resume >/dev/null \
    && pass "resumed render rebuilds from checkpoint" \
    || fail "resumed render rebuilds from checkpoint"

# Also exercise a genuine mid-render interruption: rewind the checkpoint to an
# earlier block boundary and truncate the sidecars to match, then resume.
rm -rf "$WORK/interrupted"
"$BIN" render --strategy "$STRATEGY" --events "$ROOT/examples/events_run.json" \
    --clip-id run --out "$WORK/run_ref" >/dev/null
"$BIN" render --strategy "$STRATEGY" --events "$ROOT/examples/events_run.json" \
    --clip-id run --out "$WORK/interrupted" >/dev/null
TARGET_BLOCK=$(python3 - "$WORK/interrupted/run.checkpoint" <<'PY'
import struct, sys
with open(sys.argv[1], "r+b") as f:
    data = f.read()
    magic, version, frames, ev, peak, resumes, sel = struct.unpack("<8sIQIIIQ", data)
    assert len(data) == struct.calcsize("<8sIQIIIQ"), len(data)
    block = 256
    target = (frames // block) * block - block * 8
    target = max(block, target)
    f.seek(12)
    f.write(struct.pack("<Q", target))
print(target)
PY
)
CHANNELS=2
RAW_KEEP=$(( TARGET_BLOCK * CHANNELS * 4 ))
truncate -s "$RAW_KEEP" "$WORK/interrupted/run.f32le"
# Selection log: drop records beyond the rewound point (all run events trigger
# in the first few blocks; keep whatever the checkpoint event counter allows).
EV_COUNT=$(python3 - "$WORK/interrupted/run.checkpoint" <<'PY'
import struct, sys
with open(sys.argv[1], "rb") as f:
    f.seek(20)
    print(struct.unpack("<I", f.read(4))[0])
PY
)
NDJSON="$WORK/interrupted/run.selections.ndjson"
head -n "$EV_COUNT" "$NDJSON" > "$NDJSON.tmp" && mv "$NDJSON.tmp" "$NDJSON"
rm -f "$WORK/interrupted/run.wav" "$WORK/interrupted/run.manifest.json"

"$BIN" render --strategy "$STRATEGY" --events "$ROOT/examples/events_run.json" \
    --clip-id run --out "$WORK/interrupted" --resume >/dev/null \
    && pass "resume after rewound block-level checkpoint" \
    || fail "resume after rewound block-level checkpoint"

"$BIN" verify --a "$WORK/run_ref/run.wav" --b "$WORK/interrupted/run.wav" \
    && pass "interrupted-then-resumed wav matches one-shot" \
    || fail "interrupted-then-resumed wav matches one-shot"

"$BIN" verify --a "$WORK/oneshot/walk.wav" --b "$WORK/resumed/walk.wav" \
    && pass "resumed wav is bit-identical to one-shot" \
    || fail "resumed wav is bit-identical to one-shot"

if cmp -s "$WORK/oneshot/walk.selections.ndjson" "$WORK/resumed/walk.selections.ndjson"; then
    pass "selection logs identical"
else
    fail "selection logs identical"
fi

# --- Test 3: fast run triggers overlap and retains every voice ---------------
"$BIN" render --strategy "$STRATEGY" --events "$ROOT/examples/events_run.json" \
    --clip-id run --out "$WORK/run" >/dev/null || fail "fast-run render"
OVERLAPS=$(grep -c '"overlapped": true' "$WORK/run/run.manifest.json" || true)
[ "$OVERLAPS" -ge 4 ] && pass "fast run marks $OVERLAPS overlapping steps" \
    || fail "fast run marks overlaps (found $OVERLAPS)"
PEAK=$(grep '"peak_voices"' "$WORK/run/run.manifest.json" | grep -o '[0-9]\+')
[ "$PEAK" -ge 2 ] && pass "overlapping voices coexist (peak=$PEAK)" \
    || fail "overlapping voices coexist (peak=$PEAK)"

# The tail of the first step must survive: output extends beyond the last event
# by at least one full deformed source length.
DUR=$(grep '"duration_sec"' "$WORK/run/run.manifest.json" | grep -o '[0-9.]\+' | head -1)
python3 - "$DUR" <<'PY' || fail "tail retained after last event"
import sys
assert float(sys.argv[1]) > 0.98 + 0.4, sys.argv[1]
PY
pass "tail retained after last event"

# --- Test 4: simultaneous events share one frame -----------------------------
"$BIN" render --strategy "$STRATEGY" \
    --events "$ROOT/examples/events_simultaneous.json" \
    --clip-id sim --out "$WORK/sim" >/dev/null || fail "simultaneous render"
SIM_PEAK=$(grep '"peak_voices"' "$WORK/sim/sim.manifest.json" | grep -o '[0-9]\+')
[ "$SIM_PEAK" -ge 3 ] && pass "three simultaneous voices mixed (peak=$SIM_PEAK)" \
    || fail "three simultaneous voices mixed (peak=$SIM_PEAK)"

# --- Test 5: unknown surface is rejected before any audio is written ---------
rm -rf "$WORK/bad"
if "$BIN" render --strategy "$STRATEGY" \
        --events "$ROOT/examples/events_bad_surface.json" \
        --clip-id bad --out "$WORK/bad" 2>"$WORK/bad.err"; then
    fail "unknown surface rejected"
else
    grep -q "unknown surface" "$WORK/bad.err" \
        && pass "unknown surface rejected with clear error" \
        || fail "unknown surface rejected with clear error"
fi
[ ! -e "$WORK/bad/bad.f32le" ] && pass "no partial output on validation error" \
    || fail "no partial output on validation error"

# --- Test 6: multi-clip batch -----------------------------------------------
"$BIN" render --strategy "$STRATEGY" --batch "$ROOT/examples/batch.json" \
    --out "$WORK/batch" >/dev/null || fail "batch render"
[ -s "$WORK/batch/scene_walk.wav" ] && [ -s "$WORK/batch/scene_run.wav" ] \
    && pass "batch renders every clip" || fail "batch renders every clip"

# --- Test 7: mono channel layout --------------------------------------------
MONO_STRATEGY="$WORK/strategy_mono.json"
sed 's/"stereo"/"mono"/; s|'"$WORK"'/library|'"$WORK"'/library_mono|' \
    "$STRATEGY" > "$MONO_STRATEGY"
"$BIN" genlib --strategy "$MONO_STRATEGY" >/dev/null
"$BIN" render --strategy "$MONO_STRATEGY" \
    --events "$ROOT/examples/events_simultaneous.json" \
    --clip-id mono --out "$WORK/mono" >/dev/null \
    && pass "mono render" || fail "mono render"
grep -q '"wav_channel_mask": 0' "$WORK/mono/mono.manifest.json" \
    && pass "mono wav uses channel mask 0" || fail "mono wav uses channel mask 0"

echo
if [ "$FAILURES" -eq 0 ]; then
    echo "ALL TESTS PASSED"
else
    echo "$FAILURES TEST(S) FAILED"
    exit 1
fi
