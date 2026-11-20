#!/usr/bin/env bash
# End-to-end demo: masters -> straight/premultiplied proxies -> checks.
set -u
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
CASES="gradient edge_glow transparent_color"

echo "== 1. build straight-alpha masters =="
$PY make_masters.py || exit 1

echo "== 2. build proxies (prores 4444) + checkerboard previews =="
for c in $CASES; do
  $PY make_proxy.py --master out/masters/$c --interpretation straight \
      --out out/proxies/${c}_straight.mov || exit 1
  $PY make_proxy.py --master out/masters/$c --interpretation premultiplied \
      --out out/proxies/${c}_premult.mov || exit 1
done

echo "== 3. check every proxy (auto-detects interpretation) =="
fail=0
for c in $CASES; do
  for interp in straight premult; do
    echo "--- ${c}_${interp} ---"
    $PY check_proxy.py --proxy out/proxies/${c}_${interp}.mov \
        --master out/masters/$c --report-dir out/reports/${c}_${interp} \
        || fail=1
  done
done

echo "== 4. negative test: declare the wrong interpretation on purpose =="
if $PY check_proxy.py --proxy out/proxies/edge_glow_straight.mov \
     --master out/masters/edge_glow --declared premultiplied; then
  echo "!! mismatch was NOT detected (unexpected)"
  fail=1
else
  echo "-> mismatch correctly reported (non-zero exit expected here)"
fi

echo
if [ $fail -eq 0 ]; then echo "DEMO OK"; else echo "DEMO FAILED"; exit 1; fi
