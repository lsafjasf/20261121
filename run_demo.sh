#!/usr/bin/env bash
# End-to-end demonstration on the synthetic mixed-cadence archive clip.
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1. build mixed 29.97i sample (pulldown / cut / native / cut / pulldown)"
python3 -m ivtc.cli --sample sample --outdir artifacts/sample

echo
echo "== 2. label cadence segments and cut points"
python3 -m ivtc.cli --sample analyze --report artifacts/report.json

echo
echo "== 3. field order around the cuts + output-frame provenance"
python3 -m ivtc.cli --sample fieldmap

echo
echo "== 4. inverse-telecine confident pulldown segments only"
python3 -m ivtc.cli --sample reconstruct --outdir artifacts/reconstructed

echo
echo "== 5. verify provenance and motion-edge combing (incl. negative control)"
python3 -m ivtc.cli --sample verify --report artifacts/verification.json

echo
echo "== 6. unit tests"
python3 -m unittest tests.test_ivtc -v

echo
echo "artifacts:"
ls -1 artifacts/ artifacts/sample artifacts/reconstructed | sed 's/^/  /'
