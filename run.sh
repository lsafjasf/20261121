#!/usr/bin/env bash
# Wrapper so the tool works even though the audio stack lives under the
# alternate user site-packages directory on this machine.
set -euo pipefail
export PYTHONPATH="/home/hj/.local/lib/python3.14/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 -m windtool "$@"
