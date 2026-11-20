#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
if ! "${PYTHON_BIN}" -c "import numpy, cv2, matplotlib, imageio_ffmpeg" >/dev/null 2>&1; then
  shopt -s nullglob
  for site_dir in \
    "$HOME"/.local/lib/python*/site-packages \
    "/home/$(id -un)"/.local/lib/python*/site-packages; do
    if [ -d "${site_dir}" ]; then
      export PYTHONPATH="${site_dir}${PYTHONPATH:+:${PYTHONPATH}}"
      if "${PYTHON_BIN}" -c "import numpy, cv2, matplotlib, imageio_ffmpeg" >/dev/null 2>&1; then
        break
      fi
    fi
  done
fi

exec "${PYTHON_BIN}" run_experiment.py "$@"
