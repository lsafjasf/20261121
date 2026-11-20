#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if ! python3 -c "import numpy, scipy, soundfile" >/dev/null 2>&1; then
  user_home="$(python3 -c 'import os, pwd; print(pwd.getpwuid(os.getuid()).pw_dir)')"
  for site_packages in "$user_home"/.local/lib/python*/site-packages; do
    if [ -d "$site_packages/numpy" ]; then
      export PYTHONPATH="$site_packages${PYTHONPATH:+:$PYTHONPATH}"
      break
    fi
  done
fi

if ! python3 -c "import numpy, scipy, soundfile" >/dev/null 2>&1; then
  echo "Missing audio dependencies. Run: python3 -m pip install -r requirements.txt" >&2
  exit 1
fi

python3 app.py --host 127.0.0.1 --port "${PORT:-8787}"

