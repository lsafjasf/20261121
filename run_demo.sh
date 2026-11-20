#!/usr/bin/env bash
# 离线运行全部快门积分演示，输出到 artifacts/demo。
# 本机 numpy/Pillow/matplotlib 装在用户 site-packages，通过 PYTHONPATH 引入。
set -euo pipefail

cd "$(dirname "$0")"

USER_SITE="/home/hj/.local/lib/python3.14/site-packages"
if [ -d "$USER_SITE" ]; then
  export PYTHONPATH="$USER_SITE${PYTHONPATH:+:$PYTHONPATH}"
fi

python3 run_tests.py
python3 -m shutter_demo.cli demo --out artifacts/demo

echo
echo "产物索引:"
find artifacts/demo -name "contact_sheet.png" -o -name "report.txt" | sort
