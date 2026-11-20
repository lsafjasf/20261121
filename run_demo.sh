#!/usr/bin/env bash
# 离线一键生成：配置 + 实测 + 演示音频 + 报告，再跑自检。
set -euo pipefail
cd "$(dirname "$0")"
python3 -m hearing_demo.cli all --audiogram data/sample_audiogram.json --out out
python3 tests/selftest.py

