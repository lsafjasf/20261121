#!/usr/bin/env bash
# 端到端演示: dummy 后端的 JACK 服务器 + fade_router + 虚拟设备场景 + 录音校验
set -euo pipefail
cd "$(dirname "$0")"

export JROOT="$PWD/.jack/root"
export LD_LIBRARY_PATH="$JROOT/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export JACK_DRIVER_DIR="$JROOT/usr/lib/x86_64-linux-gnu/jack"

mkdir -p build logs
make

# ---- 启动 JACK 服务器(若未运行) ----
JACKD_STARTED=""
if ! pgrep -f 'jackd -d dummy' >/dev/null 2>&1; then
    "$JROOT/usr/bin/jackd" -d dummy -r 48000 -p 256 >logs/jackd.log 2>&1 &
    echo $! > logs/jackd.pid
    JACKD_STARTED=1
    sleep 2
fi

PIDS=()
cleanup() {
    for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
    wait 2>/dev/null || true
    if [ -n "$JACKD_STARTED" ]; then
        kill "$(cat logs/jackd.pid)" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ---- 场景 1: 设备接力(重叠淡化迁移) + 设备消失 + 新设备接入 ----
echo "=== 场景 1: 接力迁移与设备消失 ==="
./build/fade_router --fade-ms 10 >logs/router1.log 2>&1 &
PIDS+=($!)
sleep 0.5
./build/recorder logs/rec1.raw 11 >logs/rec1.log 2>&1 &
PIDS+=($!)
sleep 0.5
./build/virtual_device stable vdevA 4.0 2>logs/vdevA.log &   # t=0..4   设备 A
PIDS+=($!)
sleep 2
./build/virtual_device stable vdevB 4.0 2>logs/vdevB.log &   # t=2..6   设备 B, 与 A 重叠 -> 淡化迁移
PIDS+=($!)
sleep 5                                                      # t=7: B 已消失 1s
./build/virtual_device stable vdevC 3.0 2>logs/vdevC.log &   # t=7..10  新设备 C 接入 -> 淡化迁移
PIDS+=($!)
wait "${PIDS[@]: -1}" 2>/dev/null || true
sleep 2

echo "--- 场景 1 路由日志(尾) ---"
tail -n 6 logs/router1.log
echo "--- 场景 1 录音校验 ---"
python3 scripts/verify_recording.py logs/rec1.raw

kill "${PIDS[0]}" 2>/dev/null || true   # 停掉路由器, 场景 2 重新启动
sleep 0.5

# ---- 场景 2: 松动反复重连 ----
echo "=== 场景 2: 松动反复重连 ==="
./build/fade_router --fade-ms 10 >logs/router2.log 2>&1 &
PIDS+=($!)
sleep 0.5
./build/recorder logs/rec2.raw 11 >logs/rec2.log 2>&1 &
PIDS+=($!)
sleep 0.5
./build/virtual_device flaky vdevX 9 60 200 7 2>logs/vdevX.log &
PIDS+=($!)
sleep 11.5

echo "--- 场景 2 路由日志(尾) ---"
tail -n 6 logs/router2.log
echo "--- 场景 2 录音校验 ---"
python3 scripts/verify_recording.py logs/rec2.raw

echo "=== 演示完成 ==="
