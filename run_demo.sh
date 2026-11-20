#!/bin/bash
# End-to-end demo of the four-track unified-clock recording service.
#
#   1. four capture endpoints come up at staggered times (0/2/4/6 s)
#   2. the recorder service records all four tracks on one show clock
#   3. track 3's capture endpoint is killed (outage) and restarted
#   4. the recorder process itself is restarted (persistence check)
#   5. accept.py aligns the four recordings and reports missing frames
#
set -u
cd "$(dirname "$0")"
source gstenv.sh

RUN=run
STATE=state
mkdir -p "$RUN" "$STATE"

cleanup() {
  [ -n "${SVC_PID:-}" ] && kill -INT "$SVC_PID" 2>/dev/null
  for p in "${CAM_PIDS[@]:-}"; do kill "$p" 2>/dev/null; done
  wait 2>/dev/null
}
trap cleanup EXIT

echo "== start 4 capture endpoints (staggered 0/2/4/6 s) =="
CAM_PIDS=()
for t in 0 1 2 3; do
  python3 capture_sim.py --track "$t" --delay $((t * 2)) --run-dir "$RUN" \
    > "cam${t}.log" 2>&1 &
  CAM_PIDS+=($!)
done

echo "== start recorder service =="
python3 -m recorder.service --tracks 4 --seg-seconds 2 \
  --run-dir "$RUN" --state-dir "$STATE" > service.log 2>&1 &
SVC_PID=$!

sleep 14
echo "== SIMULATE OUTAGE: kill capture endpoint of track 3 =="
kill -9 "${CAM_PIDS[3]}" 2>/dev/null
sleep 6
echo "== track 3 capture endpoint reconnects =="
python3 capture_sim.py --track 3 --delay 0 --run-dir "$RUN" \
  > cam3b.log 2>&1 &
CAM_PIDS[3]=$!

sleep 8
echo "== RESTART recorder process (persistence check) =="
kill -INT "$SVC_PID"; wait "$SVC_PID" 2>/dev/null
python3 -m recorder.service --tracks 4 --seg-seconds 2 \
  --run-dir "$RUN" --state-dir "$STATE" >> service.log 2>&1 &
SVC_PID=$!

sleep 8
echo "== stop everything =="
kill -INT "$SVC_PID"; wait "$SVC_PID" 2>/dev/null
SVC_PID=
for p in "${CAM_PIDS[@]}"; do kill "$p" 2>/dev/null; done
wait 2>/dev/null

echo
echo "== ACCEPTANCE: full timeline scan =="
python3 accept.py --state-dir "$STATE" --run-dir "$RUN" --probe \
  --at 10 --at 17 --at 30
