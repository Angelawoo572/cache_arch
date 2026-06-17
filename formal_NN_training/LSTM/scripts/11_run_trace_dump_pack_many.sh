#!/usr/bin/env bash
# Dump SPP/LSTM event tables and pack them for Colab for one or more traces.
#
# Usage:
#   TRACES="623.xalancbmk_s-700B 605.mcf_s-994B" \
#   WARMUP=25000000 SIM=25000000 MAX_JOBS=2 \
#     bash formal_NN_training/LSTM/scripts/11_run_trace_dump_pack_many.sh
#
# Defaults are safe for Sacramento/ece-style repo roots because ROOT is inferred
# from git or pwd. Missing traces are skipped, not fatal.

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

WARMUP="${WARMUP:-25000000}"
SIM="${SIM:-25000000}"
MAX_JOBS="${MAX_JOBS:-1}"
FORCE_DUMP="${FORCE_DUMP:-0}"
BUILD="${BUILD:-0}"
PATCH_SPP="${PATCH_SPP:-0}"
RESET_SPP="${RESET_SPP:-1}"

# Space-separated list. If omitted, use a small candidate list and skip missing.
TRACES_STR="${TRACES:-603.bwaves_s-2609B 607.cactuBSSN_s-2421B 621.wrf_s-8065B 623.xalancbmk_s-700B 625.x264_s-18B}"

LOG_ROOT="$ROOT/formal_NN_training/results/new_trace_pipeline/logs"
mkdir -p "$LOG_ROOT" "$ROOT/formal_NN_training/data/upload"

verify_events () {
  local trace="$1"
  local events="$ROOT/formal_NN_training/data/generated/lstm_events_${trace}.csv"
  EVENTS_PATH="$events" python3 - <<'PY'
import csv
import os
from pathlib import Path
p = Path(os.environ["EVENTS_PATH"])
if not p.exists() or p.stat().st_size == 0:
    raise SystemExit(f"[error] missing events: {p}")
with p.open(newline="") as f:
    r = csv.DictReader(f)
    n = blank = nonblank = 0
    examples = []
    for row in r:
        n += 1
        if row.get("replay_access_idx", "") == "":
            blank += 1
        else:
            nonblank += 1
        if len(examples) < 5:
            examples.append((row.get("event_id"), row.get("replay_access_idx"), row.get("addr_int", row.get("addr", ""))))
        if n >= 100000:
            break
print("[verify events]", p)
print("  checked:", n)
print("  blank:", blank)
print("  nonblank:", nonblank)
print("  examples:", examples)
if n == 0 or blank != 0 or nonblank == 0:
    raise SystemExit("[error] bad replay_access_idx in events")
PY
}

run_one () {
  local trace="$1"
  local tag="${trace%%.*}"
  local trace_file="$ROOT/traces/${trace}.champsimtrace.xz"
  local events="$ROOT/formal_NN_training/data/generated/lstm_events_${trace}.csv"

  if [ ! -s "$trace_file" ]; then
    echo "[skip missing trace] $trace_file"
    return 0
  fi

  echo "============================================================"
  echo "[trace] $trace"
  echo "warmup/sim : $WARMUP / $SIM"
  echo "tag        : $tag"
  echo "============================================================"

  if [ "$FORCE_DUMP" = "1" ] && [ -s "$events" ]; then
    local bak="$events.backup_$(date +%Y%m%d_%H%M%S)"
    echo "[backup existing events] $events -> $bak"
    mv "$events" "$bak"
  fi

  if [ -s "$events" ]; then
    echo "[skip dump] existing events: $events"
  else
    echo "[run dump] $trace"
    TRACE="$trace" \
    WARMUP="$WARMUP" \
    SIM="$SIM" \
    BUILD="$BUILD" \
    PATCH_SPP="$PATCH_SPP" \
    RESET_SPP="$RESET_SPP" \
      bash formal_NN_training/LSTM/scripts/01_run_spp_trace_dump.sh \
      > "$LOG_ROOT/${trace}.dump.log" 2>&1
  fi

  verify_events "$trace"

  echo "[pack for Colab] $trace -> upload/$tag"
  TRACE="$trace" \
  UPLOAD_TAG="$tag" \
    bash formal_NN_training/LSTM/scripts/05_pack_lstm_events_for_colab.sh \
    > "$LOG_ROOT/${trace}.pack.log" 2>&1

  echo "[upload files] formal_NN_training/data/upload/$tag"
  ls -lh "$ROOT/formal_NN_training/data/upload/$tag" || true
}

running=0
for trace in $TRACES_STR; do
  run_one "$trace" &
  running=$((running + 1))
  if [ "$running" -ge "$MAX_JOBS" ]; then
    wait -n
    running=$((running - 1))
  fi
done
wait

echo "============================================================"
echo "[done] upload dirs"
echo "============================================================"
find formal_NN_training/data/upload -maxdepth 2 -type f \( -name '*.part_*' -o -name '*.csv.gz' \) | sort | xargs -r ls -lh
