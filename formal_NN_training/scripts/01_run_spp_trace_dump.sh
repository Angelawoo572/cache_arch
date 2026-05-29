#!/usr/bin/env bash
# 01_run_spp_trace_dump.sh
#
# Build/run ChampSim with L2 spp_dev candidate logging, then convert the SPP
# event log into a CSV that LSTM_cache_action_predictor.ipynb can train on.
#
# Usage from repo root:
#   TRACE=602.gcc_s-734B bash formal_NN_training/scripts/01_run_spp_trace_dump.sh
#
# Common overrides:
#   WARMUP=25000000 SIM=25000000 TRACE=605.mcf_s-994B \
#   RESET_SPP=1 BUILD=1 bash formal_NN_training/scripts/01_run_spp_trace_dump.sh
#
# Output:
#   formal_NN_training/results/spp_trace_dump/logs/<trace>.spp.log
#   formal_NN_training/results/spp_trace_dump/events/spp_events_<trace>.csv
#   formal_NN_training/data/generated/lstm_events_<trace>.csv

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

TRACE="${TRACE:-602.gcc_s-734B}"
TRACE_DIR="${TRACE_DIR:-$ROOT/traces}"
CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
WARMUP="${WARMUP:-25000000}"
SIM="${SIM:-25000000}"
JOBS="${JOBS:-8}"
MIN_CONFIDENCE="${MIN_CONFIDENCE:-90}"
SCOPE="${SCOPE:-spp_actual_issue}"
BUILD="${BUILD:-1}"
PATCH_SPP="${PATCH_SPP:-1}"
RESET_SPP="${RESET_SPP:-0}"

OUT_ROOT="$ROOT/formal_NN_training/results/spp_trace_dump"
LOG_DIR="$OUT_ROOT/logs"
EVENT_DIR="$OUT_ROOT/events"
DATA_DIR="$ROOT/formal_NN_training/data/generated"
CFG_DIR="$ROOT/formal_NN_training/_cfg"
mkdir -p "$LOG_DIR" "$EVENT_DIR" "$DATA_DIR" "$CFG_DIR"

TR_FILE="$TRACE_DIR/${TRACE}.champsimtrace.xz"
EVENT_CSV="$EVENT_DIR/spp_events_${TRACE}.csv"
CAND_CSV="$OUT_ROOT/candidate_table_${TRACE}.csv"
LSTM_CSV="$DATA_DIR/lstm_events_${TRACE}.csv"
LOG_FILE="$LOG_DIR/${TRACE}.spp.log"
SPP_BIN="$CHAMP_DIR/bin/champsim.l2_spp_cand"

if [ ! -d "$CHAMP_DIR" ]; then
  echo "[error] ChampSim directory missing: $CHAMP_DIR"
  echo "        Run your setup script first, e.g. projects/legacy_gru_prefetch/scripts/setup_champsim.sh"
  exit 1
fi
if [ ! -f "$TR_FILE" ]; then
  echo "[error] trace missing: $TR_FILE"
  exit 1
fi

if [ "$PATCH_SPP" = "1" ]; then
  echo "[patch] spp_dev candidate logger"
  RESET_SPP="$RESET_SPP" bash projects/post_prefetch_filter/scripts/04_patch_spp_candidate_logger.sh
fi

CFG_PATH="${SPP_CONFIG:-$CFG_DIR/cfg_l2_spp.json}"
if [ ! -f "$CFG_PATH" ]; then
  cat > "$CFG_PATH" <<'JSON'
{
  "ooo_cpu": [
    {
      "L2C": { "prefetcher": "spp_dev" }
    }
  ],
  "LLC": { "replacement": "lru" }
}
JSON
fi

if [ "$BUILD" = "1" ] || [ ! -x "$SPP_BIN" ]; then
  echo "[build] ChampSim L2 spp_dev binary"
  cd "$CHAMP_DIR"
  rm -f bin/champsim
  python3 ./config.sh "$CFG_PATH"
  make -j"$JOBS"
  cp bin/champsim "$SPP_BIN"
  cd "$ROOT"
fi

if [ ! -x "$SPP_BIN" ]; then
  echo "[error] SPP binary missing after build: $SPP_BIN"
  exit 1
fi

run_with_heartbeat () {
  local log="$1"; shift
  "$@" > "$log" 2>&1 &
  local pid=$!
  local seconds=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    seconds=$((seconds + 30))
    local last
    last=$(tail -1 "$log" 2>/dev/null | head -c 160 || true)
    printf "  ...running trace=%s elapsed=%ds last='%s'\n" "$TRACE" "$seconds" "$last"
  done
  wait "$pid"
}

echo "============================================================"
echo "SPP TRACE DUMP"
echo "repo       : $ROOT"
echo "trace      : $TRACE"
echo "warmup/sim : $WARMUP / $SIM"
echo "binary     : $SPP_BIN"
echo "event csv  : $EVENT_CSV"
echo "lstm csv   : $LSTM_CSV"
echo "============================================================"

SPP_CAND_LOG="$EVENT_CSV" run_with_heartbeat "$LOG_FILE" \
  "$SPP_BIN" \
  --warmup-instructions "$WARMUP" \
  --simulation-instructions "$SIM" \
  "$TR_FILE"

if [ ! -s "$EVENT_CSV" ]; then
  echo "[error] no SPP event log was produced: $EVENT_CSV"
  echo "        See log: $LOG_FILE"
  exit 1
fi

python3 projects/post_prefetch_filter/scripts/05_events_to_candidate_table.py \
  --trace "$TRACE" \
  --events "$EVENT_CSV" \
  --out "$CAND_CSV" \
  --scope "$SCOPE" \
  --min-confidence "$MIN_CONFIDENCE"

python3 - "$CAND_CSV" "$LSTM_CSV" <<'PY'
import sys
from pathlib import Path
import pandas as pd

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
df = pd.read_csv(src)
if df.empty:
    raise SystemExit(f"[error] candidate table is empty: {src}")

out = pd.DataFrame()
out["trace"] = df.get("trace", "unknown")
out["event_id"] = df.get("cycle", range(len(df)))
out["cycle"] = df.get("cycle", range(len(df)))
out["pc"] = df["ip"]
out["addr"] = df["addr"]
out["hit"] = df.get("cache_hit", 0)
out["is_store"] = 0
out["spp_delta"] = df.get("delta", 0)
out["spp_conf"] = df.get("spp_confidence", 0)
out["mshr_occupancy"] = df.get("mshr_occupancy", 0)
out["l2_occupancy"] = 0
pq_occ = pd.to_numeric(df.get("pq_occupancy", 0), errors="coerce").fillna(0)
pq_size = pd.to_numeric(df.get("pq_size", 0), errors="coerce").fillna(0).replace(0, 1)
out["bandwidth_pressure"] = (pq_occ / pq_size).clip(0, 1)
out["semantic_class"] = "spp_candidate"

# Keep useful debug columns too. The LSTM notebook ignores unknown columns.
for col in ["pf_addr", "delta", "spp_confidence", "spp_fill_l2", "spp_issued", "outcome_useful", "outcome_duplicate"]:
    if col in df.columns:
        out[col] = df[col]

dst.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(dst, index=False)
print(f"[write] {dst} rows={len(out)} cols={list(out.columns)}")
PY

echo
echo "[done]"
echo "  event log      : $EVENT_CSV"
echo "  candidate table: $CAND_CSV"
echo "  LSTM data      : $LSTM_CSV"
echo "  ChampSim log   : $LOG_FILE"
