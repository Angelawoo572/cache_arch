#!/usr/bin/env bash
# 01_run_spp_trace_dump.sh
#
# Build/run ChampSim with L2 spp_dev candidate logging, then convert the SPP
# event log into a CSV that LSTM_cache_action_predictor.ipynb can train on.
#
# Usage from repo root:
#   TRACE=602.gcc_s-734B bash formal_NN_training/scripts/01_run_spp_trace_dump.sh
#
# Clean recommended run, avoids cumulative patch artifacts:
#   TRACE=602.gcc_s-734B WARMUP=25000000 SIM=25000000 RESET_SPP=1 BUILD=1 PATCH_SPP=1 \
#     bash formal_NN_training/scripts/01_run_spp_trace_dump.sh
#
# If ChampSim already finished and only the final conversion failed:
#   TRACE=602.gcc_s-734B CONVERT_ONLY=1 \
#     bash formal_NN_training/scripts/01_run_spp_trace_dump.sh
#
# Output:
#   formal_NN_training/results/spp_trace_dump/logs/<trace>.spp.log
#   formal_NN_training/results/spp_trace_dump/events/spp_events_<trace>.csv
#   formal_NN_training/results/spp_trace_dump/candidate_table_<trace>.csv
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
CONVERT_ONLY="${CONVERT_ONLY:-0}"

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

if [ "$CONVERT_ONLY" != "1" ]; then
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

if [ "$CONVERT_ONLY" != "1" ]; then
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
fi

if [ ! -s "$EVENT_CSV" ]; then
  echo "[error] no SPP event log found: $EVENT_CSV"
  echo "        If the run already completed, check the trace name or event directory."
  exit 1
fi

if [ "$CONVERT_ONLY" = "1" ] && [ -s "$CAND_CSV" ]; then
  echo "[reuse] existing candidate table: $CAND_CSV"
else
  python3 projects/post_prefetch_filter/scripts/05_events_to_candidate_table.py \
    --trace "$TRACE" \
    --events "$EVENT_CSV" \
    --out "$CAND_CSV" \
    --scope "$SCOPE" \
    --min-confidence "$MIN_CONFIDENCE"
fi

python3 - "$CAND_CSV" "$LSTM_CSV" <<'PY'
import csv
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

if not src.exists() or src.stat().st_size == 0:
    raise SystemExit(f"[error] candidate table missing/empty: {src}")

out_cols = [
    "trace", "event_id", "replay_access_idx", "cycle", "pc", "addr", "hit", "is_store",
    "spp_delta", "spp_conf", "mshr_occupancy", "l2_occupancy",
    "bandwidth_pressure", "semantic_class",
    # Debug / optional labels; notebook ignores unknown columns if not needed.
    "pf_addr", "delta", "spp_confidence", "spp_fill_l2", "spp_issued",
    "outcome_useful", "outcome_duplicate",
]

def get(row, key, default="0"):
    val = row.get(key, default)
    if val is None or val == "":
        return str(default)
    return str(val)

def as_float(row, key, default=0.0):
    try:
        val = row.get(key, default)
        if val is None or val == "":
            return float(default)
        return float(val)
    except Exception:
        return float(default)

rows_written = 0
dst.parent.mkdir(parents=True, exist_ok=True)
with src.open(newline="") as f_in, dst.open("w", newline="") as f_out:
    reader = csv.DictReader(f_in)
    writer = csv.DictWriter(f_out, fieldnames=out_cols)
    writer.writeheader()
    for i, row in enumerate(reader):
        pq_occ = as_float(row, "pq_occupancy", 0.0)
        pq_size = as_float(row, "pq_size", 0.0)
        if pq_size <= 0.0:
            bw = 0.0
        else:
            bw = max(0.0, min(1.0, pq_occ / pq_size))

        out = {
            "trace": get(row, "trace", "unknown"),
            "event_id": get(row, "cycle", i),
            "replay_access_idx": get(row, "replay_access_idx", get(row, "l2_replay_access_idx", get(row, "demand_access_idx", ""))),
            "cycle": get(row, "cycle", i),
            "pc": get(row, "ip", 0),
            "addr": get(row, "addr", 0),
            "hit": get(row, "cache_hit", 0),
            "is_store": "0",
            "spp_delta": get(row, "delta", 0),
            "spp_conf": get(row, "spp_confidence", 0),
            "mshr_occupancy": get(row, "mshr_occupancy", 0),
            "l2_occupancy": "0",
            "bandwidth_pressure": f"{bw:.6f}",
            "semantic_class": "spp_candidate",
            "pf_addr": get(row, "pf_addr", 0),
            "delta": get(row, "delta", 0),
            "spp_confidence": get(row, "spp_confidence", 0),
            "spp_fill_l2": get(row, "spp_fill_l2", 0),
            "spp_issued": get(row, "spp_issued", 0),
            "outcome_useful": get(row, "outcome_useful", 0),
            "outcome_duplicate": get(row, "outcome_duplicate", 0),
        }
        writer.writerow(out)
        rows_written += 1

if rows_written == 0:
    raise SystemExit(f"[error] candidate table has zero data rows: {src}")

print(f"[write] {dst} rows={rows_written} cols={out_cols}")
PY

echo
echo "[done]"
echo "  event log      : $EVENT_CSV"
echo "  candidate table: $CAND_CSV"
echo "  LSTM data      : $LSTM_CSV"
echo "  ChampSim log   : $LOG_FILE"
