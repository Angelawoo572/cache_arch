#!/usr/bin/env bash
# Collect exact trace-start retired-instruction prefixes once for both h8/h16.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/formal_NN_training/experiments/602_lstm_stride_live_inference"
OFFLINE="$ROOT/formal_NN_training/experiments/602_offline_lstm_stride"
CONFIG="${CONFIG:-$EXP/config/training_prefix_sweep.json}"
RUN_ID="${RUN_ID:-602_gcc_stride_prefix_seed7}"
RUN_DIR="${RUN_DIR:-$EXP/runs/$RUN_ID}"
CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
TRACE_FILE="${TRACE_FILE:-$ROOT/traces/602.gcc_s-734B.champsimtrace.xz}"
BIN="${BIN:-$CHAMP_DIR/bin/champsim.602_offline_replay}"
BUDGETS="${BUDGETS:-all}"
FORCE="${FORCE:-0}"
BUILD="${BUILD:-1}"
DRY_RUN="${DRY_RUN:-0}"
COMMAND="${1:-collect}"

EVENT_DIR="$RUN_DIR/training_events"
PREFIX_DIR="$RUN_DIR/training_prefixes"
EVAL_DIR="$RUN_DIR/evaluation"
LOG_DIR="$RUN_DIR/logs"

usage() {
  cat <<'EOF'
Usage: collect_training_prefixes.sh [collect|status]

Environment:
  BUDGETS=all|i100,i1k|100,1000   Select exact retired-instruction budgets.
  RUN_DIR=PATH                    Override ignored output directory.
  BUILD=0                         Reuse an existing collector binary.
  FORCE=1                         Rerun selected points after archiving old files.
  DRY_RUN=1                       Print commands without executing ChampSim.
  CHAMP_DIR=PATH TRACE_FILE=PATH BIN=PATH

The collector uses --warmup_instructions=0 and
--simulation_instructions=N for every training prefix. Each stream is shared
by h8 and h16. Existing valid artifacts are never overwritten by default.
EOF
}

[[ "$COMMAND" != "-h" && "$COMMAND" != "--help" ]] || { usage; exit 0; }
[[ "$COMMAND" == "collect" || "$COMMAND" == "status" ]] || { usage >&2; exit 2; }

mkdir -p "$EVENT_DIR" "$PREFIX_DIR" "$EVAL_DIR" "$LOG_DIR"

POINT_TEXT="$(python3 - "$CONFIG" "$BUDGETS" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
requested = [item.strip() for item in sys.argv[2].split(",") if item.strip()]
points = config["instruction_budgets"]
by_tag = {item["tag"]: item for item in points}
by_value = {str(item["instructions"]): item for item in points}
if requested == ["all"]:
    chosen = points
else:
    chosen, seen = [], set()
    for token in requested:
        item = by_tag.get(token) or by_value.get(token)
        if item is None:
            raise SystemExit("unknown budget/tag: {}".format(token))
        if item["tag"] not in seen:
            chosen.append(item)
            seen.add(item["tag"])
for item in chosen:
    print("{} {}".format(item["tag"], item["instructions"]))
PY
)"
mapfile -t POINTS <<< "$POINT_TEXT"

archive_existing() {
  local path="$1"
  [[ ! -e "$path" ]] && return
  local backup="$RUN_DIR/replaced/$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$backup"
  mv "$path" "$backup/"
  echo "[archived] $path -> $backup/"
}

write_header_only_log() {
  python3 - "$1" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
path.write_text(
    "event,event_id,cpu,cycle,cache,op,type,ip,addr,line,hit,"
    "was_prefetch,late,accepted,duplicate,base_addr,pf_addr,pf_line,"
    "fill_level,pq_occ,pq_size,mshr_occ,mshr_size\n"
)
PY
}

ensure_collector() {
  if [[ "$DRY_RUN" == 1 ]]; then
    return 0
  fi
  [[ -s "$TRACE_FILE" ]] || { echo "[error] missing trace: $TRACE_FILE" >&2; exit 3; }
  if [[ "$BUILD" == 1 ]]; then
    CHAMP_DIR="$CHAMP_DIR" bash "$OFFLINE/linux/patch_demand_logger.sh"
    CHAMP_DIR="$CHAMP_DIR" OUT="$BIN" bash "$OFFLINE/linux/build_keyed_replayer.sh"
  fi
  [[ -x "$BIN" ]] || { echo "[error] missing collector binary: $BIN" >&2; exit 3; }
}

collect_point() {
  local tag="$1" budget="$2"
  local raw="$EVENT_DIR/602.gcc_s-734B.$tag.events.csv"
  local gz="$raw.gz"
  local log="$LOG_DIR/collect.$tag.log"
  local point_dir="$PREFIX_DIR/$tag"
  local stream="$point_dir/602.gcc_s-734B.$tag.train_stream.csv.gz"
  local manifest="$point_dir/training_manifest.json"
  local command="$BIN --l2c_prefetcher_types=none --warmup_instructions=0 --simulation_instructions=$budget -traces $TRACE_FILE"

  if [[ "$FORCE" != 1 && -s "$manifest" && -s "$stream" ]] && gzip -t "$stream"; then
    echo "[skip valid] $tag ($budget instructions)"
    return
  fi
  if [[ "$FORCE" != 1 && ( -e "$manifest" || -e "$stream" || -e "$gz" ) ]]; then
    echo "[error] incomplete existing point $tag; inspect it or use FORCE=1" >&2
    exit 4
  fi
  if [[ "$FORCE" == 1 ]]; then
    archive_existing "$point_dir"
    archive_existing "$gz"
    archive_existing "$log"
  fi
  mkdir -p "$point_dir"
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "[dry-run] DEMAND_EVENT_LOG=$raw $command > $log 2>&1"
    return
  fi
  DEMAND_EVENT_LOG="$raw" "$BIN" --l2c_prefetcher_types=none --warmup_instructions=0 --simulation_instructions="$budget" -traces "$TRACE_FILE" > "$log" 2>&1
  [[ -e "$raw" ]] || write_header_only_log "$raw"
  gzip -n "$raw"
  python3 "$EXP/python/build_training_manifest.py" --events "$gz" --stream "$stream" --manifest "$manifest" --instruction-budget "$budget" --budget-tag "$tag" --collection-command "$command"
}

collect_evaluation_once() {
  local raw="$EVAL_DIR/602.gcc_s-734B.evaluation.events.csv"
  local gz="$raw.gz"
  local stream="$EVAL_DIR/602.gcc_s-734B.eval_stream.csv.gz"
  local log="$LOG_DIR/collect.evaluation.log"
  if [[ -s "$stream" ]] && gzip -t "$stream"; then
    echo "[skip valid] fixed 25M warmup + 25M evaluation stream"
    return
  fi
  if [[ -e "$stream" || -e "$gz" ]]; then
    echo "[error] incomplete evaluation artifacts; inspect before rerunning" >&2
    exit 4
  fi
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "[dry-run] collect fixed evaluation stream (25M warmup + 25M measured)"
    return
  fi
  DEMAND_EVENT_LOG="$raw" "$BIN" --l2c_prefetcher_types=none --warmup_instructions=25000000 --simulation_instructions=25000000 -traces "$TRACE_FILE" > "$log" 2>&1
  [[ -s "$raw" ]] || { echo "[error] evaluation produced no callbacks" >&2; exit 5; }
  gzip -n "$raw"
  python3 "$OFFLINE/python/normalize_events.py" --events "$gz" --out "$stream"
  sha256sum "$gz" "$stream" > "$EVAL_DIR/SHA256SUMS"
}

show_status() {
  python3 "$EXP/python/summarize_training_prefixes.py" --prefix-root "$PREFIX_DIR" --out-dir "$RUN_DIR"
}

if [[ "$COMMAND" == "status" ]]; then
  show_status
  exit 0
fi

ensure_collector
for point in "${POINTS[@]}"; do
  read -r tag budget <<< "$point"
  collect_point "$tag" "$budget"
done
collect_evaluation_once
if [[ "$DRY_RUN" == 0 ]]; then
  show_status
fi
