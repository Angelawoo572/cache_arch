#!/usr/bin/env bash
# Reuse the existing keyed replay path for every valid offline point.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/formal_NN_training/experiments/602_lstm_stride_live_inference"
RUN_DIR="${RUN_DIR:-$EXP/runs/602_gcc_stride_prefix_seed7}"
BIN="${BIN:-$ROOT/external/ChampSim/bin/champsim.602_offline_replay}"
TRACE_FILE="${TRACE_FILE:-$ROOT/traces/602.gcc_s-734B.champsimtrace.xz}"
HIDDEN_SIZES="${HIDDEN_SIZES:-8,16}"
BUDGETS="${BUDGETS:-all}"
SEEDS="${SEEDS:-7}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
COMMAND="${1:-run}"
WARMUP_INSTRUCTIONS="${WARMUP_INSTRUCTIONS:-25000000}"
SIMULATION_INSTRUCTIONS="${SIMULATION_INSTRUCTIONS:-25000000}"

usage() {
  cat <<'EOF'
Usage: run_offline_sweep.sh [run|status]

Selectors:
  HIDDEN_SIZES=8|16|8,16
  BUDGETS=all|i100,i1m|100,1000000
  SEEDS=7

Controls:
  FORCE=1 DRY_RUN=1
  WARMUP_INSTRUCTIONS=25000000 SIMULATION_INSTRUCTIONS=25000000

The no-pref, conventional live Stride, and keyed offline Stride references are
run once and reused. Existing valid point logs are not overwritten.
EOF
}

[[ "$COMMAND" != "-h" && "$COMMAND" != "--help" ]] || { usage; exit 0; }
[[ "$COMMAND" == "run" || "$COMMAND" == "status" ]] || { usage >&2; exit 2; }

mapfile -t POINTS < <(python3 - "$RUN_DIR" "$HIDDEN_SIZES" "$BUDGETS" "$SEEDS" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
hidden = {int(x) for x in sys.argv[2].split(",") if x}
budgets = {x for x in sys.argv[3].split(",") if x}
seeds = {int(x) for x in sys.argv[4].split(",") if x}
for path in run_dir.glob("points/h*/*/seed*/point_metadata.json"):
    row = json.loads(path.read_text())
    if row.get("hidden_size") not in hidden or row.get("seed") not in seeds:
        continue
    if budgets != {"all"} and (
        row.get("budget_tag") not in budgets
        and str(row.get("instruction_budget")) not in budgets
    ):
        continue
    if row.get("status") in {
        "no_callbacks", "single_class_no_act", "single_class_no_silent",
        "insufficient_rows", "training_failed",
    }:
        continue
    replay = path.parent / "offline/offline_lstm.replay.csv"
    if replay.is_file():
        print("{} {} {}".format(
            row["hidden_size"], row["budget_tag"], row["seed"]
        ))
PY
)

if [[ "$COMMAND" == "status" ]]; then
  python3 "$EXP/python/aggregate_offline_results.py" --run-dir "$RUN_DIR"
  exit 0
fi
[[ "${#POINTS[@]}" -gt 0 ]] || { echo "[error] no valid offline points" >&2; exit 3; }
if [[ "$DRY_RUN" != 1 ]]; then
  [[ -x "$BIN" ]] || { echo "[error] missing keyed replay binary: $BIN" >&2; exit 3; }
  [[ -s "$TRACE_FILE" ]] || { echo "[error] missing trace: $TRACE_FILE" >&2; exit 3; }
fi

archive_file() {
  local path="$1"
  [[ -e "$path" ]] || return 0
  local destination="$RUN_DIR/replaced/$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$destination"
  mv "$path" "$destination/"
}

run_method() {
  local label="$1" prefetcher="$2" replay_list="$3" log="$4" raw="$5"
  if [[ -s "$log" && "$FORCE" != 1 ]] &&
     grep -Eq 'Core_0_IPC|Finished CPU 0 instructions:' "$log"; then
    echo "[skip valid] $label"
    return
  fi
  if [[ ( -e "$log" || -e "$raw.gz" ) && "$FORCE" != 1 ]]; then
    echo "[error] incomplete output for $label; inspect or set FORCE=1" >&2
    exit 4
  fi
  archive_file "$log"
  archive_file "$raw.gz"
  mkdir -p "$(dirname "$log")" "$(dirname "$raw")"
  command=("$BIN" "--l2c_prefetcher_types=$prefetcher" "--warmup_instructions=$WARMUP_INSTRUCTIONS" "--simulation_instructions=$SIMULATION_INSTRUCTIONS" "-traces" "$TRACE_FILE")
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "[dry-run] $label ${command[*]}"
    return
  fi
  if [[ -n "$replay_list" ]]; then
    DEMAND_EVENT_LOG="$raw" PFETCH_LIST_PATH="$replay_list" "${command[@]}" > "$log" 2>&1
  else
    DEMAND_EVENT_LOG="$raw" "${command[@]}" > "$log" 2>&1
  fi
  [[ -s "$raw" ]] || { echo "[error] no event log for $label" >&2; exit 5; }
  gzip -n "$raw"
}

reference_root="$RUN_DIR/references"
mkdir -p "$reference_root/logs" "$reference_root/events"
first_point="${POINTS[0]}"
read -r first_hidden first_budget first_seed <<< "$first_point"
normal_list="$RUN_DIR/points/h$first_hidden/$first_budget/seed$first_seed/offline/offline_stride.replay.csv"
[[ "$DRY_RUN" == 1 || -s "$normal_list" ]] || {
  echo "[error] normal offline Stride replay list absent: $normal_list" >&2
  exit 3
}
if [[ "$DRY_RUN" != 1 ]]; then
  python3 - "$RUN_DIR" "$normal_list" <<'PY'
import hashlib
import sys
from pathlib import Path
root = Path(sys.argv[1])
reference = Path(sys.argv[2])
expected = hashlib.sha256(reference.read_bytes()).hexdigest()
for path in root.glob("points/h*/*/seed*/offline/offline_stride.replay.csv"):
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise SystemExit(
            "fixed evaluation teacher list differs: {}".format(path)
        )
print("PASS: one fixed offline Stride reference list is reusable")
PY
fi
run_method no_pref none "" "$reference_root/logs/no_pref.log" "$reference_root/events/no_pref.events.csv"
run_method live_stride stride "" "$reference_root/logs/live_stride.log" "$reference_root/events/live_stride.events.csv"
run_method offline_stride list_replayer "$normal_list" "$reference_root/logs/offline_stride.log" "$reference_root/events/offline_stride.events.csv"

for point in "${POINTS[@]}"; do
  read -r hidden budget seed <<< "$point"
  point_dir="$RUN_DIR/points/h$hidden/$budget/seed$seed"
  replay="$point_dir/offline/offline_lstm.replay.csv"
  log="$point_dir/offline/replay.log"
  raw="$point_dir/offline/replay.events.csv"
  if run_method "h$hidden $budget seed$seed" list_replayer "$replay" "$log" "$raw"; then
    if [[ "$DRY_RUN" != 1 ]]; then
      python3 - "$point_dir/point_metadata.json" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
data = json.loads(path.read_text())
history = data.setdefault("status_history", [])
if "offline_replay_complete" not in history:
    history.append("offline_replay_complete")
data["status"] = "offline_replay_complete"
data["failure_reason"] = None
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
    fi
  else
    python3 - "$point_dir/point_metadata.json" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
data = json.loads(path.read_text())
data.setdefault("status_history", []).append("offline_replay_failed")
data["status"] = "offline_replay_failed"
data["failure_reason"] = "keyed ChampSim command failed; inspect replay.log"
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
  fi
done
if [[ "$DRY_RUN" != 1 ]]; then
  python3 "$EXP/python/aggregate_offline_results.py" --run-dir "$RUN_DIR"
fi
