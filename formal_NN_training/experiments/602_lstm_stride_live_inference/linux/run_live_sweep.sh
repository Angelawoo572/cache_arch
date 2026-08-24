#!/usr/bin/env bash
# Run explicit or approved selected frozen-weight live points.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/formal_NN_training/experiments/602_lstm_stride_live_inference"
RUN_DIR="${RUN_DIR:-$EXP/runs/602_gcc_stride_prefix_seed7}"
SELECTION="${SELECTION:-$EXP/config/live_selection.json}"
BIN="${BIN:-$RUN_DIR/bin/champsim.602_stride_lstm_live}"
TRACE_FILE="${TRACE_FILE:-$ROOT/traces/602.gcc_s-734B.champsimtrace.xz}"
HIDDEN_SIZES="${HIDDEN_SIZES:-}"
BUDGETS="${BUDGETS:-}"
SEEDS="${SEEDS:-7}"
LIVE_ALL_VALID="${LIVE_ALL_VALID:-0}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
WARMUP_INSTRUCTIONS="${WARMUP_INSTRUCTIONS:-25000000}"
SIMULATION_INSTRUCTIONS="${SIMULATION_INSTRUCTIONS:-25000000}"
STATE_MODE="${STATE_MODE:-parity}"
RUN_MODE="${RUN_MODE:-live}"
COMMAND="${1:-run}"

usage() {
  cat <<'EOF'
Usage: run_live_sweep.sh [run|status]

Default: run only points in an approved live_selection.json.
Selectors:
  HIDDEN_SIZES=16 BUDGETS=i20m SEEDS=7
  HIDDEN_SIZES=8,16 BUDGETS=i1m,i20m
  LIVE_ALL_VALID=1       Expensive: all exported valid points.
  LIVE_ALL_VALID=1 HIDDEN_SIZES=8
                         All exported valid h8 points. A disjoint h16 process
                         may run concurrently.

Controls:
  WARMUP_INSTRUCTIONS=25000000 SIMULATION_INSTRUCTIONS=25000000
  STATE_MODE=parity|realistic_state_warmup
  RUN_MODE=smoke|live FORCE=1 DRY_RUN=1

Existing valid logs are never overwritten by default.
EOF
}

[[ "$COMMAND" != "-h" && "$COMMAND" != "--help" ]] || { usage; exit 0; }
[[ "$COMMAND" == "run" || "$COMMAND" == "status" ]] || { usage >&2; exit 2; }

select_points() {
  python3 - "$RUN_DIR" "$SELECTION" "$HIDDEN_SIZES" "$BUDGETS" "$SEEDS" "$LIVE_ALL_VALID" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
selection_path = Path(sys.argv[2])
hidden_csv, budget_csv, seed_csv, all_valid = sys.argv[3:]
seeds = [int(item) for item in seed_csv.split(",") if item]
if all_valid == "1":
    hidden_filter = (
        {int(item) for item in hidden_csv.split(",") if item}
        if hidden_csv else None
    )
    budget_filter = (
        {item for item in budget_csv.split(",") if item}
        if budget_csv else None
    )
    chosen = []
    for path in run_dir.glob("points/h*/*/seed*/point_metadata.json"):
        metadata = json.loads(path.read_text())
        model = path.parent / "export/model.bin"
        if (
            model.is_file()
            and metadata.get("status") not in {
                "training_failed", "offline_replay_failed", "export_failed",
            }
            and (
                hidden_filter is None
                or metadata["hidden_size"] in hidden_filter
            )
            and (
                budget_filter is None
                or metadata["budget_tag"] in budget_filter
            )
        ):
            chosen.append((
                metadata["hidden_size"], metadata["budget_tag"],
                metadata["seed"],
            ))
elif hidden_csv or budget_csv:
    if not hidden_csv or not budget_csv:
        raise SystemExit("explicit selection requires HIDDEN_SIZES and BUDGETS")
    hidden = [int(item) for item in hidden_csv.split(",") if item]
    budgets = [item for item in budget_csv.split(",") if item]
    chosen = [(h, b, seed) for h in hidden for b in budgets for seed in seeds]
else:
    selection = json.loads(selection_path.read_text())
    if not selection.get("approved"):
        raise SystemExit(
            "live_selection.json is not approved; inspect/edit it or use explicit selectors"
        )
    chosen = [
        (item["hidden_size"], item["budget_tag"], item.get("seed", 7))
        for item in selection.get("selected_points", [])
    ]
for hidden, budget, seed in sorted(set(chosen)):
    if hidden not in (8, 16):
        raise SystemExit("only h8/h16 are in scope")
    print("{} {} {}".format(hidden, budget, seed))
PY
}

mapfile -t POINTS < <(select_points)
if [[ "$COMMAND" == "status" ]]; then
  find "$RUN_DIR/points" -path '*/live/*/run.log' -type f -print 2>/dev/null | sort || true
  exit 0
fi
[[ -x "$BIN" ]] || { echo "[error] missing live binary: $BIN" >&2; exit 3; }
[[ -s "$TRACE_FILE" ]] || { echo "[error] missing trace: $TRACE_FILE" >&2; exit 3; }
[[ "${#POINTS[@]}" -gt 0 ]] || { echo "[error] no live points selected" >&2; exit 3; }

for point in "${POINTS[@]}"; do
  read -r hidden budget seed <<< "$point"
  point_dir="$RUN_DIR/points/h$hidden/$budget/seed$seed"
  model="$point_dir/export/model.bin"
  metadata="$point_dir/export/model_metadata.json"
  live_dir="$point_dir/live/$STATE_MODE/$RUN_MODE"
  log="$live_dir/run.log"
  open_log="$live_dir/openat.log"
  [[ -s "$model" && -s "$metadata" ]] || {
    echo "[error] missing export for h$hidden $budget seed$seed" >&2
    exit 4
  }
  python3 - "$model" "$metadata" <<'PY'
import json
import sys
from pathlib import Path
model, metadata_path = map(Path, sys.argv[1:])
metadata = json.loads(metadata_path.read_text())
if not model.is_file() or not metadata.get("weights_frozen"):
    raise SystemExit("invalid frozen export: {}".format(model))
if metadata.get("format_version") != 1:
    raise SystemExit("unsupported model format: {}".format(model))
PY
  if [[ -s "$log" && "$FORCE" != 1 ]] &&
     grep -Fq 'stride_lstm_live_measured_callbacks ' "$log"; then
    echo "[skip valid] h$hidden $budget seed$seed $STATE_MODE $RUN_MODE"
    continue
  fi
  if [[ -e "$live_dir" && "$FORCE" != 1 ]]; then
    echo "[error] incomplete live run; inspect or set FORCE=1: $live_dir" >&2
    exit 5
  fi
  if [[ -e "$live_dir" ]]; then
    archive="$RUN_DIR/replaced/$(date -u +%Y%m%dT%H%M%SZ)/points/h$hidden/$budget/seed$seed/live"
    mkdir -p "$(dirname "$archive")"
    mv "$live_dir" "$archive"
  fi
  command=("$BIN" "--l2c_prefetcher_types=stride_lstm_live" "--warmup_instructions=$WARMUP_INSTRUCTIONS" "--simulation_instructions=$SIMULATION_INSTRUCTIONS" "-traces" "$TRACE_FILE")
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "[dry-run] STRIDE_LSTM_MODEL_BIN=$model STRIDE_LSTM_STATE_MODE=$STATE_MODE ${command[*]}"
    continue
  fi
  mkdir -p "$live_dir"
  python3 - "$live_dir/run_identity.json" "$hidden" "$budget" "$seed" "$STATE_MODE" "$WARMUP_INSTRUCTIONS" "$SIMULATION_INSTRUCTIONS" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
payload = {
    "trace": "602.gcc_s-734B",
    "hidden_size": int(sys.argv[2]),
    "budget_tag": sys.argv[3],
    "seed": int(sys.argv[4]),
    "mode": "live",
    "state_mode": sys.argv[5],
    "warmup_instructions": int(sys.argv[6]),
    "simulation_instructions": int(sys.argv[7]),
    "weights_frozen": True,
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  set +e
  if [[ "${TRACE_FILE_OPENS:-0}" == 1 ]] && command -v strace >/dev/null 2>&1; then
    STRIDE_LSTM_MODEL_BIN="$model" STRIDE_LSTM_STATE_MODE="$STATE_MODE" strace -f -e trace=openat -o "$open_log" "${command[@]}" > "$log" 2>&1
  else
    STRIDE_LSTM_MODEL_BIN="$model" STRIDE_LSTM_STATE_MODE="$STATE_MODE" "${command[@]}" > "$log" 2>&1
  fi
  run_status=$?
  set -e
  if [[ "$run_status" != 0 ]]; then
    python3 - "$point_dir/point_metadata.json" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
data = json.loads(path.read_text())
data.setdefault("status_history", []).append("live_run_failed")
data["status"] = "live_run_failed"
data["failure_reason"] = "ChampSim live command failed; inspect run.log"
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
    exit "$run_status"
  fi
  grep -Fq 'adding L2C_PREFETCHER: stride_lstm_live' "$log"
  grep -Fq 'stride_lstm_live_weights frozen' "$log"
  grep -Fq 'stride_lstm_live_measured_callbacks ' "$log"
  if [[ -s "$open_log" ]] && grep -Fq 'replay.csv' "$open_log"; then
    echo "[error] forbidden replay file access observed" >&2
    exit 6
  fi
  python3 - "$point_dir/point_metadata.json" "$RUN_MODE" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
status = (
    "live_smoke_complete" if sys.argv[2] == "smoke"
    else "live_run_complete"
)
data = json.loads(path.read_text())
history = data.setdefault("status_history", [])
if status not in history:
    history.append(status)
data["status"] = status
data["failure_reason"] = None
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
  echo "[complete] h$hidden $budget seed$seed $STATE_MODE $RUN_MODE"
done
