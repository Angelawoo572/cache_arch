#!/usr/bin/env bash
# Run every current-run v3.9 candidate listed in a replay plan.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PLAN="${PLAN:-$ROOT/formal_NN_training/artifacts/v3_9/v3_9_replay_plan.csv}"
RUN_TAG="${RUN_TAG:-v3_9_replay_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT/formal_NN_training/results/v3_9_replay/$RUN_TAG}"
WARMUP="${WARMUP:-25000000}"
SIM="${SIM:-25000000}"
MAX_JOBS="${MAX_JOBS:-2}"
FORCE="${FORCE:-0}"

BIN="${BIN:-$ROOT/external/ChampSim/bin/champsim.standalone_nn_replayer}"
PREP="$ROOT/formal_NN_training/scripts/07_prepare_keyed_replay_input.py"
PARSER="$ROOT/formal_NN_training/scripts/20_parse_v3_9_replay_plan.py"
ORACLE_DIR="${ORACLE_DIR:-$ROOT/formal_NN_training/results/standalone_nn_data/oracle}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-$ROOT/formal_NN_training/results/prefetcher_baselines/summary.csv}"

[[ -s "$PLAN" ]] || { echo "[error] missing plan: $PLAN" >&2; exit 2; }
[[ -x "$BIN" && -f "$PREP" && -f "$PARSER" && -f "$BASELINE_SUMMARY" ]] || { echo "[error] missing binary/script/baseline" >&2; exit 2; }
[[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "[error] invalid MAX_JOBS" >&2; exit 2; }

LOG_DIR="$OUT_DIR/logs"
REPLAY_DIR="$OUT_DIR/replay_inputs"
mkdir -p "$LOG_DIR" "$REPLAY_DIR"

PLAN_ABS="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$PLAN")"
CANDIDATES="$OUT_DIR/candidates.tsv"

python3 -c '
import csv, sys
from pathlib import Path
plan = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
seen = set()
with plan.open(newline="") as f:
    r = csv.DictReader(f)
    req = {"tag", "trace", "source_rel"}
    miss = req - set(r.fieldnames or [])
    if miss: raise SystemExit("plan missing " + str(sorted(miss)))
    for n, row in enumerate(r, 2):
        tag, trace, rel = [(row.get(k) or "").strip() for k in ("tag", "trace", "source_rel")]
        if not tag or not trace or not rel: raise SystemExit("empty plan field at row {}".format(n))
        if tag in seen: raise SystemExit("duplicate tag " + tag)
        seen.add(tag)
        p = Path(rel)
        options = [p] if p.is_absolute() else [plan.parent / p, root / p]
        source = next((x.resolve() for x in options if x.is_file() and x.stat().st_size), None)
        if source is None: raise SystemExit("missing list for {}: {}".format(tag, rel))
        print("{}\t{}\t{}".format(tag, trace, source))
' "$PLAN_ABS" "$ROOT" > "$CANDIDATES"

[[ -s "$CANDIDATES" ]] || { echo "[error] plan contains no candidates" >&2; exit 2; }
cut -f2 "$CANDIDATES" | LC_ALL=C sort -u > "$OUT_DIR/traces.txt"
cp -f "$PLAN_ABS" "$OUT_DIR/v3_9_replay_plan.csv"

run_no_pref() {
  local trace="$1" trace_file="$ROOT/traces/$1.champsimtrace.xz" log="$LOG_DIR/$1.same_binary_no_pref.log"
  [[ -s "$trace_file" ]] || return 1
  if [[ "$FORCE" != 1 && -s "$log" ]]; then return 0; fi
  env -u PFETCH_LIST_PATH "$BIN" --l2c_prefetcher_types=none --warmup_instructions="$WARMUP" --simulation_instructions="$SIM" -traces "$trace_file" > "$log" 2>&1
  grep -Fq Core_0_IPC "$log"
}

while IFS= read -r trace; do
  echo "[same-binary no-pref] $trace"
  run_no_pref "$trace"
done < "$OUT_DIR/traces.txt"

run_one() {
  local tag="$1" trace="$2" rich="$3"
  local trace_file="$ROOT/traces/$trace.champsimtrace.xz"
  local oracle="$ORACLE_DIR/$trace.oracle.csv.gz"
  local keyed="$REPLAY_DIR/$tag.pc_line_occ.csv"
  local log="$LOG_DIR/$tag.list_replayer.log"
  [[ -s "$rich" && -s "$oracle" && -s "$trace_file" ]] || return 1
  if [[ "$FORCE" != 1 && -s "$log" && -s "$keyed" ]]; then return 0; fi
  python3 "$PREP" --rich-list "$rich" --oracle "$oracle" --out "$keyed" > "$LOG_DIR/$tag.prepare.log" 2>&1
  PFETCH_LIST_PATH="$keyed" "$BIN" --l2c_prefetcher_types=list_replayer --warmup_instructions="$WARMUP" --simulation_instructions="$SIM" -traces "$trace_file" > "$log" 2>&1
  grep -Fq 'adding L2C_PREFETCHER: list_replayer' "$log"
  grep -Fq 'key=pc_line_occ' "$log"
  grep -Fq Core_0_IPC "$log"
}

running=0
status=0
while IFS=$'\t' read -r tag trace rich; do
  echo "[queue] $tag"
  run_one "$tag" "$trace" "$rich" &
  running=$((running + 1))
  if (( running >= MAX_JOBS )); then
    wait -n || status=1
    running=$((running - 1))
  fi
done < "$CANDIDATES"
while (( running > 0 )); do
  wait -n || status=1
  running=$((running - 1))
done
(( status == 0 )) || exit "$status"

python3 "$PARSER" --plan "$PLAN_ABS" --log-root "$LOG_DIR" --replay-input-root "$REPLAY_DIR" --same-binary-log-root "$LOG_DIR" --baseline-summary "$BASELINE_SUMMARY" --out-dir "$OUT_DIR" --no-pref-ipc-tolerance "${NO_PREF_IPC_TOLERANCE:-0.002}"
echo "[done] $OUT_DIR/v3_9_replay_results.csv"
