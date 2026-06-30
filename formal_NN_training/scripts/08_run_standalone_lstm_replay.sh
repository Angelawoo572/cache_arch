#!/usr/bin/env bash
# Replay standalone NN exports with PC-line-occurrence keys.
#
# Legacy mode replays one conventional per-trace artifact name.  Plan mode
# (REPLAY_PLAN=...) replays every current-run candidate listed by the v3.9
# notebook and keeps logs/inputs separate by candidate tag.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TRACES="${TRACES:-602.gcc_s-734B 619.lbm_s-4268B}"
WARMUP="${WARMUP:-25000000}"
SIM="${SIM:-25000000}"
MAX_JOBS="${MAX_JOBS:-2}"
CHUNK_LEN="${CHUNK_LEN:-1024}"
DEDUP_CAPACITY="${DEDUP_CAPACITY:-256}"
EXPORT_SUFFIX="${EXPORT_SUFFIX:-pure_balanced_lru${DEDUP_CAPACITY}}"
ART_DIR="${ART_DIR:-formal_NN_training/artifacts/standalone_multihorizon_lstm_v3_2}"
RUN_TAG="${RUN_TAG:-standalone_lstm_cl${CHUNK_LEN}_${EXPORT_SUFFIX}}"
OUT_DIR="${OUT_DIR:-formal_NN_training/results/standalone_lstm_replay/${RUN_TAG}}"
LOG_DIR="$OUT_DIR/logs"
REPLAY_DIR="$OUT_DIR/replay_inputs"
RUN_SAME_BINARY_NO_PREF="${RUN_SAME_BINARY_NO_PREF:-1}"
FORCE="${FORCE:-0}"

# Optional v3.9 plan mode.  The plan must contain tag,trace,source_rel.  A
# source_rel value is resolved against PLAN_ROOT, which defaults to the plan's
# directory.  No historical artifact is inferred in plan mode.
REPLAY_PLAN="${REPLAY_PLAN:-}"
PLAN_ROOT="${PLAN_ROOT:-}"
PLAN_ENTRIES="$OUT_DIR/replay_plan_entries.tsv"

BIN="${BIN:-$ROOT/external/ChampSim/bin/champsim.standalone_nn_replayer}"
PREP="$ROOT/formal_NN_training/scripts/07_prepare_keyed_replay_input.py"
PARSER="$ROOT/formal_NN_training/scripts/09_parse_standalone_lstm_replay.py"
ORACLE_DIR="${ORACLE_DIR:-formal_NN_training/results/standalone_nn_data/oracle}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-formal_NN_training/results/prefetcher_baselines/summary.csv}"

mkdir -p "$LOG_DIR" "$REPLAY_DIR"
[[ -x "$BIN" ]] || { echo "[error] run scripts/06_install_keyed_listreplayer.sh first" >&2; exit 2; }
[[ -f "$PREP" && -f "$PARSER" && -f "$BASELINE_SUMMARY" ]] || { echo "[error] missing script or baseline table" >&2; exit 2; }
[[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "[error] MAX_JOBS must be a positive integer" >&2; exit 2; }

plan_entries() {
  local plan="$1" root="$2" out="$3"
  python3 - "$plan" "$root" "$out" <<'PY'
from __future__ import print_function
import csv
import re
import sys
from pathlib import Path

plan = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
out = Path(sys.argv[3])
if not plan.is_file():
    raise SystemExit("missing replay plan: {}".format(plan))
if not root.is_dir():
    raise SystemExit("missing plan root: {}".format(root))
safe = re.compile(r"^[A-Za-z0-9_.-]+$")
required = {"tag", "trace", "source_rel"}
rows, seen = [], set()
with plan.open(newline="") as handle:
    reader = csv.DictReader(handle)
    fields = set(reader.fieldnames or [])
    missing = sorted(required - fields)
    if missing:
        raise SystemExit("replay plan missing columns: {}".format(missing))
    for line_no, row in enumerate(reader, start=2):
        tag = (row.get("tag") or "").strip()
        trace = (row.get("trace") or "").strip()
        source_rel = (row.get("source_rel") or "").strip()
        if not tag or not trace or not source_rel:
            raise SystemExit("blank tag/trace/source_rel at replay plan row {}".format(line_no))
        if not safe.match(tag) or not safe.match(trace):
            raise SystemExit("unsafe tag or trace at replay plan row {}".format(line_no))
        if tag in seen:
            raise SystemExit("duplicate plan tag: {}".format(tag))
        seen.add(tag)
        source = Path(source_rel)
        source = source if source.is_absolute() else root / source
        source = source.resolve()
        if not source.is_file() or source.stat().st_size == 0:
            raise SystemExit("missing/nonempty rich list for {}: {}".format(tag, source))
        if source.suffix.lower() != ".csv":
            raise SystemExit("rich list for {} is not a CSV: {}".format(tag, source))
        rows.append((tag, trace, str(source)))
if not rows:
    raise SystemExit("replay plan has no candidates")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as handle:
    for tag, trace, source in rows:
        handle.write("{}\t{}\t{}\n".format(tag, trace, source))
print("[plan] {} candidates from {}".format(len(rows), plan))
PY
}

run_same_binary_no_pref() {
  local trace="$1"
  local same_bin_log="$LOG_DIR/${trace}.same_binary_no_pref.log"
  local trace_file="traces/${trace}.champsimtrace.xz"
  [[ -s "$trace_file" ]] || { echo "[error] missing trace for $trace: $trace_file" >&2; return 1; }
  [[ "$RUN_SAME_BINARY_NO_PREF" == "1" ]] || return 0
  if [[ "$FORCE" != "1" && -s "$same_bin_log" ]] && grep -Fq 'Core_0_IPC' "$same_bin_log"; then
    echo "[skip same-binary no_pref] $trace"
    return 0
  fi
  echo "[run same-binary no_pref] $trace"
  env -u PFETCH_LIST_PATH "$BIN" --l2c_prefetcher_types=none \
    --warmup_instructions="$WARMUP" --simulation_instructions="$SIM" \
    -traces "$trace_file" > "$same_bin_log" 2>&1
  grep -Fq 'Core_0_IPC' "$same_bin_log"
}

run_one_legacy() {
  local trace="$1"
  local rich="$ART_DIR/prefetch_list_${trace}_cl${CHUNK_LEN}_${EXPORT_SUFFIX}.csv"
  local oracle="$ORACLE_DIR/${trace}.oracle.csv.gz"
  local keyed="$REPLAY_DIR/${trace}.pc_line_occ.csv"
  local log="$LOG_DIR/${trace}.standalone_lstm.log"
  local trace_file="traces/${trace}.champsimtrace.xz"
  [[ -s "$rich" && -s "$oracle" && -s "$trace_file" ]] || { echo "[error] missing input for $trace" >&2; return 1; }

  if [[ "$FORCE" != "1" && -s "$log" && -s "$keyed" ]]; then
    echo "[skip replay] $trace"
    return 0
  fi
  python3 "$PREP" --rich-list "$rich" --oracle "$oracle" --out "$keyed" > "$LOG_DIR/${trace}.prepare.log" 2>&1
  PFETCH_LIST_PATH="$keyed" "$BIN" --l2c_prefetcher_types=list_replayer \
    --warmup_instructions="$WARMUP" --simulation_instructions="$SIM" \
    -traces "$trace_file" > "$log" 2>&1
  grep -Fq 'adding L2C_PREFETCHER: list_replayer' "$log"
  grep -Fq 'PC-line-occ triggers' "$log"
  grep -Fq 'key=pc_line_occ' "$log"
  echo "[done] $trace"
}

run_one_plan() {
  local tag="$1" trace="$2" rich="$3"
  local oracle="$ORACLE_DIR/${trace}.oracle.csv.gz"
  local keyed="$REPLAY_DIR/${tag}.pc_line_occ.csv"
  local log="$LOG_DIR/${tag}.standalone_lstm.log"
  local trace_file="traces/${trace}.champsimtrace.xz"
  [[ -s "$rich" && -s "$oracle" && -s "$trace_file" ]] || {
    echo "[error] missing plan input tag=$tag trace=$trace" >&2; return 1; }

  if [[ "$FORCE" != "1" && -s "$log" && -s "$keyed" ]] && grep -Fq 'key=pc_line_occ' "$log"; then
    echo "[skip replay] $tag"
    return 0
  fi
  python3 "$PREP" --rich-list "$rich" --oracle "$oracle" --out "$keyed" > "$LOG_DIR/${tag}.prepare.log" 2>&1
  PFETCH_LIST_PATH="$keyed" "$BIN" --l2c_prefetcher_types=list_replayer \
    --warmup_instructions="$WARMUP" --simulation_instructions="$SIM" \
    -traces "$trace_file" > "$log" 2>&1
  grep -Fq 'adding L2C_PREFETCHER: list_replayer' "$log"
  grep -Fq 'PC-line-occ triggers' "$log"
  grep -Fq 'key=pc_line_occ' "$log"
  echo "[done] $tag ($trace)"
}

cat > "$OUT_DIR/RUN_INFO.txt" <<EOF
RUN_KIND=standalone_keyed_replay
REPLAY_PLAN=$REPLAY_PLAN
PLAN_ROOT=$PLAN_ROOT
ART_DIR=$ART_DIR
TRACES=$TRACES
WARMUP=$WARMUP
SIM=$SIM
CHUNK_LEN=$CHUNK_LEN
DEDUP_CAPACITY=$DEDUP_CAPACITY
EXPORT_SUFFIX=$EXPORT_SUFFIX
BIN=$BIN
ORACLE_DIR=$ORACLE_DIR
BASELINE_SUMMARY=$BASELINE_SUMMARY
RUN_SAME_BINARY_NO_PREF=$RUN_SAME_BINARY_NO_PREF
MAX_JOBS=$MAX_JOBS
FORCE=$FORCE
EOF

status=0
if [[ -n "$REPLAY_PLAN" ]]; then
  [[ -z "$PLAN_ROOT" ]] && PLAN_ROOT="$(cd "$(dirname "$REPLAY_PLAN")" && pwd)"
  plan_entries "$REPLAY_PLAN" "$PLAN_ROOT" "$PLAN_ENTRIES"
  cp -f "$REPLAY_PLAN" "$OUT_DIR/v3_9_replay_plan.csv"

  # Controls are shared per trace and run once before list candidates, avoiding
  # races between two candidates of the same trace.
  running=0
  while IFS= read -r trace; do
    run_same_binary_no_pref "$trace" &
    running=$((running + 1))
    if (( running >= MAX_JOBS )); then
      wait -n || status=1
      running=$((running - 1))
    fi
  done < <(awk -F '\t' '{print $2}' "$PLAN_ENTRIES" | sort -u)
  while (( running > 0 )); do
    wait -n || status=1
    running=$((running - 1))
  done
  (( status == 0 )) || exit "$status"

  running=0
  while IFS=$'\t' read -r tag trace rich; do
    run_one_plan "$tag" "$trace" "$rich" &
    running=$((running + 1))
    if (( running >= MAX_JOBS )); then
      wait -n || status=1
      running=$((running - 1))
    fi
  done < "$PLAN_ENTRIES"
  while (( running > 0 )); do
    wait -n || status=1
    running=$((running - 1))
  done
  (( status == 0 )) || exit "$status"

  python3 "$PARSER" \
    --log-root "$LOG_DIR" \
    --replay-input-root "$REPLAY_DIR" \
    --out "$OUT_DIR/summary.csv" \
    --traces "$(awk -F '\t' '{print $2}' "$PLAN_ENTRIES" | sort -u | xargs)" \
    --baseline-summary "$BASELINE_SUMMARY" \
    --replay-plan "$REPLAY_PLAN" \
    --plan-root "$PLAN_ROOT" \
    --winner-out "$OUT_DIR/v3_9_nn_winners.csv" \
    --same-binary-log-root "$LOG_DIR"
else
  running=0
  for trace in $TRACES; do
    run_same_binary_no_pref "$trace" &
    running=$((running + 1))
    if (( running >= MAX_JOBS )); then
      wait -n || status=1
      running=$((running - 1))
    fi
  done
  while (( running > 0 )); do
    wait -n || status=1
    running=$((running - 1))
  done
  (( status == 0 )) || exit "$status"

  running=0
  for trace in $TRACES; do
    run_one_legacy "$trace" &
    running=$((running+1))
    if (( running >= MAX_JOBS )); then
      wait -n || status=1
      running=$((running-1))
    fi
  done
  while (( running > 0 )); do
    wait -n || status=1
    running=$((running-1))
  done
  (( status == 0 )) || exit "$status"

  parse_args=(--log-root "$LOG_DIR" --replay-input-root "$REPLAY_DIR" --out "$OUT_DIR/summary.csv" --traces "$TRACES" --baseline-summary "$BASELINE_SUMMARY")
  if [[ "$RUN_SAME_BINARY_NO_PREF" == "1" ]]; then
    parse_args+=(--same-binary-log-root "$LOG_DIR")
  fi
  python3 "$PARSER" "${parse_args[@]}"
fi
echo "[done] $OUT_DIR/summary.csv"
