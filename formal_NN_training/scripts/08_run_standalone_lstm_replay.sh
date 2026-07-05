#!/usr/bin/env bash
# Replay frozen standalone NN exports with PC-line-occurrence keys.
# Plan mode replays one keyed rich list per plan entry and emits a current-run
# candidate summary plus one winner per trace.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
TRACES="${TRACES:-602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B 620.omnetpp_s-874B 623.xalancbmk_s-700B}"
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
SKIP_BASELINE_REFERENCE="${SKIP_BASELINE_REFERENCE:-0}"
FORCE="${FORCE:-0}"
REPLAY_PLAN="${REPLAY_PLAN:-}"
PLAN_ROOT="${PLAN_ROOT:-}"
BASELINE_TOLERANCE="${BASELINE_TOLERANCE:-}"
BIN="${BIN:-$ROOT/external/ChampSim/bin/champsim.standalone_nn_replayer}"
ORACLE_DIR="${ORACLE_DIR:-formal_NN_training/results/standalone_nn_data/oracle}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-formal_NN_training/results/prefetcher_baselines/summary.csv}"
PREP="$ROOT/formal_NN_training/scripts/07_prepare_keyed_replay_input.py"
PARSER="$ROOT/formal_NN_training/scripts/09_parse_standalone_lstm_replay.py"
PLAN_RESOLVER="$ROOT/formal_NN_training/scripts/replay/resolve_replay_plan.py"
BASELINE_GUARD="$ROOT/formal_NN_training/scripts/replay/verify_same_binary_no_pref.py"

[[ "$RUN_SAME_BINARY_NO_PREF" == 0 || "$RUN_SAME_BINARY_NO_PREF" == 1 ]] || { echo "[error] RUN_SAME_BINARY_NO_PREF must be 0 or 1" >&2; exit 2; }
[[ "$SKIP_BASELINE_REFERENCE" == 0 || "$SKIP_BASELINE_REFERENCE" == 1 ]] || { echo "[error] SKIP_BASELINE_REFERENCE must be 0 or 1" >&2; exit 2; }
if [[ "$SKIP_BASELINE_REFERENCE" == 1 ]]; then
  BASELINE_REFERENCE_JSON=""
else
  BASELINE_REFERENCE_JSON="${BASELINE_REFERENCE_JSON:-$ROOT/formal_NN_training/_cfg/replay_same_binary_no_pref_reference_v4_0.json}"
fi

mkdir -p "$LOG_DIR" "$REPLAY_DIR"
[[ -x "$BIN" ]] || { echo "[error] run scripts/06_install_keyed_listreplayer.sh first" >&2; exit 2; }
[[ -f "$PREP" && -f "$PARSER" && -f "$PLAN_RESOLVER" && -f "$BASELINE_SUMMARY" ]] || { echo "[error] missing replay helper or baseline table" >&2; exit 2; }
[[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "[error] MAX_JOBS must be a positive integer" >&2; exit 2; }
if [[ -n "$BASELINE_REFERENCE_JSON" ]]; then
  [[ -f "$BASELINE_REFERENCE_JSON" && -f "$BASELINE_GUARD" ]] || { echo "[error] missing baseline reference or guard" >&2; exit 2; }
fi

completed_log() {
  [[ -s "$1" ]] && grep -Fq "$2" "$1"
}

completed_replay_log() {
  local log="$1"
  completed_log "$log" 'adding L2C_PREFETCHER: list_replayer' \
    && completed_log "$log" 'PC-line-occ triggers' \
    && completed_log "$log" 'key=pc_line_occ'
}

run_same_binary_no_pref() {
  local trace="$1"
  local trace_file="traces/${trace}.champsimtrace.xz"
  local log="$LOG_DIR/${trace}.same_binary_no_pref.log"
  [[ -s "$trace_file" ]] || { echo "[error] missing trace: $trace_file" >&2; return 1; }
  [[ "$RUN_SAME_BINARY_NO_PREF" == 1 ]] || return 0
  if [[ "$FORCE" != 1 ]] && completed_log "$log" Core_0_IPC; then
    echo "[skip same-binary no_pref] $trace"
    return 0
  fi
  echo "[run same-binary no_pref] $trace"
  if ! env -u PFETCH_LIST_PATH "$BIN" --l2c_prefetcher_types=none --warmup_instructions="$WARMUP" --simulation_instructions="$SIM" -traces "$trace_file" > "$log" 2>&1; then
    echo "[error] same-binary no_pref simulator failed: $trace" >&2
    return 1
  fi
  completed_log "$log" Core_0_IPC || { echo "[error] same-binary no_pref has no final IPC: $trace" >&2; return 1; }
}

verify_same_binary_no_pref() {
  local input="$1"
  [[ "$RUN_SAME_BINARY_NO_PREF" == 1 ]] || return 0
  [[ -n "$BASELINE_REFERENCE_JSON" ]] || { echo "[baseline guard skipped] SKIP_BASELINE_REFERENCE=1" >&2; return 0; }
  local args=(--reference "$BASELINE_REFERENCE_JSON" --log-root "$LOG_DIR" --out "$OUT_DIR/same_binary_no_pref_verification.json")
  [[ -z "$BASELINE_TOLERANCE" ]] || args+=(--tolerance "$BASELINE_TOLERANCE")
  local trace traces=()
  while IFS=$'\t' read -r trace _; do [[ -n "$trace" ]] && traces+=("$trace"); done < "$input"
  python3 "$BASELINE_GUARD" "${args[@]}" --traces "${traces[@]}"
}

run_candidate() {
  local tag="$1" trace="$2" rich="$3"
  local oracle="$ORACLE_DIR/${trace}.oracle.csv.gz"
  local keyed="$REPLAY_DIR/${tag}.pc_line_occ.csv"
  local log="$LOG_DIR/${tag}.standalone_lstm.log"
  local trace_file="traces/${trace}.champsimtrace.xz"
  [[ -s "$rich" && -s "$oracle" && -s "$trace_file" ]] || { echo "[error] missing replay input for $tag" >&2; return 1; }
  if [[ "$FORCE" != 1 && -s "$keyed" ]] && completed_replay_log "$log"; then
    echo "[skip replay] $tag ($trace)"
    return 0
  fi
  if ! python3 "$PREP" --rich-list "$rich" --oracle "$oracle" --out "$keyed" > "$LOG_DIR/${tag}.prepare.log" 2>&1; then
    echo "[error] replay-input preparation failed: $tag" >&2
    return 1
  fi
  echo "[replay] $tag ($trace)"
  if ! PFETCH_LIST_PATH="$keyed" "$BIN" --l2c_prefetcher_types=list_replayer --warmup_instructions="$WARMUP" --simulation_instructions="$SIM" -traces "$trace_file" > "$log" 2>&1; then
    echo "[error] simulator replay failed: $tag ($trace)" >&2
    return 1
  fi
  completed_replay_log "$log" || { echo "[error] incomplete replay log: $tag ($trace)" >&2; return 1; }
}

run_parallel() {
  local worker="$1" input="$2" status=0 running=0 a b c
  while IFS=$'\t' read -r a b c; do
    [[ -n "$a" ]] || continue
    if [[ -n "$c" ]]; then "$worker" "$a" "$b" "$c" &
    elif [[ -n "$b" ]]; then "$worker" "$a" "$b" &
    else "$worker" "$a" &
    fi
    running=$((running + 1))
    if (( running >= MAX_JOBS )); then wait -n || status=1; running=$((running - 1)); fi
  done < "$input"
  while (( running > 0 )); do wait -n || status=1; running=$((running - 1)); done
  return "$status"
}

if [[ -n "$REPLAY_PLAN" ]]; then
  [[ -f "$REPLAY_PLAN" ]] || { echo "[error] replay plan not found: $REPLAY_PLAN" >&2; exit 2; }
  [[ -n "$PLAN_ROOT" ]] || PLAN_ROOT="$(dirname "$REPLAY_PLAN")"
  PLAN_CSV="$REPLAY_PLAN"
  PLAN_ROOT_FOR_PARSE="$PLAN_ROOT"
else
  PLAN_CSV="$OUT_DIR/legacy_replay_plan.csv"
  PLAN_ROOT_FOR_PARSE="$ROOT"
  printf 'tag,trace,source_rel\n' > "$PLAN_CSV"
  for trace in $TRACES; do
    rich="$ART_DIR/prefetch_list_${trace}_cl${CHUNK_LEN}_${EXPORT_SUFFIX}.csv"
    printf 'legacy_%s,%s,%s\n' "$trace" "$trace" "$rich" >> "$PLAN_CSV"
  done
fi

RESOLVED_PLAN="$OUT_DIR/replay_plan_resolved.tsv"
python3 "$PLAN_RESOLVER" --plan "$PLAN_CSV" --root "$PLAN_ROOT_FOR_PARSE" --out "$RESOLVED_PLAN"
SAME_BIN_INPUT="$OUT_DIR/same_binary_no_pref_traces.tsv"
cut -f2 "$RESOLVED_PLAN" | sort -u | awk 'NF {print $0 "\t"}' > "$SAME_BIN_INPUT"
run_parallel run_same_binary_no_pref "$SAME_BIN_INPUT"
verify_same_binary_no_pref "$SAME_BIN_INPUT"
run_parallel run_candidate "$RESOLVED_PLAN"
python3 "$PARSER" --log-root "$LOG_DIR" --replay-input-root "$REPLAY_DIR" --out "$OUT_DIR/summary.csv" --baseline-summary "$BASELINE_SUMMARY" --same-binary-log-root "$LOG_DIR" --replay-plan "$PLAN_CSV" --plan-root "$PLAN_ROOT_FOR_PARSE" --winner-out "$OUT_DIR/winners.csv"

cat > "$OUT_DIR/RUN_INFO.txt" <<EOF
RUN_KIND=standalone_keyed_replay
TRACES=$TRACES
REPLAY_PLAN=$REPLAY_PLAN
PLAN_ROOT=$PLAN_ROOT_FOR_PARSE
WARMUP=$WARMUP
SIM=$SIM
MAX_JOBS=$MAX_JOBS
RUN_SAME_BINARY_NO_PREF=$RUN_SAME_BINARY_NO_PREF
SKIP_BASELINE_REFERENCE=$SKIP_BASELINE_REFERENCE
BASELINE_REFERENCE_JSON=$BASELINE_REFERENCE_JSON
BIN=$BIN
EOF

echo "[done] $OUT_DIR"
