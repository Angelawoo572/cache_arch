#!/usr/bin/env bash
# Replay frozen standalone exports through the keyed ListReplayer.
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
elif [[ -z "${BASELINE_REFERENCE_JSON+x}" ]]; then
  BASELINE_REFERENCE_JSON="$ROOT/formal_NN_training/_cfg/replay_same_binary_no_pref_reference_v4_0.json"
fi

mkdir -p "$LOG_DIR" "$REPLAY_DIR"
[[ -x "$BIN" ]] || { echo "[error] run scripts/06_install_keyed_listreplayer.sh first" >&2; exit 2; }
[[ -f "$PREP" && -f "$PARSER" && -f "$PLAN_RESOLVER" && -f "$BASELINE_SUMMARY" ]] || { echo "[error] missing replay helper or baseline table" >&2; exit 2; }
[[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "[error] MAX_JOBS must be a positive integer" >&2; exit 2; }
[[ -z "${BASELINE_REFERENCE_JSON:-}" || -f "$BASELINE_REFERENCE_JSON" ]] || { echo "[error] missing baseline reference: $BASELINE_REFERENCE_JSON" >&2; exit 2; }
[[ -z "${BASELINE_REFERENCE_JSON:-}" || -f "$BASELINE_GUARD" ]] || { echo "[error] missing baseline guard: $BASELINE_GUARD" >&2; exit 2; }

has() { [[ -s "$1" ]] && grep -Fq "$2" "$1"; }
replay_done() { has "$1" 'adding L2C_PREFETCHER: list_replayer' && has "$1" 'PC-line-occ triggers' && has "$1" 'key=pc_line_occ'; }

run_same_binary_no_pref() {
  local trace="$1" trace_file="traces/${1}.champsimtrace.xz" log="$LOG_DIR/${1}.same_binary_no_pref.log"
  [[ -s "$trace_file" ]] || { echo "[error] missing trace: $trace_file" >&2; return 1; }
  [[ "$RUN_SAME_BINARY_NO_PREF" == 1 ]] || return 0
  if [[ "$FORCE" != 1 ]] && has "$log" Core_0_IPC; then echo "[skip same-binary no_pref] $trace"; return; fi
  echo "[run same-binary no_pref] $trace"
  env -u PFETCH_LIST_PATH "$BIN" --l2c_prefetcher_types=none --warmup_instructions="$WARMUP" --simulation_instructions="$SIM" -traces "$trace_file" > "$log" 2>&1
  has "$log" Core_0_IPC || { echo "[error] same-binary no_pref has no final IPC: $trace" >&2; return 1; }
}

verify_same_binary_no_pref() {
  local input="$1" trace traces=()
  [[ "$RUN_SAME_BINARY_NO_PREF" == 1 ]] || return 0
  [[ -n "${BASELINE_REFERENCE_JSON:-}" ]] || { echo "[baseline guard skipped] no frozen reference configured" >&2; return 0; }
  while IFS=$'\t' read -r trace _; do [[ -n "$trace" ]] && traces+=("$trace"); done < "$input"
  local args=(--reference "$BASELINE_REFERENCE_JSON" --log-root "$LOG_DIR" --out "$OUT_DIR/same_binary_no_pref_verification.json")
  [[ -z "$BASELINE_TOLERANCE" ]] || args+=(--tolerance "$BASELINE_TOLERANCE")
  python3 "$BASELINE_GUARD" "${args[@]}" --traces "${traces[@]}"
}

run_candidate() {
  local tag="$1" trace="$2" rich="$3" oracle="$ORACLE_DIR/${2}.oracle.csv.gz" keyed="$REPLAY_DIR/${1}.pc_line_occ.csv" log="$LOG_DIR/${1}.standalone_lstm.log" trace_file="traces/${2}.champsimtrace.xz"
  [[ -s "$rich" && -s "$oracle" && -s "$trace_file" ]] || { echo "[error] missing replay input for $tag" >&2; return 1; }
  if [[ "$FORCE" != 1 && -s "$keyed" ]] && replay_done "$log"; then echo "[skip replay] $tag ($trace)"; return; fi
  python3 "$PREP" --rich-list "$rich" --oracle "$oracle" --out "$keyed" > "$LOG_DIR/${tag}.prepare.log" 2>&1
  echo "[replay] $tag ($trace)"
  PFETCH_LIST_PATH="$keyed" "$BIN" --l2c_prefetcher_types=list_replayer --warmup_instructions="$WARMUP" --simulation_instructions="$SIM" -traces "$trace_file" > "$log" 2>&1
  replay_done "$log" || { echo "[error] incomplete replay log: $tag ($trace)" >&2; return 1; }
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
  PLAN_CSV="$REPLAY_PLAN"; PLAN_ROOT_FOR_PARSE="$PLAN_ROOT"
else
  PLAN_CSV="$OUT_DIR/legacy_replay_plan.csv"; PLAN_ROOT_FOR_PARSE="$ROOT"
  printf 'tag,trace,source_rel\n' > "$PLAN_CSV"
  for trace in $TRACES; do printf 'legacy_%s,%s,%s\n' "$trace" "$trace" "$ART_DIR/prefetch_list_${trace}_cl${CHUNK_LEN}_${EXPORT_SUFFIX}.csv" >> "$PLAN_CSV"; done
fi

RESOLVED_PLAN="$OUT_DIR/replay_plan_resolved.tsv"
python3 "$PLAN_RESOLVER" --plan "$PLAN_CSV" --root "$PLAN_ROOT_FOR_PARSE" --out "$RESOLVED_PLAN"
SAME_BIN_INPUT="$OUT_DIR/same_binary_no_pref_traces.tsv"
cut -f2 "$RESOLVED_PLAN" | sort -u | awk 'NF {print $0 "\t"}' > "$SAME_BIN_INPUT"
run_parallel run_same_binary_no_pref "$SAME_BIN_INPUT"
verify_same_binary_no_pref "$SAME_BIN_INPUT"
run_parallel run_candidate "$RESOLVED_PLAN"
python3 "$PARSER" --log-root "$LOG_DIR" --replay-input-root "$REPLAY_DIR" --out "$OUT_DIR/summary.csv" --baseline-summary "$BASELINE_SUMMARY" --same-binary-log-root "$LOG_DIR" --replay-plan "$PLAN_CSV" --plan-root "$PLAN_ROOT_FOR_PARSE" --winner-out "$OUT_DIR/winners.csv"
printf 'RUN_KIND=standalone_keyed_replay\nTRACES=%s\nREPLAY_PLAN=%s\nPLAN_ROOT=%s\nWARMUP=%s\nSIM=%s\nMAX_JOBS=%s\nRUN_SAME_BINARY_NO_PREF=%s\nSKIP_BASELINE_REFERENCE=%s\nBASELINE_REFERENCE_JSON=%s\nBIN=%s\n' "$TRACES" "$REPLAY_PLAN" "$PLAN_ROOT_FOR_PARSE" "$WARMUP" "$SIM" "$MAX_JOBS" "$RUN_SAME_BINARY_NO_PREF" "$SKIP_BASELINE_REFERENCE" "${BASELINE_REFERENCE_JSON:-}" "$BIN" > "$OUT_DIR/RUN_INFO.txt"
echo "[done] $OUT_DIR"
