#!/usr/bin/env bash
# Replay frozen standalone NN exports with PC-line-occurrence keys.
#
# Legacy mode replays one conventional list per trace from ART_DIR.
# Plan mode (REPLAY_PLAN=...) replays every fresh current-run list in a plan
# independently, preserving tag-specific logs and keyed inputs.
#
# Before candidate replay, a frozen same-binary/no-pref IPC guard verifies that
# the simulator/binary/window tuple did not drift from the registered reference.
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
REPLAY_PLAN="${REPLAY_PLAN:-}"
PLAN_ROOT="${PLAN_ROOT:-}"
# Set BASELINE_REFERENCE_JSON= to intentionally disable the guard for an
# explicitly new simulator/window experiment. Normal replay defaults to v4.0.
BASELINE_REFERENCE_JSON="${BASELINE_REFERENCE_JSON:-$ROOT/formal_NN_training/_cfg/replay_same_binary_no_pref_reference_v4_0.json}"
BASELINE_TOLERANCE="${BASELINE_TOLERANCE:-}"

BIN="${BIN:-$ROOT/external/ChampSim/bin/champsim.standalone_nn_replayer}"
PREP="$ROOT/formal_NN_training/scripts/07_prepare_keyed_replay_input.py"
PARSER="$ROOT/formal_NN_training/scripts/09_parse_standalone_lstm_replay.py"
BASELINE_GUARD="$ROOT/formal_NN_training/scripts/10_verify_same_binary_no_pref.py"
ORACLE_DIR="${ORACLE_DIR:-formal_NN_training/results/standalone_nn_data/oracle}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-formal_NN_training/results/prefetcher_baselines/summary.csv}"

mkdir -p "$LOG_DIR" "$REPLAY_DIR"
[[ -x "$BIN" ]] || { echo "[error] run scripts/06_install_keyed_listreplayer.sh first" >&2; exit 2; }
[[ -f "$PREP" && -f "$PARSER" && -f "$BASELINE_SUMMARY" ]] || { echo "[error] missing script or baseline table" >&2; exit 2; }
[[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "[error] MAX_JOBS must be a positive integer" >&2; exit 2; }
if [[ -n "$BASELINE_REFERENCE_JSON" ]]; then
  [[ -f "$BASELINE_REFERENCE_JSON" && -f "$BASELINE_GUARD" ]] || {
    echo "[error] missing baseline reference or guard: $BASELINE_REFERENCE_JSON" >&2; exit 2;
  }
fi

run_same_binary_no_pref() {
  local trace="$1"
  local trace_file="traces/${trace}.champsimtrace.xz"
  local same_bin_log="$LOG_DIR/${trace}.same_binary_no_pref.log"
  [[ -s "$trace_file" ]] || { echo "[error] missing trace file for $trace: $trace_file" >&2; return 1; }
  if [[ "$RUN_SAME_BINARY_NO_PREF" != "1" ]]; then
    return 0
  fi
  if [[ "$FORCE" != "1" && -s "$same_bin_log" ]]; then
    echo "[skip same-binary no_pref] $trace"
    return 0
  fi
  echo "[run same-binary no_pref] $trace"
  env -u PFETCH_LIST_PATH "$BIN" --l2c_prefetcher_types=none \
    --warmup_instructions="$WARMUP" --simulation_instructions="$SIM" \
    -traces "$trace_file" > "$same_bin_log" 2>&1
  grep -Fq 'Core_0_IPC' "$same_bin_log"
}

verify_same_binary_no_pref() {
  local trace_file="$1"
  [[ "$RUN_SAME_BINARY_NO_PREF" == "1" ]] || return 0
  [[ -n "$BASELINE_REFERENCE_JSON" ]] || {
    echo "[baseline guard skipped] BASELINE_REFERENCE_JSON is empty" >&2
    return 0
  }
  local args=(--reference "$BASELINE_REFERENCE_JSON" --log-root "$LOG_DIR"
              --out "$OUT_DIR/same_binary_no_pref_verification.json")
  if [[ -n "$BASELINE_TOLERANCE" ]]; then
    args+=(--tolerance "$BASELINE_TOLERANCE")
  fi
  local trace
  local traces=()
  while IFS=$'\t' read -r trace _; do
    [[ -n "$trace" ]] && traces+=("$trace")
  done < "$trace_file"
  python3 "$BASELINE_GUARD" "${args[@]}" --traces "${traces[@]}"
}

run_plan_candidate() {
  local tag="$1"
  local trace="$2"
  local rich="$3"
  local oracle="$ORACLE_DIR/${trace}.oracle.csv.gz"
  local keyed="$REPLAY_DIR/${tag}.pc_line_occ.csv"
  local log="$LOG_DIR/${tag}.standalone_lstm.log"
  local trace_file="traces/${trace}.champsimtrace.xz"

  [[ -s "$rich" && -s "$oracle" && -s "$trace_file" ]] || {
    echo "[error] missing input for tag=$tag trace=$trace rich=$rich oracle=$oracle trace_file=$trace_file" >&2
    return 1
  }
  if [[ "$FORCE" != "1" && -s "$log" && -s "$keyed" ]]; then
    echo "[skip replay] $tag ($trace)"
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

run_parallel_file() {
  local worker="$1"
  local infile="$2"
  local status=0
  local running=0
  local a b c
  while IFS=$'\t' read -r a b c; do
    [[ -n "$a" ]] || continue
    if [[ -n "$c" ]]; then
      "$worker" "$a" "$b" "$c" &
    elif [[ -n "$b" ]]; then
      "$worker" "$a" "$b" &
    else
      "$worker" "$a" &
    fi
    running=$((running + 1))
    if (( running >= MAX_JOBS )); then
      wait -n || status=1
      running=$((running - 1))
    fi
  done < "$infile"
  while (( running > 0 )); do
    wait -n || status=1
    running=$((running - 1))
  done
  return "$status"
}

if [[ -n "$REPLAY_PLAN" ]]; then
  [[ -f "$REPLAY_PLAN" ]] || { echo "[error] replay plan not found: $REPLAY_PLAN" >&2; exit 2; }
  if [[ -z "$PLAN_ROOT" ]]; then
    PLAN_ROOT="$(dirname "$REPLAY_PLAN")"
  fi
  [[ -d "$PLAN_ROOT" ]] || { echo "[error] plan root not found: $PLAN_ROOT" >&2; exit 2; }

  RESOLVED_PLAN="$OUT_DIR/replay_plan_resolved.tsv"
  python3 - "$REPLAY_PLAN" "$PLAN_ROOT" "$RESOLVED_PLAN" <<'PY'
import csv
import re
import sys
from pathlib import Path

plan = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
out = Path(sys.argv[3]).resolve()
required = {"tag", "trace", "source_rel"}
safe = re.compile(r"^[A-Za-z0-9_.-]+$")
rows = []
seen = set()
with plan.open(newline="") as handle:
    reader = csv.DictReader(handle)
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise SystemExit("[error] replay plan missing columns: {}".format(sorted(missing)))
    for line_no, row in enumerate(reader, start=2):
        tag = (row.get("tag") or "").strip()
        trace = (row.get("trace") or "").strip()
        source_rel = (row.get("source_rel") or "").strip()
        if not tag or not trace or not source_rel:
            raise SystemExit("[error] blank tag/trace/source_rel at plan row {}".format(line_no))
        if not safe.match(tag) or not safe.match(trace):
            raise SystemExit("[error] unsafe tag or trace at plan row {}".format(line_no))
        if tag in seen:
            raise SystemExit("[error] duplicate replay tag: {}".format(tag))
        seen.add(tag)
        rich = Path(source_rel)
        rich = rich if rich.is_absolute() else root / rich
        rich = rich.resolve()
        if not rich.is_file() or rich.stat().st_size <= 0:
            raise SystemExit("[error] missing/nonempty rich list for {}: {}".format(tag, rich))
        rows.append((tag, trace, str(rich)))
if not rows:
    raise SystemExit("[error] replay plan has no candidates")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as handle:
    for tag, trace, rich in rows:
        handle.write("{}\t{}\t{}\n".format(tag, trace, rich))
print("[plan preflight PASS] {} candidates".format(len(rows)))
for tag, trace, _ in rows:
    print("  {}  {}".format(trace, tag))
PY

  printf 'RUN_KIND=standalone_keyed_replay_plan\nREPLAY_PLAN=%s\nPLAN_ROOT=%s\n' "$REPLAY_PLAN" "$PLAN_ROOT" > "$OUT_DIR/RUN_INFO.txt"
  {
    printf 'WARMUP=%s\nSIM=%s\nMAX_JOBS=%s\nBIN=%s\nORACLE_DIR=%s\nBASELINE_SUMMARY=%s\nRUN_SAME_BINARY_NO_PREF=%s\nBASELINE_REFERENCE_JSON=%s\nBASELINE_TOLERANCE=%s\n' \
      "$WARMUP" "$SIM" "$MAX_JOBS" "$BIN" "$ORACLE_DIR" "$BASELINE_SUMMARY" "$RUN_SAME_BINARY_NO_PREF" "$BASELINE_REFERENCE_JSON" "$BASELINE_TOLERANCE"
  } >> "$OUT_DIR/RUN_INFO.txt"

  cut -f2 "$RESOLVED_PLAN" | sort -u | awk '{print $1 "\t"}' > "$OUT_DIR/unique_traces.tsv"
  run_parallel_file run_same_binary_no_pref "$OUT_DIR/unique_traces.tsv" || exit $?
  verify_same_binary_no_pref "$OUT_DIR/unique_traces.tsv"
  run_parallel_file run_plan_candidate "$RESOLVED_PLAN" || exit $?

  PLAN_STEM="$(basename "$REPLAY_PLAN" .csv)"
  WINNER_PREFIX="${PLAN_STEM%_replay_plan}"
  [[ "$WINNER_PREFIX" != "$PLAN_STEM" ]] || WINNER_PREFIX="$PLAN_STEM"
  python3 "$PARSER" \
    --log-root "$LOG_DIR" \
    --replay-input-root "$REPLAY_DIR" \
    --out "$OUT_DIR/summary.csv" \
    --baseline-summary "$BASELINE_SUMMARY" \
    --same-binary-log-root "$LOG_DIR" \
    --replay-plan "$REPLAY_PLAN" \
    --plan-root "$PLAN_ROOT" \
    --winner-out "$OUT_DIR/${WINNER_PREFIX}_nn_winners.csv"
  echo "[done] $OUT_DIR/summary.csv"
  exit 0
fi

# Legacy one-list-per-trace mode retained for previous workflows.
cat > "$OUT_DIR/RUN_INFO.txt" <<EOF
RUN_KIND=standalone_keyed_replay_legacy
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
BASELINE_REFERENCE_JSON=$BASELINE_REFERENCE_JSON
BASELINE_TOLERANCE=$BASELINE_TOLERANCE
EOF

legacy_plan="$OUT_DIR/legacy_plan.tsv"
: > "$legacy_plan"
for trace in $TRACES; do
  rich="$ART_DIR/prefetch_list_${trace}_cl${CHUNK_LEN}_${EXPORT_SUFFIX}.csv"
  printf '%s\t%s\t%s\n' "$trace" "$trace" "$rich" >> "$legacy_plan"
done
cut -f2 "$legacy_plan" | sort -u | awk '{print $1 "\t"}' > "$OUT_DIR/unique_traces.tsv"
run_parallel_file run_same_binary_no_pref "$OUT_DIR/unique_traces.tsv" || exit $?
verify_same_binary_no_pref "$OUT_DIR/unique_traces.tsv"
run_parallel_file run_plan_candidate "$legacy_plan" || exit $?

python3 "$PARSER" --log-root "$LOG_DIR" --replay-input-root "$REPLAY_DIR" --out "$OUT_DIR/summary.csv" --traces "$TRACES" --baseline-summary "$BASELINE_SUMMARY" --same-binary-log-root "$LOG_DIR"
echo "[done] $OUT_DIR/summary.csv"
