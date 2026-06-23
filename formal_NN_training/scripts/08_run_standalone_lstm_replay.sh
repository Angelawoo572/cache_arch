#!/usr/bin/env bash
# Replay frozen standalone NN exports with PC-line-occurrence keys.
#
# This is offline keyed replay, not in-simulator PyTorch inference. The driver
# also runs a no-prefetch control with the exact same replayer binary so the
# replay summary has both current-normal and same-binary comparisons.
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

BIN="${BIN:-$ROOT/external/ChampSim/bin/champsim.standalone_nn_replayer}"
PREP="$ROOT/formal_NN_training/scripts/07_prepare_keyed_replay_input.py"
PARSER="$ROOT/formal_NN_training/scripts/09_parse_standalone_lstm_replay.py"
ORACLE_DIR="${ORACLE_DIR:-formal_NN_training/results/standalone_nn_data/oracle}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-formal_NN_training/results/prefetcher_baselines/summary.csv}"

mkdir -p "$LOG_DIR" "$REPLAY_DIR"
[[ -x "$BIN" ]] || { echo "[error] run scripts/06_install_keyed_listreplayer.sh first" >&2; exit 2; }
[[ -f "$PREP" && -f "$PARSER" && -f "$BASELINE_SUMMARY" ]] || { echo "[error] missing script or baseline table" >&2; exit 2; }

run_one() {
  local trace="$1"
  local rich="$ART_DIR/prefetch_list_${trace}_cl${CHUNK_LEN}_${EXPORT_SUFFIX}.csv"
  local oracle="$ORACLE_DIR/${trace}.oracle.csv.gz"
  local keyed="$REPLAY_DIR/${trace}.pc_line_occ.csv"
  local log="$LOG_DIR/${trace}.standalone_lstm.log"
  local same_bin_log="$LOG_DIR/${trace}.same_binary_no_pref.log"
  local trace_file="traces/${trace}.champsimtrace.xz"
  [[ -s "$rich" && -s "$oracle" && -s "$trace_file" ]] || { echo "[error] missing input for $trace" >&2; return 1; }

  if [[ "$RUN_SAME_BINARY_NO_PREF" == "1" && ( "$FORCE" == "1" || ! -s "$same_bin_log" ) ]]; then
    echo "[run same-binary no_pref] $trace"
    env -u PFETCH_LIST_PATH "$BIN" --l2c_prefetcher_types=none \
      --warmup_instructions="$WARMUP" --simulation_instructions="$SIM" \
      -traces "$trace_file" > "$same_bin_log" 2>&1
    grep -Fq 'Core_0_IPC' "$same_bin_log"
  fi

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

cat > "$OUT_DIR/RUN_INFO.txt" <<EOF
RUN_KIND=standalone_keyed_replay
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
EOF

running=0
status=0
for trace in $TRACES; do
  run_one "$trace" &
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
echo "[done] $OUT_DIR/summary.csv"