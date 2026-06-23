#!/usr/bin/env bash
# Replay frozen standalone LSTM exports with PC-line-occurrence keys.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
TRACES="${TRACES:-602.gcc_s-734B 619.lbm_s-4268B}"
WARMUP="${WARMUP:-25000000}"
SIM="${SIM:-25000000}"
MAX_JOBS="${MAX_JOBS:-2}"
CHUNK_LEN="${CHUNK_LEN:-128}"
DEDUP_CAPACITY="${DEDUP_CAPACITY:-256}"
EXPORT_SUFFIX="${EXPORT_SUFFIX:-fair_dedup_lru${DEDUP_CAPACITY}}"
ART_DIR="${ART_DIR:-formal_NN_training/artifacts/standalone_multihorizon_lstm}"
RUN_TAG="${RUN_TAG:-standalone_lstm_cl${CHUNK_LEN}_${EXPORT_SUFFIX}}"
OUT_DIR="${OUT_DIR:-formal_NN_training/results/standalone_lstm_replay/${RUN_TAG}}"
LOG_DIR="$OUT_DIR/logs"
REPLAY_DIR="$OUT_DIR/replay_inputs"
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
  local trace_file="traces/${trace}.champsimtrace.xz"
  [[ -s "$rich" && -s "$oracle" && -s "$trace_file" ]] || { echo "[error] missing input for $trace" >&2; return 1; }
  python3 "$PREP" --rich-list "$rich" --oracle "$oracle" --out "$keyed" > "$LOG_DIR/${trace}.prepare.log" 2>&1
  PFETCH_LIST_PATH="$keyed" "$BIN" --l2c_prefetcher_types=list_replayer --warmup_instructions="$WARMUP" --simulation_instructions="$SIM" -traces "$trace_file" > "$log" 2>&1
  grep -Fq 'adding L2C_PREFETCHER: list_replayer' "$log"
  grep -Fq 'PC-line-occ triggers' "$log"
  grep -Fq 'key=pc_line_occ' "$log"
  echo "[done] $trace"
}

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

python3 "$PARSER" --log-root "$LOG_DIR" --replay-input-root "$REPLAY_DIR" --out "$OUT_DIR/summary.csv" --traces "$TRACES" --baseline-summary "$BASELINE_SUMMARY"
echo "[done] $OUT_DIR/summary.csv"
