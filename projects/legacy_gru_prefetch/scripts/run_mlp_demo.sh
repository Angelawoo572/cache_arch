#!/usr/bin/env bash
# run_mlp_demo.sh
# Older ChampSim-ML is fragile. This script only handles its baseline path
# conservatively and points NN replay users to projects/legacy_gru_prefetch/scripts/run_nn_replay.sh, which
# uses the main external/ChampSim tree and is more reliable.

set -uo pipefail

WORKDIR="$(pwd)"
ML_DIR="$WORKDIR/external/ChampSim-ML"
TRACE_DIR="$WORKDIR/traces"
OUT_DIR="$WORKDIR/results"
mkdir -p "$OUT_DIR/mlp_demo"

if [ ! -d "$ML_DIR" ]; then
  echo "[error] $ML_DIR not found. Run setup_champsim.sh first."
  exit 1
fi

TRACE="${TRACE:-605.mcf_s-994B}"
TR_FILE="$TRACE_DIR/${TRACE}.champsimtrace.xz"
TR_BASENAME="${TRACE}.champsimtrace.xz"
if [ ! -f "$TR_FILE" ]; then
  echo "[error] Trace $TR_FILE not found."
  echo "[hint] Available traces:"
  ls "$TRACE_DIR" 2>/dev/null
  exit 1
fi

PFETCH="${PFETCH:-$WORKDIR/prefetch_list.txt}"
if [ ! -f "$PFETCH" ]; then
  echo "[error] Prefetch list not found at $PFETCH"
  echo "[hint] Download prefetch_list.txt from the Colab notebook to: $PFETCH"
  exit 1
fi

cd "$ML_DIR" || exit 1

# Quangmire/ChampSim-ML run_champsim.sh searches ./dpc3_traces/<trace-arg>.
# Therefore the argument must include the .champsimtrace.xz suffix.
if [ -L "$ML_DIR/dpc3_traces" ] || [ ! -e "$ML_DIR/dpc3_traces" ]; then
  rm -f "$ML_DIR/dpc3_traces"
  ln -s "$TRACE_DIR" "$ML_DIR/dpc3_traces"
  echo "[fix] symlinked $TRACE_DIR -> $ML_DIR/dpc3_traces"
fi

BIN_TAG="bimodal-no-no-no-no-lru-1core"
if [ ! -x "bin/${BIN_TAG}" ]; then
  echo "[build] one-time build (~3 min)"
  ./build_champsim.sh bimodal no no no no lru 1 2>&1 | tail -20
fi
if [ ! -x "bin/${BIN_TAG}" ]; then
  echo "[error] ChampSim-ML build failed."
  ls bin/ 2>&1 || true
  exit 2
fi

echo
echo "============================================================"
echo "[run] ChampSim-ML baseline (no prefetch)  trace=$TR_BASENAME"
echo "============================================================"
BASE_LOG="$OUT_DIR/mlp_demo/baseline.${TRACE}.log"

echo "[cmd] ./run_champsim.sh $BIN_TAG 1 5 $TR_BASENAME"
./run_champsim.sh "$BIN_TAG" 1 5 "$TR_BASENAME" 2>&1 | tee "$BASE_LOG" | tail -25

RESULT_DIR="$ML_DIR/results_5M"
BASE_RESULT_FILE="$RESULT_DIR/${TR_BASENAME}-${BIN_TAG}.txt"
if [ ! -f "$BASE_RESULT_FILE" ]; then
  echo "[warn] expected result file $BASE_RESULT_FILE not found"
  ls "$RESULT_DIR" 2>&1 || true
fi

parse_ipc () {
  grep -E "cumulative IPC" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1
}

BASE_IPC=$(parse_ipc "$BASE_LOG")
if [ -z "$BASE_IPC" ] && [ -f "$BASE_RESULT_FILE" ]; then
  BASE_IPC=$(parse_ipc "$BASE_RESULT_FILE")
fi
BASE_IPC="${BASE_IPC:-NA}"

echo
echo "============================================================"
echo "RESULTS"
echo "============================================================"
echo "Trace                : $TRACE"
echo "ChampSim-ML baseline : $BASE_IPC"
echo
echo "[note] For NN replay, use the main ChampSim path instead:"
echo "       bash projects/legacy_gru_prefetch/scripts/install_and_build.sh"
echo "       bash projects/legacy_gru_prefetch/scripts/run_nn_replay.sh"
echo "============================================================"

SUMMARY="$OUT_DIR/mlp_demo_summary.csv"
[ -f "$SUMMARY" ] || echo "trace,baseline_IPC,nn_IPC,speedup" > "$SUMMARY"
echo "$TRACE,$BASE_IPC,NA,NA" >> "$SUMMARY"
echo "[done] saved to $SUMMARY"
