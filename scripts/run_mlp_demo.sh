#!/usr/bin/env bash
# run_mlp_demo.sh -- v3 (fixed trace path issue)
#
# Quangmire/ChampSim-ML's run_champsim.sh looks for traces in a relative
# directory called dpc3_traces/ inside the ChampSim-ML repo. v2 of this
# script passed an absolute trace path which the older fork's run_champsim.sh
# doesn't accept. Fix: create dpc3_traces/ as a symlink to our traces/ dir
# and call run_champsim.sh with just the trace basename.

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

cd "$ML_DIR"

if [ ! -e "$ML_DIR/dpc3_traces" ]; then
  ln -s "$TRACE_DIR" "$ML_DIR/dpc3_traces"
  echo "[fix] symlinked $TRACE_DIR -> $ML_DIR/dpc3_traces"
fi

BIN_TAG="bimodal-no-no-no-no-lru-1core"
if [ ! -x "bin/${BIN_TAG}" ]; then
  echo "[build] one-time build (~3 min)"
  ./build_champsim.sh bimodal no no no no lru 1 2>&1 | tail -5
fi
if [ ! -x "bin/${BIN_TAG}" ]; then
  echo "[error] ChampSim-ML build failed."
  ls bin/ 2>&1 || true
  exit 2
fi

echo
echo "============================================================"
echo "[run] baseline (no prefetch)  trace=$TRACE"
echo "============================================================"
BASE_LOG="$OUT_DIR/mlp_demo/baseline.${TRACE}.log"

echo "[cmd] ./run_champsim.sh $BIN_TAG 1 5 $TRACE"
./run_champsim.sh "$BIN_TAG" 1 5 "$TRACE" 2>&1 | tee "$BASE_LOG" | tail -25

RESULT_DIR="$ML_DIR/results_5M"
BASE_RESULT_FILE="$RESULT_DIR/${TRACE}-${BIN_TAG}.txt"
if [ ! -f "$BASE_RESULT_FILE" ]; then
  echo "[warn] expected result file $BASE_RESULT_FILE not found"
  ls "$RESULT_DIR" 2>&1 || true
fi

echo
echo "============================================================"
echo "[run] NN prefetch list ($PFETCH)  trace=$TRACE"
echo "============================================================"
NN_LOG="$OUT_DIR/mlp_demo/nn.${TRACE}.log"

if [ -f ./ml_prefetch_sim.py ]; then
  ./ml_prefetch_sim.py run "$TR_FILE" --prefetch "$PFETCH" 2>&1 | tee "$NN_LOG" | tail -25
else
  echo "[warn] ./ml_prefetch_sim.py not present in $ML_DIR. Skipping NN comparison."
  echo "[hint] cd $ML_DIR && git pull   # to refresh master"
fi

parse_ipc () {
  grep -E "cumulative IPC" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1
}

BASE_IPC=$(parse_ipc "$BASE_LOG")
if [ -z "$BASE_IPC" ] && [ -f "$BASE_RESULT_FILE" ]; then
  BASE_IPC=$(parse_ipc "$BASE_RESULT_FILE")
fi
NN_IPC=$(parse_ipc "$NN_LOG")
BASE_IPC="${BASE_IPC:-NA}"
NN_IPC="${NN_IPC:-NA}"

echo
echo "============================================================"
echo "RESULTS"
echo "============================================================"
echo "Trace          : $TRACE"
echo "Baseline IPC   : $BASE_IPC"
echo "NN-prefetch IPC: $NN_IPC"
if [ "$BASE_IPC" != "NA" ] && [ "$NN_IPC" != "NA" ]; then
  SPEEDUP=$(python3 -c "print(f'{float(\"$NN_IPC\")/float(\"$BASE_IPC\"):.3f}')")
  echo "Speedup        : ${SPEEDUP}x"
fi
echo "============================================================"

SUMMARY="$OUT_DIR/mlp_demo_summary.csv"
[ -f "$SUMMARY" ] || echo "trace,baseline_IPC,nn_IPC,speedup" > "$SUMMARY"
echo "$TRACE,$BASE_IPC,$NN_IPC,${SPEEDUP:-NA}" >> "$SUMMARY"
echo "[done] saved to $SUMMARY"