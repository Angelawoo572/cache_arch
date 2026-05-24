#!/usr/bin/env bash
# run_bypass.sh
# Compares baseline LRU vs LRU+bypass on a single trace.
#
# Inputs:
#   TRACE          (default 605.mcf_s-994B)
#   BYPASS_PC_LIST (default $WORKDIR/bypass_pc_list.txt)
#   WARMUP / SIM   (default 1M / 5M)
#
# Output: appends one row to results/bypass_summary.csv

set -uo pipefail

WORKDIR="$(pwd)"
CHAMP_DIR="$WORKDIR/external/ChampSim"
TRACE_DIR="$WORKDIR/traces"
OUT_DIR="$WORKDIR/results"
mkdir -p "$OUT_DIR"

TRACE="${TRACE:-605.mcf_s-994B}"
WARMUP="${WARMUP:-1000000}"
SIM="${SIM:-5000000}"
BYPASS_PC_LIST="${BYPASS_PC_LIST:-$WORKDIR/bypass_pc_list.txt}"
TAG="${TAG:-default}"

TR_FILE="$TRACE_DIR/${TRACE}.champsimtrace.xz"
BASE_BIN="$CHAMP_DIR/bin/champsim.baseline"
BYP_BIN="$CHAMP_DIR/bin/champsim.bypass_lru"

[ -x "$BASE_BIN" ] || { echo "[error] $BASE_BIN missing -- run install_and_build.sh"; exit 1; }
[ -x "$BYP_BIN"  ] || { echo "[error] $BYP_BIN missing -- run install_bypass.sh"; exit 1; }
[ -f "$TR_FILE" ] || { echo "[error] trace $TR_FILE missing"; exit 1; }
[ -f "$BYPASS_PC_LIST" ] || { echo "[error] BYPASS_PC_LIST=$BYPASS_PC_LIST missing"; exit 1; }

echo "[config] TRACE=$TRACE  TAG=$TAG"
echo "[config] BYPASS_PC_LIST=$BYPASS_PC_LIST  ($(grep -cv '^#\|^$' "$BYPASS_PC_LIST") PCs)"

run_with_heartbeat () {
  local logf=$1; shift
  "$@" > "$logf" 2>&1 &
  local pid=$!; local s=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30; s=$((s+30))
    local last=$(tail -1 "$logf" 2>/dev/null | tr -d '\n' | head -c 70)
    printf "  ...running (%ds)  last: %s\n" "$s" "$last"
  done
  wait "$pid"
}

parse_ipc () {
  grep -E "cumulative IPC" "$1" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1
}

# ---- baseline ----
echo
echo "============================================"
echo "[run] BASELINE (LRU, no bypass)  trace=$TRACE"
echo "============================================"
BASE_LOG="$OUT_DIR/bypass_demo.baseline.${TRACE}.${TAG}.log"
run_with_heartbeat "$BASE_LOG" \
  "$BASE_BIN" --warmup-instructions "$WARMUP" --simulation-instructions "$SIM" "$TR_FILE"
BASE_IPC=$(parse_ipc "$BASE_LOG")
echo "[parse] baseline IPC = '${BASE_IPC:-<empty>}'"

# ---- bypass ----
echo
echo "============================================"
echo "[run] LRU + BYPASS    trace=$TRACE  list=$BYPASS_PC_LIST"
echo "============================================"
BYP_LOG="$OUT_DIR/bypass_demo.bypass.${TRACE}.${TAG}.log"
export BYPASS_PC_LIST
run_with_heartbeat "$BYP_LOG" \
  "$BYP_BIN" --warmup-instructions "$WARMUP" --simulation-instructions "$SIM" "$TR_FILE"
BYP_IPC=$(parse_ipc "$BYP_LOG")
BYP_STATS=$(grep "bypassed.*fills" "$BYP_LOG" | tail -1)
echo "[parse] bypass IPC   = '${BYP_IPC:-<empty>}'"
echo "[parse] bypass stats = '$BYP_STATS'"

# ---- summary ----
BASE_IPC="${BASE_IPC:-NA}"; BYP_IPC="${BYP_IPC:-NA}"
SPEEDUP="NA"
if [[ "$BASE_IPC" =~ ^[0-9.]+$ ]] && [[ "$BYP_IPC" =~ ^[0-9.]+$ ]]; then
  SPEEDUP=$(python3 -c "print(f'{float(\"$BYP_IPC\")/float(\"$BASE_IPC\"):.4f}')")
fi
BYP_N=$(echo "$BYP_STATS"  | grep -oE "bypassed [0-9]+" | grep -oE "[0-9]+" | head -1)
TOT_N=$(echo "$BYP_STATS"  | grep -oE "of [0-9]+"      | grep -oE "[0-9]+" | head -1)
LIST_N=$(grep -cv '^#\|^$' "$BYPASS_PC_LIST")
BYP_N="${BYP_N:-NA}"; TOT_N="${TOT_N:-NA}"

echo
echo "============================================"
echo "RESULTS  trace=$TRACE  tag=$TAG"
echo "  baseline IPC : $BASE_IPC"
echo "  bypass IPC   : $BYP_IPC"
echo "  speedup      : ${SPEEDUP}x"
echo "  bypassed     : $BYP_N / $TOT_N fills"
echo "  PC list size : $LIST_N"
echo "============================================"

SUMMARY="$OUT_DIR/bypass_summary.csv"
if [ ! -f "$SUMMARY" ] || ! grep -q "^trace,baseline_IPC" "$SUMMARY"; then
  echo "trace,baseline_IPC,bypass_IPC,speedup,pc_list_size,bypassed,total_fills,tag" > "$SUMMARY"
fi
echo "$TRACE,$BASE_IPC,$BYP_IPC,$SPEEDUP,$LIST_N,$BYP_N,$TOT_N,$TAG" >> "$SUMMARY"
echo "[done] appended to $SUMMARY"
cat "$SUMMARY"
