#!/usr/bin/env bash
# Run a Pythia/ChampSim prefetch behavior audit for multiple traces/prefetchers.
#
# Default first formal audit:
#   cd ~/cache
#   TRACES="602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B" \
#   WARMUP=25000000 SIM=25000000 MAX_JOBS=3 NODUP=1 \
#     bash formal_NN_training/scripts/18_run_prefetch_behavior_audit.sh
#
# Optional IPCP / combo audit, once configs are confirmed locally:
#   PREFETCHERS="no_pref spp ipcp spp_ipcp" bash formal_NN_training/scripts/18_run_prefetch_behavior_audit.sh
#
# Output:
#   formal_NN_training/results/LSTM/behavior_audit/logs/*.log
#   formal_NN_training/results/LSTM/behavior_audit/summary_nodup.csv

set -euo pipefail

if git rev-parse --show-toplevel >/dev/null 2>&1; then
  ROOT="$(git rev-parse --show-toplevel)"
else
  ROOT="${ROOT:-$HOME/cache}"
fi
cd "$ROOT"

TRACES_STR="${TRACES:-602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B}"
# Keep the default focused on SPP behavior. IPCP support is built in but opt-in.
PREFETCHERS_STR="${PREFETCHERS:-no_pref spp}"

WARMUP="${WARMUP:-25000000}"
SIM="${SIM:-25000000}"
MAX_JOBS="${MAX_JOBS:-3}"
FORCE_REPLAY="${FORCE_REPLAY:-0}"
BUILD="${BUILD:-1}"
NODUP="${NODUP:-1}"

TRACE_DIR="${TRACE_DIR:-$ROOT/traces}"
CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
OUT_ROOT="${OUT_ROOT:-$ROOT/formal_NN_training/results/LSTM/behavior_audit}"
LOG_DIR="$OUT_ROOT/logs"
CFG_DIR="$ROOT/formal_NN_training/_cfg/pythia_prefetchers"
BIN="${BIN:-$CHAMP_DIR/bin/perceptron-no-multi-no-ship-1core}"

mkdir -p "$LOG_DIR" "$CFG_DIR"

if [ ! -d "$CHAMP_DIR" ]; then
  echo "[error] missing ChampSim/Pythia directory: $CHAMP_DIR"
  echo "        Expected repo layout: ~/cache/external/ChampSim"
  exit 1
fi

write_cfgs () {
  cat > "$CFG_DIR/no_pref.ini" <<'EOF'
l2c_prefetcher_types = none
EOF

  cat > "$CFG_DIR/spp.ini" <<'EOF'
l2c_prefetcher_types = spp_dev2
spp_dev2_fill_threshold = 90
spp_dev2_pf_threshold = 40
EOF

  cat > "$CFG_DIR/ipcp.ini" <<'EOF'
l2c_prefetcher_types = ipcp
EOF

  cat > "$CFG_DIR/spp_ipcp.ini" <<'EOF'
l2c_prefetcher_types = spp_dev2,ipcp
spp_dev2_fill_threshold = 90
spp_dev2_pf_threshold = 40
EOF
}

build_if_needed () {
  if [ "$BUILD" != "1" ] && [ -x "$BIN" ]; then
    return 0
  fi
  if [ -x "$BIN" ]; then
    echo "[build skip] existing binary: $BIN"
    return 0
  fi
  echo "[build] Pythia multi-L2 binary"
  echo "        $CHAMP_DIR/build_champsim.sh no multi no 1"
  (
    cd "$CHAMP_DIR"
    bash ./build_champsim.sh no multi no 1
  )
  if [ ! -x "$BIN" ]; then
    echo "[error] expected binary missing after build: $BIN"
    echo "        Check bin/ under $CHAMP_DIR"
    exit 1
  fi
}

pref_types () {
  case "$1" in
    no_pref|none|nopref) echo "none" ;;
    spp|spp_dev2) echo "spp_dev2" ;;
    ipcp) echo "ipcp" ;;
    spp_ipcp|spp+ipcp) echo "spp_dev2,ipcp" ;;
    *) echo "$1" ;;
  esac
}

pref_cfg () {
  case "$1" in
    no_pref|none|nopref) echo "$CFG_DIR/no_pref.ini" ;;
    spp|spp_dev2) echo "$CFG_DIR/spp.ini" ;;
    ipcp) echo "$CFG_DIR/ipcp.ini" ;;
    spp_ipcp|spp+ipcp) echo "$CFG_DIR/spp_ipcp.ini" ;;
    *) echo "$CFG_DIR/no_pref.ini" ;;
  esac
}

run_one () {
  local trace="$1"
  local pf="$2"
  local trfile="$TRACE_DIR/${trace}.champsimtrace.xz"
  local log="$LOG_DIR/${trace}.${pf}.log"
  local types cfg
  types="$(pref_types "$pf")"
  cfg="$(pref_cfg "$pf")"

  if [ ! -s "$trfile" ]; then
    echo "[skip missing trace] $trfile"
    return 0
  fi

  if [ "$FORCE_REPLAY" != "1" ] && [ -s "$log" ]; then
    echo "[skip existing] $log"
    return 0
  fi

  echo "============================================================"
  echo "[run audit] trace=$trace prefetcher=$pf types=$types"
  echo "log        : $log"
  echo "warmup/sim : $WARMUP / $SIM"
  echo "============================================================"

  "$BIN" \
    --warmup_instructions="$WARMUP" \
    --simulation_instructions="$SIM" \
    --config="$cfg" \
    --l2c_prefetcher_types="$types" \
    -traces "$trfile" \
    > "$log" 2>&1
}

wait_slot () {
  local running_ref="$1"
  local running="${!running_ref}"
  if [ "$running" -ge "$MAX_JOBS" ]; then
    wait -n
    running=$((running - 1))
    printf -v "$running_ref" '%s' "$running"
  fi
}

write_cfgs
build_if_needed

cat > "$OUT_ROOT/RUN_INFO.txt" <<EOF
RUN_KIND=prefetch_behavior_audit
ROOT=$ROOT
CHAMP_DIR=$CHAMP_DIR
BIN=$BIN
TRACE_DIR=$TRACE_DIR
TRACES=$TRACES_STR
PREFETCHERS=$PREFETCHERS_STR
WARMUP=$WARMUP
SIM=$SIM
MAX_JOBS=$MAX_JOBS
NODUP=$NODUP
FORCE_REPLAY=$FORCE_REPLAY
EOF

running=0
for trace in $TRACES_STR; do
  for pf in $PREFETCHERS_STR; do
    run_one "$trace" "$pf" &
    running=$((running + 1))
    wait_slot running
  done
done
wait

SUMMARY="$OUT_ROOT/summary.csv"
if [ "$NODUP" = "1" ]; then
  SUMMARY="$OUT_ROOT/summary_nodup.csv"
  NODUP_FLAG="--nodup"
else
  NODUP_FLAG=""
fi

python3 formal_NN_training/scripts/17_parse_prefetch_behavior_audit.py \
  --log-root "$LOG_DIR" \
  --out "$SUMMARY" \
  --traces "$TRACES_STR" \
  --prefetchers "$PREFETCHERS_STR" \
  $NODUP_FLAG

echo
echo "[done] behavior audit"
echo "  logs   : $LOG_DIR"
echo "  summary: $SUMMARY"
