#!/usr/bin/env bash
# Run demand-centric residual audit for Pythia L2 prefetchers.
#
# This emits one demand-centric event CSV per trace/prefetcher using a Pythia
# binary that has already been patched for RESIDUAL_AUDIT_LOG.
#
# Default formal run:
#   cd ~/cache
#   git pull
#   TRACES="602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B 620.omnetpp_s-874B 623.xalancbmk_s-700B" \
#   PREFETCHERS="no_pref spp ipcp sms ampm sandbox streamer stride power7" \
#   OUT_ROOT=formal_NN_training/results/base_prefetcher_zoo/residual_audit \
#   WARMUP=25000000 SIM=25000000 MAX_JOBS=4 BUILD=0 COMPRESS=1 FORCE_REPLAY=0 \
#     bash formal_NN_training/scripts/05_run_residual_demand_audit.sh
#
# Output:
#   $OUT_ROOT/events/*.events.csv.gz
#   $OUT_ROOT/logs/*.log
#   $OUT_ROOT/summary.csv

set -euo pipefail

if git rev-parse --show-toplevel >/dev/null 2>&1; then
  ROOT="$(git rev-parse --show-toplevel)"
else
  ROOT="${ROOT:-$HOME/cache}"
fi
cd "$ROOT"

DEFAULT_TRACES="602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B 620.omnetpp_s-874B 623.xalancbmk_s-700B"
DEFAULT_PREFETCHERS="no_pref spp ipcp sms ampm sandbox streamer stride power7"

TRACES_STR="${TRACES:-$DEFAULT_TRACES}"
PREFETCHERS_STR="${PREFETCHERS:-$DEFAULT_PREFETCHERS}"

WARMUP="${WARMUP:-25000000}"
SIM="${SIM:-25000000}"
MAX_JOBS="${MAX_JOBS:-3}"
FORCE_REPLAY="${FORCE_REPLAY:-0}"
BUILD="${BUILD:-1}"
RESET_PATCH="${RESET_PATCH:-0}"
COMPRESS="${COMPRESS:-1}"

TRACE_DIR="${TRACE_DIR:-$ROOT/traces}"
CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
OUT_ROOT="${OUT_ROOT:-$ROOT/formal_NN_training/results/LSTM/residual_audit}"
LOG_DIR="$OUT_ROOT/logs"
EVENT_DIR="$OUT_ROOT/events"
CFG_DIR="$ROOT/formal_NN_training/_cfg/pythia_prefetchers"
BIN="${BIN:-$CHAMP_DIR/bin/perceptron-no-multi-no-ship-1core}"

mkdir -p "$LOG_DIR" "$EVENT_DIR" "$CFG_DIR"

if [ ! -d "$CHAMP_DIR" ]; then
  echo "[error] missing ChampSim/Pythia directory: $CHAMP_DIR"
  echo "        Expected repo layout: ~/cache/external/ChampSim"
  exit 1
fi

ensure_libbf () {
  if [ -f "$CHAMP_DIR/libbf/bf/all.hpp" ] && [ -f "$CHAMP_DIR/libbf/build/lib/libbf.a" ]; then
    echo "[libbf] found"
    return 0
  fi

  echo "[libbf] missing bf/all.hpp or build/lib/libbf.a"
  echo "[libbf] setting up under $CHAMP_DIR/libbf"

  if [ ! -d "$CHAMP_DIR/libbf/.git" ]; then
    rm -rf "$CHAMP_DIR/libbf"
    git clone https://github.com/mavam/libbf.git "$CHAMP_DIR/libbf"
  fi

  mkdir -p "$CHAMP_DIR/libbf/build"
  (
    cd "$CHAMP_DIR/libbf/build"
    cmake ..
    make clean
    make -j"${JOBS:-8}"
  )
}

pref_type () {
  case "$1" in
    no_pref|none|nopref) echo "none" ;;
    spp|spp_dev2) echo "spp_dev2" ;;
    spp_ppf|spp_ppf_dev) echo "spp_ppf_dev" ;;
    spp_ipcp|spp+ipcp) echo "spp_dev2 ipcp" ;;
    *) echo "$1" ;;
  esac
}

pref_cfg () {
  local pf="$1"
  local cfg="$CFG_DIR/${pf}.ini"
  local type types
  types="$(pref_type "$pf")"

  if [ ! -s "$cfg" ] || [ "$FORCE_REPLAY" = "1" ]; then
    : > "$cfg"
    for type in $types; do
      echo "l2c_prefetcher_types = $type" >> "$cfg"
    done
    if echo "$types" | grep -qw "spp_dev2" || echo "$types" | grep -qw "spp_ppf_dev"; then
      {
        echo "spp_dev2_fill_threshold = 90"
        echo "spp_dev2_pf_threshold = 40"
      } >> "$cfg"
    fi
  fi
  echo "$cfg"
}

patch_and_build () {
  if [ "$BUILD" != "1" ] && [ -x "$BIN" ]; then
    echo "[patch/build skip] BUILD=$BUILD existing binary: $BIN"
    echo "                   assuming this binary already supports RESIDUAL_AUDIT_LOG"
    return 0
  fi

  local patch_script="formal_NN_training/scripts/03_patch_pythia_residual_logger.sh"
  if [ ! -f "$patch_script" ]; then
    echo "[error] missing $patch_script"
    echo "        Your current repo can reuse an already-patched binary with BUILD=0."
    echo "        For a fresh rebuild, restore/add the residual logger patch script first."
    exit 1
  fi

  RESET_PATCH="$RESET_PATCH" CHAMP_DIR="$CHAMP_DIR" bash "$patch_script"

  ensure_libbf
  echo "[build] rebuilding patched Pythia multi-L2 binary"
  rm -f "$BIN"
  (
    cd "$CHAMP_DIR"
    bash ./build_champsim.sh no multi no 1
  )
  if [ ! -x "$BIN" ]; then
    echo "[error] expected binary missing after build: $BIN"
    exit 1
  fi
}

run_one () {
  local trace="$1"
  local pf="$2"
  local trfile="$TRACE_DIR/${trace}.champsimtrace.xz"
  local cfg types log event_raw event_gz
  cfg="$(pref_cfg "$pf")"
  types="$(pref_type "$pf")"
  log="$LOG_DIR/${trace}.${pf}.log"
  event_raw="$EVENT_DIR/${trace}.${pf}.events.csv"
  event_gz="$event_raw.gz"

  if [ ! -s "$trfile" ]; then
    echo "[skip missing trace] $trfile"
    return 0
  fi

  if [ "$FORCE_REPLAY" != "1" ]; then
    if [ "$COMPRESS" = "1" ] && [ -s "$event_gz" ] && [ -s "$log" ]; then
      echo "[skip existing] $event_gz"
      return 0
    fi
    if [ "$COMPRESS" != "1" ] && [ -s "$event_raw" ] && [ -s "$log" ]; then
      echo "[skip existing] $event_raw"
      return 0
    fi
  fi

  rm -f "$event_raw" "$event_gz"

  echo "============================================================"
  echo "[run residual audit] trace=$trace prefetcher=$pf types=$types"
  echo "config     : $cfg"
  echo "event csv  : $event_raw"
  echo "log        : $log"
  echo "warmup/sim : $WARMUP / $SIM"
  echo "============================================================"

  if RESIDUAL_AUDIT_LOG="$event_raw" \
    "$BIN" \
      --warmup_instructions="$WARMUP" \
      --simulation_instructions="$SIM" \
      --config="$cfg" \
      -traces "$trfile" \
      > "$log" 2>&1; then
    if [ "$COMPRESS" = "1" ] && [ -s "$event_raw" ]; then
      gzip -f "$event_raw"
    fi
    return 0
  fi

  local rc=$?
  {
    echo
    echo "RESIDUAL_RUN_FAILED $rc"
  } >> "$log"
  echo "[warn] failed residual trace=$trace prefetcher=$pf rc=$rc; continuing"
  rm -f "$event_raw" "$event_gz"
  return 0
}

wait_slot () {
  local running_ref="$1"
  local running="${!running_ref}"
  if [ "$running" -ge "$MAX_JOBS" ]; then
    wait -n || true
    running=$((running - 1))
    printf -v "$running_ref" '%s' "$running"
  fi
}

patch_and_build

cat > "$OUT_ROOT/RUN_INFO.txt" <<EOF
RUN_KIND=residual_demand_audit
ROOT=$ROOT
CHAMP_DIR=$CHAMP_DIR
BIN=$BIN
TRACE_DIR=$TRACE_DIR
TRACES=$TRACES_STR
PREFETCHERS=$PREFETCHERS_STR
WARMUP=$WARMUP
SIM=$SIM
MAX_JOBS=$MAX_JOBS
FORCE_REPLAY=$FORCE_REPLAY
COMPRESS=$COMPRESS
RESET_PATCH=$RESET_PATCH
BUILD=$BUILD
EOF

running=0
for trace in $TRACES_STR; do
  for pf in $PREFETCHERS_STR; do
    run_one "$trace" "$pf" &
    running=$((running + 1))
    wait_slot running
  done
done
wait || true

SUMMARY="$OUT_ROOT/summary.csv"
COMPRESSED_FLAG=""
if [ "$COMPRESS" = "1" ]; then
  COMPRESSED_FLAG="--compressed"
fi

python3 formal_NN_training/scripts/04_parse_residual_demand_audit.py \
  --event-root "$EVENT_DIR" \
  --out "$SUMMARY" \
  --traces "$TRACES_STR" \
  --prefetchers "$PREFETCHERS_STR" \
  $COMPRESSED_FLAG

echo
echo "[done] residual demand audit"
echo "  events : $EVENT_DIR"
echo "  logs   : $LOG_DIR"
echo "  summary: $SUMMARY"
