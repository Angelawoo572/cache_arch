#!/usr/bin/env bash
# Run a broad single-base-prefetcher behavior sweep on Pythia.
#
# Purpose:
#   Pick a better "normal prefetcher" baseline before changing the LSTM labels/features.
#   This is counter-level only: IPC, miss reduction, accuracy, timeliness, duplicate proxy.
#
# Default safe zoo:
#   no_pref next_line stride streamer ampm bop spp ipcp sms bingo mlop sandbox scooby dspatch power7
#
# Example:
#   cd ~/cache
#   git pull
#   TRACES="602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B 620.omnetpp_s-874B 623.xalancbmk_s-700B" \
#   WARMUP=25000000 SIM=25000000 MAX_JOBS=6 BUILD=0 FORCE_REPLAY=0 \
#     bash formal_NN_training/scripts/06_run_base_prefetcher_zoo_audit.sh
#
# Output:
#   formal_NN_training/results/base_prefetcher_zoo/logs/*.log
#   formal_NN_training/results/base_prefetcher_zoo/summary_nodup.csv
#   formal_NN_training/results/base_prefetcher_zoo/RUN_INFO.txt

set -euo pipefail

if git rev-parse --show-toplevel >/dev/null 2>&1; then
  ROOT="$(git rev-parse --show-toplevel)"
else
  ROOT="${ROOT:-$HOME/cache}"
fi
cd "$ROOT"

DEFAULT_TRACES="602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B 620.omnetpp_s-874B 623.xalancbmk_s-700B"
DEFAULT_PREFETCHERS="no_pref next_line stride streamer ampm bop spp ipcp sms bingo mlop sandbox scooby dspatch power7"

TRACES_STR="${TRACES:-$DEFAULT_TRACES}"
PREFETCHERS_STR="${PREFETCHERS:-$DEFAULT_PREFETCHERS}"

WARMUP="${WARMUP:-25000000}"
SIM="${SIM:-25000000}"
MAX_JOBS="${MAX_JOBS:-4}"
FORCE_REPLAY="${FORCE_REPLAY:-0}"
BUILD="${BUILD:-1}"
NODUP="${NODUP:-1}"

TRACE_DIR="${TRACE_DIR:-$ROOT/traces}"
CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
OUT_ROOT="${OUT_ROOT:-$ROOT/formal_NN_training/results/base_prefetcher_zoo}"
LOG_DIR="$OUT_ROOT/logs"
CFG_DIR="$ROOT/formal_NN_training/_cfg/pythia_prefetcher_zoo"
BIN="${BIN:-$CHAMP_DIR/bin/perceptron-no-multi-no-ship-1core}"

mkdir -p "$LOG_DIR" "$CFG_DIR"

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

  if [ ! -f "$CHAMP_DIR/libbf/bf/all.hpp" ] || [ ! -f "$CHAMP_DIR/libbf/build/lib/libbf.a" ]; then
    echo "[error] libbf setup failed"
    exit 1
  fi
}

build_if_needed () {
  if [ "$BUILD" != "1" ] && [ -x "$BIN" ]; then
    echo "[build skip] BUILD=$BUILD existing binary: $BIN"
    return 0
  fi
  if [ -x "$BIN" ]; then
    echo "[build skip] existing binary: $BIN"
    return 0
  fi
  ensure_libbf
  echo "[build] Pythia multi-L2 binary"
  echo "        $CHAMP_DIR/build_champsim.sh no multi no 1"
  (
    cd "$CHAMP_DIR"
    bash ./build_champsim.sh no multi no 1
  )
  if [ ! -x "$BIN" ]; then
    echo "[error] expected binary missing after build: $BIN"
    exit 1
  fi
}

pref_type () {
  case "$1" in
    no_pref|none|nopref) echo "none" ;;
    spp|spp_dev2) echo "spp_dev2" ;;
    spp_ppf|spp_ppf_dev) echo "spp_ppf_dev" ;;
    *) echo "$1" ;;
  esac
}

pref_cfg () {
  local pf="$1"
  local type
  type="$(pref_type "$pf")"
  local cfg="$CFG_DIR/${pf}.ini"

  if [ ! -s "$cfg" ] || [ "$FORCE_REPLAY" = "1" ]; then
    {
      echo "l2c_prefetcher_types = $type"
      if [ "$type" = "spp_dev2" ] || [ "$type" = "spp_ppf_dev" ]; then
        echo "spp_dev2_fill_threshold = 90"
        echo "spp_dev2_pf_threshold = 40"
      fi
    } > "$cfg"
  fi
  echo "$cfg"
}

run_one () {
  local trace="$1"
  local pf="$2"
  local trfile="$TRACE_DIR/${trace}.champsimtrace.xz"
  local log="$LOG_DIR/${trace}.${pf}.log"
  local cfg type
  cfg="$(pref_cfg "$pf")"
  type="$(pref_type "$pf")"

  if [ ! -s "$trfile" ]; then
    echo "[skip missing trace] $trfile"
    return 0
  fi

  if [ "$FORCE_REPLAY" != "1" ] && [ -s "$log" ]; then
    echo "[skip existing] $log"
    return 0
  fi

  echo "============================================================"
  echo "[run zoo] trace=$trace prefetcher=$pf type=$type"
  echo "config     : $cfg"
  echo "log        : $log"
  echo "warmup/sim : $WARMUP / $SIM"
  echo "============================================================"

  if "$BIN" \
      --warmup_instructions="$WARMUP" \
      --simulation_instructions="$SIM" \
      --config="$cfg" \
      -traces "$trfile" \
      > "$log" 2>&1; then
    return 0
  fi

  local rc=$?
  {
    echo
    echo "ZOO_RUN_FAILED $rc"
  } >> "$log"
  echo "[warn] failed trace=$trace prefetcher=$pf rc=$rc; continuing"
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

build_if_needed

cat > "$OUT_ROOT/RUN_INFO.txt" <<EOF
RUN_KIND=base_prefetcher_zoo_audit
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
if [ "$NODUP" = "1" ]; then
  SUMMARY="$OUT_ROOT/summary_nodup.csv"
  NODUP_FLAG="--nodup"
else
  NODUP_FLAG=""
fi

python3 formal_NN_training/scripts/01_parse_prefetch_behavior_audit.py \
  --log-root "$LOG_DIR" \
  --out "$SUMMARY" \
  --traces "$TRACES_STR" \
  --prefetchers "$PREFETCHERS_STR" \
  $NODUP_FLAG

echo
echo "[done] base prefetcher zoo audit"
echo "  logs   : $LOG_DIR"
echo "  summary: $SUMMARY"
