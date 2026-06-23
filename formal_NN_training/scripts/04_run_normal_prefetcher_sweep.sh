#!/usr/bin/env bash
# Canonical normal-prefetcher baseline sweep.
#
# These runs are comparison points only. Their outputs never become standalone
# NN runtime inputs, training labels, or policy-calibration objectives.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TRACES="${TRACES:-602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B 620.omnetpp_s-874B 623.xalancbmk_s-700B}"
PREFETCHERS="${PREFETCHERS:-no_pref stride streamer ampm spp ipcp sms sandbox power7}"
WARMUP="${WARMUP:-25000000}"
SIM="${SIM:-25000000}"
MAX_JOBS="${MAX_JOBS:-4}"
BUILD="${BUILD:-0}"
FORCE="${FORCE:-${FORCE_REPLAY:-0}}"
JOBS="${JOBS:-8}"

CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
TRACE_DIR="${TRACE_DIR:-$ROOT/traces}"
OUT_ROOT="${OUT_ROOT:-$ROOT/formal_NN_training/results/prefetcher_baselines}"
LOG_DIR="$OUT_ROOT/logs"
CFG_DIR="$OUT_ROOT/configs"
BIN="${BIN:-$CHAMP_DIR/bin/perceptron-no-multi-no-ship-1core}"
PARSER="$ROOT/formal_NN_training/scripts/01_parse_prefetch_behavior_audit.py"

mkdir -p "$LOG_DIR" "$CFG_DIR"
[[ -d "$CHAMP_DIR" ]] || { echo "[error] missing ChampSim checkout: $CHAMP_DIR" >&2; exit 2; }
[[ -f "$PARSER" ]] || { echo "[error] missing parser: $PARSER" >&2; exit 2; }

ensure_libbf() {
  if [[ -f "$CHAMP_DIR/libbf/bf/all.hpp" && -f "$CHAMP_DIR/libbf/build/lib/libbf.a" ]]; then
    return 0
  fi
  echo "[libbf] missing; building local dependency"
  if [[ ! -d "$CHAMP_DIR/libbf/.git" ]]; then
    rm -rf "$CHAMP_DIR/libbf"
    git clone https://github.com/mavam/libbf.git "$CHAMP_DIR/libbf"
  fi
  mkdir -p "$CHAMP_DIR/libbf/build"
  (
    cd "$CHAMP_DIR/libbf/build"
    cmake ..
    make -j"$JOBS"
  )
  [[ -f "$CHAMP_DIR/libbf/bf/all.hpp" && -f "$CHAMP_DIR/libbf/build/lib/libbf.a" ]] || {
    echo "[error] libbf setup failed" >&2
    exit 2
  }
}

if [[ ! -x "$BIN" ]]; then
  [[ "$BUILD" == "1" ]] || { echo "[error] missing $BIN; rerun with BUILD=1" >&2; exit 2; }
  ensure_libbf
  ( cd "$CHAMP_DIR" && bash ./build_champsim.sh no multi no 1 )
fi
[[ -x "$BIN" ]] || { echo "[error] missing baseline binary after build: $BIN" >&2; exit 2; }

pref_type() {
  case "$1" in
    no_pref|none|nopref) echo "none" ;;
    spp|spp_dev2) echo "spp_dev2" ;;
    spp_ppf|spp_ppf_dev) echo "spp_ppf_dev" ;;
    *) echo "$1" ;;
  esac
}

write_cfg() {
  local pf="$1" type="$2" cfg="$3"
  {
    echo "l2c_prefetcher_types = $type"
    if [[ "$type" == "spp_dev2" || "$type" == "spp_ppf_dev" ]]; then
      echo "spp_dev2_fill_threshold = 90"
      echo "spp_dev2_pf_threshold = 40"
    fi
  } > "$cfg"
}

run_one() {
  local trace="$1" pf="$2"
  local type cfg log trace_file
  type="$(pref_type "$pf")"
  trace_file="$TRACE_DIR/${trace}.champsimtrace.xz"
  cfg="$CFG_DIR/${pf}.ini"
  log="$LOG_DIR/${trace}.${pf}.log"

  [[ -s "$trace_file" ]] || { echo "[error] missing trace: $trace_file" >&2; return 1; }
  if [[ "$FORCE" != "1" && -s "$log" ]]; then
    echo "[skip] $trace $pf"
    return 0
  fi
  write_cfg "$pf" "$type" "$cfg"
  echo "[run] trace=$trace prefetcher=$pf"
  "$BIN" --warmup_instructions="$WARMUP" --simulation_instructions="$SIM" --config="$cfg" -traces "$trace_file" > "$log" 2>&1
}

cat > "$OUT_ROOT/RUN_INFO.txt" <<EOF
RUN_KIND=normal_prefetcher_baseline
ROOT=$ROOT
CHAMP_DIR=$CHAMP_DIR
BIN=$BIN
TRACE_DIR=$TRACE_DIR
TRACES=$TRACES
PREFETCHERS=$PREFETCHERS
WARMUP=$WARMUP
SIM=$SIM
MAX_JOBS=$MAX_JOBS
FORCE=$FORCE
EOF

running=0
status=0
for trace in $TRACES; do
  for pf in $PREFETCHERS; do
    run_one "$trace" "$pf" &
    running=$((running + 1))
    if (( running >= MAX_JOBS )); then
      wait -n || status=1
      running=$((running - 1))
    fi
  done
done
while (( running > 0 )); do
  wait -n || status=1
  running=$((running - 1))
done
(( status == 0 )) || exit "$status"

python3 "$PARSER" --log-root "$LOG_DIR" --out "$OUT_ROOT/summary.csv" --traces "$TRACES" --prefetchers "$PREFETCHERS" --nodup

echo "[done] $OUT_ROOT/summary.csv"