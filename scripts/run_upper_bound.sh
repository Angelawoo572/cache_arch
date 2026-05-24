#!/usr/bin/env bash
# run_upper_bound.sh -- v3 with live progress
#
# 4 configs x 5 traces = 20 runs. Each ~1-3 min. Total ~40-60 minutes wall clock.
# Builds 4 binaries first (~3 min total), then runs all configs.

set -uo pipefail

WARMUP="${WARMUP:-1000000}"
SIM="${SIM:-5000000}"

WORKDIR="$(pwd)"
CHAMP_DIR="$WORKDIR/external/ChampSim"
TRACE_DIR="$WORKDIR/traces"
OUT_DIR="$WORKDIR/results"
LOG_DIR="$WORKDIR/results/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

if [ ! -d "$CHAMP_DIR" ]; then
  echo "[error] ChampSim dir not found at $CHAMP_DIR. Run setup_champsim.sh first."; exit 1
fi

declare -a TRACES=(
  "619.lbm_s-4268B"
  "605.mcf_s-994B"
  "620.omnetpp_s-874B"
  "602.gcc_s-734B"
  "623.xalancbmk_s-700B"
)

mkdir -p "$CHAMP_DIR/_cfg"

cat > "$CHAMP_DIR/_cfg/cfg_lru.json" <<'JSON'
{ "LLC": { "replacement": "lru" } }
JSON

cat > "$CHAMP_DIR/_cfg/cfg_lru_ipstride.json" <<'JSON'
{
  "ooo_cpu": [{ "L1D": { "prefetcher": "ip_stride" } }],
  "LLC":     { "replacement": "lru" }
}
JSON

cat > "$CHAMP_DIR/_cfg/cfg_lru_spp.json" <<'JSON'
{
  "L2C": { "prefetcher": "spp_dev" },
  "LLC": { "replacement": "lru" }
}
JSON

cat > "$CHAMP_DIR/_cfg/cfg_srrip_spp.json" <<'JSON'
{
  "L2C": { "prefetcher": "spp_dev" },
  "LLC": { "replacement": "srrip" }
}
JSON

build_for_cfg () {
  local tag=$1
  local cfg_path=$2
  echo
  echo "[build] tag=$tag  cfg=$cfg_path"
  cd "$CHAMP_DIR"
  ./config.sh "$cfg_path"  > /dev/null
  make -j8                 > /dev/null 2>&1
  if [ ! -x bin/champsim ]; then
    echo "[error] build failed for $tag"; return 1
  fi
  cp bin/champsim "bin/champsim.${tag}"
  echo "[build] OK -> bin/champsim.${tag}"
  return 0
}

run_with_heartbeat () {
  local cmd_log="$1"
  local trace_name="$2"
  shift 2
  "$@" > "$cmd_log" 2>&1 &
  local pid=$!
  local seconds=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    seconds=$((seconds + 30))
    local last=$(tail -1 "$cmd_log" 2>/dev/null | head -c 200)
    printf "  ...still running %s (elapsed %ds)  last: %s\n" \
           "$trace_name" "$seconds" "${last:0:80}"
  done
  wait "$pid"
  return $?
}

CSV="$OUT_DIR/upper_bound.csv"
echo "trace,config,IPC,LLC_MPKI" > "$CSV"

run_for_cfg () {
  local tag=$1
  local cfg_idx=$2
  local cfg_total=$3
  for t in "${TRACES[@]}"; do
    TR_FILE="$TRACE_DIR/${t}.champsimtrace.xz"
    if [ ! -f "$TR_FILE" ]; then
      echo "[skip] $t (no trace)"; continue
    fi

    LOG="$LOG_DIR/${t}.${tag}.log"
    ELAPSED=$(( $(date +%s) - T0 ))
    echo "[run cfg ${cfg_idx}/${cfg_total}] $t  tag=$tag  (total elapsed: ${ELAPSED}s)"

    if ! run_with_heartbeat "$LOG" "${t}+${tag}" \
          "$CHAMP_DIR/bin/champsim.${tag}" \
          --warmup-instructions "$WARMUP" \
          --simulation-instructions "$SIM" \
          "$TR_FILE"; then
      echo "  [FAIL] $t $tag -- see $LOG"
      echo "$t,$tag,FAIL,FAIL" >> "$CSV"
      continue
    fi

    IPC=$(grep -E "cumulative IPC" "$LOG" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
    LLC_MPKI=$(grep -E "LLC.*MPKI" "$LOG" | head -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
    if [ -z "$LLC_MPKI" ]; then
      MISS=$(grep -E "cpu0->LLC TOTAL" "$LOG" | head -1 | grep -oE "MISS:[ ]+[0-9]+" | grep -oE "[0-9]+")
      INSTR=$(grep "cumulative IPC" "$LOG" | tail -1 | grep -oE "instructions: [0-9]+" | grep -oE "[0-9]+")
      if [ -n "$MISS" ] && [ -n "$INSTR" ] && [ "$INSTR" -gt 0 ]; then
        LLC_MPKI=$(python3 -c "print(f'{$MISS*1000/$INSTR:.3f}')")
      fi
    fi
    IPC="${IPC:-NA}"; LLC_MPKI="${LLC_MPKI:-NA}"
    echo "$t,$tag,$IPC,$LLC_MPKI" >> "$CSV"
    echo "  [done] IPC=$IPC  LLC_MPKI=$LLC_MPKI"
  done
}

T0=$(date +%s)

echo "============================================"
echo "Phase 1/2: build 4 binaries (~3 minutes total)"
echo "============================================"
build_for_cfg "lru"           "_cfg/cfg_lru.json"
build_for_cfg "lru_ipstride"  "_cfg/cfg_lru_ipstride.json"
build_for_cfg "lru_spp"       "_cfg/cfg_lru_spp.json"
build_for_cfg "srrip_spp"     "_cfg/cfg_srrip_spp.json"

echo
echo "============================================"
echo "Phase 2/2: run 4 configs x 5 traces = 20 runs"
echo "Estimated wall clock: 40-60 minutes."
echo "Each run prints a heartbeat every 30s so you can"
echo "verify it's making progress. DO NOT Ctrl+C unless"
echo "you see no heartbeat for several minutes."
echo "============================================"
run_for_cfg "lru"           1 4
run_for_cfg "lru_ipstride"  2 4
run_for_cfg "lru_spp"       3 4
run_for_cfg "srrip_spp"     4 4

TOTAL=$(( $(date +%s) - T0 ))
echo
echo "[ALL DONE] total wall-clock: ${TOTAL}s ($(( TOTAL / 60 )) min)"
echo "[csv] $CSV"
column -t -s, "$CSV" 2>/dev/null || cat "$CSV"

cat <<'NOTE'

[note] These configurations use only ChampSim built-in policies.
       The chart shows "strongest built-in combination", not literal Belady-OPT.
       For literal OPT or oracle prefetch, would need an external module.
NOTE