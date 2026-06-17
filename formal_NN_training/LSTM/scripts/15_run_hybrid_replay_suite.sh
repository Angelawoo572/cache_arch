#!/usr/bin/env bash
# One-command SPP+LSTM direct-hybrid replay suite.
#
# It creates a new run directory, prepares Colab action outputs, runs normal replay,
# timing-filter replay, optional capacity sweep, and generates CSV/SVG reports.
# Nothing is written to final_tables/, so old figures are not overwritten.

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

RUN_TAG="${RUN_TAG:-hybrid_$(date +%Y%m%d_%H%M%S)}"
SUITE_ROOT="${SUITE_ROOT:-$ROOT/formal_NN_training/results/hybrid_replay_suites/$RUN_TAG}"
TRACES_STR="${TRACES:-602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B 620.omnetpp_s-874B 623.xalancbmk_s-700B}"
CAPS_STR="${CAPS:-256K 512K 1M 2M}"
TIMING_RANGES_STR="${TIMING_RANGES:-t0_7:0:7 t1_5:1:5 t2_4:2:4 t3_3:3:3}"

WARMUP="${WARMUP:-25000000}"
SIM="${SIM:-25000000}"
MAX_JOBS="${MAX_JOBS:-2}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"
FORCE_REPLAY="${FORCE_REPLAY:-0}"
RUN_NORMAL="${RUN_NORMAL:-1}"
RUN_TIMING="${RUN_TIMING:-1}"
RUN_CAPACITY="${RUN_CAPACITY:-1}"
RUN_FIGURES="${RUN_FIGURES:-1}"
SKIP_MISSING="${SKIP_MISSING:-1}"
ALLOW_BYPASS_PREFETCH="${ALLOW_BYPASS_PREFETCH:-1}"
# Empty hybrid lists are valid: they mean the learned policy chose no prefetches for that trace/filter.
# Keep this default at 0 so a single zero-emit trace (for example mcf) does not kill an overnight suite.
STRICT_EMPTY_LIST="${STRICT_EMPTY_LIST:-0}"
MAX_PREFETCH_FRAC="${MAX_PREFETCH_FRAC:-0.10}"

TRACE_DIR="${TRACE_DIR:-$ROOT/traces}"
CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
NO_BIN="${NO_BIN:-$CHAMP_DIR/bin/champsim.baseline}"
SPP_BIN="${SPP_BIN:-$CHAMP_DIR/bin/champsim.l2_spp}"
if [ ! -x "$SPP_BIN" ] && [ -x "$CHAMP_DIR/bin/champsim.l2_spp_cand" ]; then
  SPP_BIN="$CHAMP_DIR/bin/champsim.l2_spp_cand"
fi
REPL_BIN="${REPL_BIN:-$CHAMP_DIR/bin/champsim.l2_replayer}"

REPLAY_ROOT="$SUITE_ROOT/replay_compare"
REPLAY_LOG_DIR="$REPLAY_ROOT/logs"
REPLAY_PFETCH_DIR="$REPLAY_ROOT/prefetch_lists"
CAP_ROOT="$SUITE_ROOT/capacity_sweep"
CAP_LOG_DIR="$CAP_ROOT/logs"
CAP_PFETCH_DIR="$CAP_ROOT/prefetch_lists"
TABLE_DIR="$SUITE_ROOT/tables"
FIG_DIR="$SUITE_ROOT/figures"
mkdir -p "$REPLAY_LOG_DIR" "$REPLAY_PFETCH_DIR" "$CAP_LOG_DIR" "$CAP_PFETCH_DIR" "$TABLE_DIR" "$FIG_DIR"

RUN_INFO="$SUITE_ROOT/RUN_INFO.txt"
{
  echo "RUN_TAG=$RUN_TAG"
  echo "DATE=$(date)"
  echo "ROOT=$ROOT"
  echo "TRACES=$TRACES_STR"
  echo "CAPS=$CAPS_STR"
  echo "TIMING_RANGES=$TIMING_RANGES_STR"
  echo "WARMUP=$WARMUP"
  echo "SIM=$SIM"
  echo "MAX_JOBS=$MAX_JOBS"
  echo "RUN_NORMAL=$RUN_NORMAL RUN_TIMING=$RUN_TIMING RUN_CAPACITY=$RUN_CAPACITY RUN_FIGURES=$RUN_FIGURES"
  echo "NO_BIN=$NO_BIN"
  echo "SPP_BIN=$SPP_BIN"
  echo "REPL_BIN=$REPL_BIN"
} > "$RUN_INFO"

need_exec () {
  local p="$1"
  if [ ! -x "$p" ]; then
    echo "[error] missing executable: $p"
    exit 1
  fi
}
need_exec "$NO_BIN"
need_exec "$SPP_BIN"
need_exec "$REPL_BIN"

trace_tag () { local trace="$1"; echo "${trace%%.*}"; }

run_if_needed () {
  local log="$1"; shift
  if [ "$FORCE_REPLAY" != "1" ] && [ -s "$log" ]; then
    echo "[skip existing log] $log"
    return 0
  fi
  echo "[run] $log"
  "$@" > "$log" 2>&1
}

run_replayer_if_needed () {
  local log="$1"
  local pfetch="$2"
  local trfile="$3"
  local bin="$4"
  if [ "$FORCE_REPLAY" != "1" ] && [ -s "$log" ]; then
    echo "[skip existing log] $log"
    return 0
  fi
  echo "[run] $log"
  ( export PFETCH_LIST_PATH="$pfetch"; "$bin" --warmup-instructions "$WARMUP" --simulation-instructions "$SIM" "$trfile" ) > "$log" 2>&1
}

make_prefetch_list () {
  local actions="$1"
  local out="$2"
  local tmin="${3:-}"
  local tmax="${4:-}"
  local cmd=(python3 formal_NN_training/LSTM/scripts/02_actions_to_prefetch_list.py
    --actions "$actions"
    --out "$out"
    --policy action)
  if [ "$ALLOW_BYPASS_PREFETCH" = "1" ]; then
    cmd+=(--allow-bypass-prefetch)
  fi
  if [ -n "$tmin" ] && [ -n "$tmax" ]; then
    cmd+=(--timing-min-bin "$tmin" --timing-max-bin "$tmax")
  fi
  "${cmd[@]}"
  local lines rows frac_ok
  lines=$(wc -l < "$out" || echo 0)
  rows=$(wc -l < "$actions" || echo 0)
  if [ "$rows" -gt 0 ]; then
    rows=$((rows - 1))
  fi
  echo "[prefetch list] $out lines=$lines action_rows=$rows max_frac=$MAX_PREFETCH_FRAC"
  if [ "$lines" -eq 0 ]; then
    if [ "$STRICT_EMPTY_LIST" = "1" ]; then
      echo "[error] empty prefetch list: $out"
      exit 1
    fi
    echo "[warn] empty prefetch list; replay will be equivalent to no learned prefetches for this method"
  fi
  frac_ok=$(python3 - <<PY_FRAC
lines = float("$lines")
rows = max(float("$rows"), 1.0)
max_frac = float("$MAX_PREFETCH_FRAC")
print(1 if (lines / rows) <= max_frac else 0)
PY_FRAC
)
  if [ "$frac_ok" != "1" ]; then
    echo "[error] prefetch list is too large: lines=$lines rows=$rows max_frac=$MAX_PREFETCH_FRAC"
    echo "[hint] This usually means the Colab export was unbudgeted. Rerun guarded R10/R13 or the budget hotfix before replay."
    exit 1
  fi
}

prepare_trace () {
  local trace="$1"
  local tag
  tag="$(trace_tag "$trace")"
  local trfile="$TRACE_DIR/${trace}.champsimtrace.xz"
  local actions="$ROOT/formal_NN_training/results/LSTM/draft/artifacts/by_trace/${trace}/full_lstm_cache_actions.csv"
  local packed_dir="$ROOT/formal_NN_training/results/LSTM/draft/artifacts/packed/$tag"

  if [ ! -s "$trfile" ]; then
    echo "[skip missing trace file] $trace -> $trfile"
    return 1
  fi

  if [ "$FORCE_PREPARE" = "1" ]; then
    python3 formal_NN_training/LSTM/scripts/07_prepare_actions_for_replay.py --trace "$trace" --restore-packed
  elif [ -s "$actions" ]; then
    python3 formal_NN_training/LSTM/scripts/07_prepare_actions_for_replay.py --trace "$trace"
  elif ls "$packed_dir"/full_lstm_cache_actions.csv.gz.part_* >/dev/null 2>&1 || [ -s "$packed_dir/full_lstm_cache_actions.csv.gz" ]; then
    python3 formal_NN_training/LSTM/scripts/07_prepare_actions_for_replay.py --trace "$trace" --restore-packed
  else
    echo "[skip missing actions] $trace -> $actions or $packed_dir/full_lstm_cache_actions.csv.gz.part_*"
    return 1
  fi

  if [ ! -s "$actions" ]; then
    echo "[skip no prepared action csv] $trace -> $actions"
    return 1
  fi
  echo "$trace" >> "$SUITE_ROOT/.prepared_traces.tmp"
  return 0
}

run_normal_and_timing_one_trace () {
  local trace="$1"
  local trfile="$TRACE_DIR/${trace}.champsimtrace.xz"
  local actions="$ROOT/formal_NN_training/results/LSTM/draft/artifacts/by_trace/${trace}/full_lstm_cache_actions.csv"
  local main_list="$REPLAY_PFETCH_DIR/prefetch_list_${trace}.hybrid_action.txt"

  echo "============================================================"
  echo "[normal/timing replay] $trace"
  echo "============================================================"

  if [ "$RUN_NORMAL" = "1" ]; then
    run_if_needed "$REPLAY_LOG_DIR/${trace}.no_prefetch.log" \
      "$NO_BIN" --warmup-instructions "$WARMUP" --simulation-instructions "$SIM" "$trfile"
    run_if_needed "$REPLAY_LOG_DIR/${trace}.spp.log" \
      "$SPP_BIN" --warmup-instructions "$WARMUP" --simulation-instructions "$SIM" "$trfile"
    make_prefetch_list "$actions" "$main_list"
    run_replayer_if_needed "$REPLAY_LOG_DIR/${trace}.LSTM_hybrid_action.log" "$main_list" "$trfile" "$REPL_BIN"
  fi

  if [ "$RUN_TIMING" = "1" ]; then
    for spec in $TIMING_RANGES_STR; do
      local name="${spec%%:*}"
      local rest="${spec#*:}"
      local tmin="${rest%%:*}"
      local tmax="${rest##*:}"
      local pfetch="$REPLAY_PFETCH_DIR/prefetch_list_${trace}.hybrid_action_${name}.txt"
      make_prefetch_list "$actions" "$pfetch" "$tmin" "$tmax"
      run_replayer_if_needed "$REPLAY_LOG_DIR/${trace}.LSTM_hybrid_action_${name}.log" "$pfetch" "$trfile" "$REPL_BIN"
    done
  fi
}

cap_bin () {
  local kind="$1"
  local cap="$2"
  case "$kind" in
    no) echo "$CHAMP_DIR/bin/champsim.baseline.L2_${cap}" ;;
    spp) echo "$CHAMP_DIR/bin/champsim.spp.L2_${cap}" ;;
    repl) echo "$CHAMP_DIR/bin/champsim.replayer.L2_${cap}" ;;
  esac
}

run_capacity_one () {
  local trace="$1"
  local cap="$2"
  local trfile="$TRACE_DIR/${trace}.champsimtrace.xz"
  local actions="$ROOT/formal_NN_training/results/LSTM/draft/artifacts/by_trace/${trace}/full_lstm_cache_actions.csv"
  local pfetch="$CAP_PFETCH_DIR/prefetch_list_${trace}.L2_${cap}.hybrid_action.txt"
  local no_bin spp_bin repl_bin
  no_bin="$(cap_bin no "$cap")"
  spp_bin="$(cap_bin spp "$cap")"
  repl_bin="$(cap_bin repl "$cap")"

  if [ ! -x "$no_bin" ] || [ ! -x "$spp_bin" ] || [ ! -x "$repl_bin" ]; then
    echo "[skip capacity missing binaries] trace=$trace cap=$cap"
    echo "  no=$no_bin"
    echo "  spp=$spp_bin"
    echo "  repl=$repl_bin"
    return 0
  fi

  echo "============================================================"
  echo "[capacity replay] trace=$trace cap=$cap"
  echo "============================================================"
  make_prefetch_list "$actions" "$pfetch"
  run_if_needed "$CAP_LOG_DIR/${trace}.L2_${cap}.no_prefetch.log" \
    "$no_bin" --warmup-instructions "$WARMUP" --simulation-instructions "$SIM" "$trfile"
  run_if_needed "$CAP_LOG_DIR/${trace}.L2_${cap}.spp.log" \
    "$spp_bin" --warmup-instructions "$WARMUP" --simulation-instructions "$SIM" "$trfile"
  run_replayer_if_needed "$CAP_LOG_DIR/${trace}.L2_${cap}.hybrid_action.log" "$pfetch" "$trfile" "$repl_bin"
}

wait_slot () {
  local running_ref="$1"
  # shellcheck disable=SC2163
  local running="${!running_ref}"
  if [ "$running" -ge "$MAX_JOBS" ]; then
    wait -n
    running=$((running - 1))
    printf -v "$running_ref" '%s' "$running"
  fi
}

rm -f "$SUITE_ROOT/.prepared_traces.tmp"
touch "$SUITE_ROOT/.prepared_traces.tmp"

echo "============================================================"
echo "[suite] RUN_TAG=$RUN_TAG"
echo "[suite] SUITE_ROOT=$SUITE_ROOT"
echo "[suite] preparing traces"
echo "============================================================"
for trace in $TRACES_STR; do
  if ! prepare_trace "$trace"; then
    if [ "$SKIP_MISSING" != "1" ]; then
      echo "[error] trace not ready and SKIP_MISSING=0: $trace"
      exit 1
    fi
  fi
done

mapfile -t PREPARED_TRACES < "$SUITE_ROOT/.prepared_traces.tmp"
if [ "${#PREPARED_TRACES[@]}" -eq 0 ]; then
  echo "[error] no traces prepared; nothing to run"
  exit 1
fi
printf '%s\n' "${PREPARED_TRACES[@]}" > "$SUITE_ROOT/prepared_traces.txt"
echo "[prepared traces] ${PREPARED_TRACES[*]}"

running=0
if [ "$RUN_NORMAL" = "1" ] || [ "$RUN_TIMING" = "1" ]; then
  for trace in "${PREPARED_TRACES[@]}"; do
    run_normal_and_timing_one_trace "$trace" &
    running=$((running + 1))
    wait_slot running
  done
  wait
fi

if [ "$RUN_CAPACITY" = "1" ]; then
  running=0
  for trace in "${PREPARED_TRACES[@]}"; do
    for cap in $CAPS_STR; do
      run_capacity_one "$trace" "$cap" &
      running=$((running + 1))
      wait_slot running
    done
  done
  wait
fi

if [ "$RUN_FIGURES" = "1" ]; then
  python3 formal_NN_training/LSTM/scripts/16_make_hybrid_replay_figures.py \
    --suite-root "$SUITE_ROOT" \
    --suite-tag "$RUN_TAG" \
    --traces "${PREPARED_TRACES[*]}" \
    --caps "$CAPS_STR"
fi

cat <<EOF
============================================================
[done] hybrid replay suite finished
RUN_TAG=$RUN_TAG
SUITE_ROOT=$SUITE_ROOT
Tables:  $SUITE_ROOT/tables
Figures: $SUITE_ROOT/figures
Logs:    $SUITE_ROOT/replay_compare/logs
Capacity logs: $SUITE_ROOT/capacity_sweep/logs
============================================================
EOF
