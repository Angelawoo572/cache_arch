#!/usr/bin/env bash
# Re-run normal and frozen standalone policies with per-event L2C logging.
# Analysis-only: normal outcomes never become standalone-NN labels or inputs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TRACES="${TRACES:-602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B 620.omnetpp_s-874B 623.xalancbmk_s-700B}"
NORMAL_PREFETCHERS="${NORMAL_PREFETCHERS:-no_pref stride streamer ampm spp ipcp sms sandbox power7}"
# Semicolon-separated LABEL=ARTIFACT_DIRECTORY entries.
NN_VARIANTS="${NN_VARIANTS:-v3_1=formal_NN_training/artifacts/standalone_multihorizon_lstm_v3_1;v3_3=formal_NN_training/artifacts/standalone_multihorizon_lstm_v3_3_context_coverage}"
MODE="${MODE:-both}" # normal, lstm, or both
WARMUP="${WARMUP:-25000000}"
SIM="${SIM:-25000000}"
MAX_JOBS="${MAX_JOBS:-2}"
FORCE="${FORCE:-0}"
BUILD="${BUILD:-1}"
RESET_PATCH="${RESET_PATCH:-0}"
CHUNK_LEN="${CHUNK_LEN:-1024}"
DEDUP_CAPACITY="${DEDUP_CAPACITY:-256}"
EXPORT_SUFFIX="${EXPORT_SUFFIX:-pure_balanced_lru${DEDUP_CAPACITY}}"

CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
TRACE_DIR="${TRACE_DIR:-$ROOT/traces}"
ORACLE_DIR="${ORACLE_DIR:-$ROOT/formal_NN_training/results/standalone_nn_data/oracle}"
RUN_TAG="${RUN_TAG:-event_audit_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/formal_NN_training/results/prefetch_explainability/$RUN_TAG}"
PATCH="$ROOT/formal_NN_training/scripts/02_patch_pythia_demand_logger.sh"
PREP="$ROOT/formal_NN_training/scripts/07_prepare_keyed_replay_input.py"
REPLAYER_BUILD="$ROOT/formal_NN_training/scripts/06_install_keyed_listreplayer.sh"
NORMAL_PARSER="$ROOT/formal_NN_training/scripts/01_parse_prefetch_behavior_audit.py"
NORMAL_BIN="${NORMAL_BIN:-$CHAMP_DIR/bin/perceptron-no-multi-no-ship-1core}"
REPLAY_BIN="${REPLAY_BIN:-$CHAMP_DIR/bin/champsim.standalone_nn_replayer}"

mkdir -p "$OUT_ROOT/normal/events" "$OUT_ROOT/normal/logs" "$OUT_ROOT/normal/configs" \
         "$OUT_ROOT/lstm" "$OUT_ROOT/replay_inputs"

pref_type() {
  case "$1" in
    no_pref|none|nopref) echo none ;;
    spp|spp_dev2) echo spp_dev2 ;;
    *) echo "$1" ;;
  esac
}

write_cfg() {
  local type="$1" cfg="$2"
  {
    echo "l2c_prefetcher_types = $type"
    if [[ "$type" == spp_dev2 ]]; then
      echo "spp_dev2_fill_threshold = 90"
      echo "spp_dev2_pf_threshold = 40"
    fi
  } > "$cfg"
}

build_all() {
  [[ "$BUILD" == 1 ]] || return 0
  RESET_PATCH="$RESET_PATCH" CHAMP_DIR="$CHAMP_DIR" bash "$PATCH"
  ( cd "$CHAMP_DIR" && bash ./build_champsim.sh no multi no 1 )
  CHAMP_DIR="$CHAMP_DIR" bash "$REPLAYER_BUILD"
}

run_normal() {
  local trace="$1" pf="$2"
  local raw="$OUT_ROOT/normal/events/$trace.$pf.events.csv"
  local out="$raw.gz"
  local log="$OUT_ROOT/normal/logs/$trace.$pf.log"
  local cfg="$OUT_ROOT/normal/configs/$pf.ini"
  local trace_file="$TRACE_DIR/$trace.champsimtrace.xz"
  [[ -s "$trace_file" ]] || { echo "[error] missing $trace_file" >&2; return 1; }
  [[ "$FORCE" == 1 || ! -s "$out" || ! -s "$log" ]] || { echo "[skip normal] $trace $pf"; return 0; }
  write_cfg "$(pref_type "$pf")" "$cfg"
  echo "[normal] $trace $pf"
  DEMAND_EVENT_LOG="$raw" "$NORMAL_BIN" \
    --warmup_instructions="$WARMUP" --simulation_instructions="$SIM" \
    --config="$cfg" -traces "$trace_file" > "$log" 2>&1
  [[ -s "$raw" ]] && grep -Fq Core_0_IPC "$log" || { echo "[error] normal run failed: $trace $pf" >&2; return 1; }
  gzip -f "$raw"
}

run_lstm() {
  local trace="$1" label="$2" art_dir="$3"
  local variant_root="$OUT_ROOT/lstm/$label"
  local trace_file="$TRACE_DIR/$trace.champsimtrace.xz"
  local rich="$art_dir/prefetch_list_${trace}_cl${CHUNK_LEN}_${EXPORT_SUFFIX}.csv"
  local oracle="$ORACLE_DIR/$trace.oracle.csv.gz"
  local keyed="$OUT_ROOT/replay_inputs/$label/$trace.pc_line_occ.csv"
  local raw="$variant_root/events/$trace.events.csv"
  local out="$raw.gz"
  local log="$variant_root/logs/$trace.standalone_lstm.log"
  mkdir -p "$variant_root/events" "$variant_root/logs" "$(dirname "$keyed")"
  [[ -s "$trace_file" && -s "$rich" && -s "$oracle" ]] || {
    echo "[error] missing trace, rich export, or oracle for $label/$trace" >&2; return 1; }
  [[ "$FORCE" == 1 || ! -s "$out" || ! -s "$log" || ! -s "$keyed" ]] || { echo "[skip lstm] $label $trace"; return 0; }
  python3 "$PREP" --rich-list "$rich" --oracle "$oracle" --out "$keyed" > "$variant_root/logs/$trace.prepare.log" 2>&1
  echo "[lstm] $label $trace"
  PFETCH_LIST_PATH="$keyed" DEMAND_EVENT_LOG="$raw" "$REPLAY_BIN" \
    --l2c_prefetcher_types=list_replayer \
    --warmup_instructions="$WARMUP" --simulation_instructions="$SIM" \
    -traces "$trace_file" > "$log" 2>&1
  [[ -s "$raw" ]] && grep -Fq 'key=pc_line_occ' "$log" || { echo "[error] replay failed: $label $trace" >&2; return 1; }
  gzip -f "$raw"
}

build_all
[[ -x "$NORMAL_BIN" && -x "$REPLAY_BIN" ]] || { echo "[error] expected binaries missing" >&2; exit 2; }
[[ -f "$NORMAL_PARSER" ]] || { echo "[error] missing normal parser: $NORMAL_PARSER" >&2; exit 2; }

running=0
status=0
launch() {
  "$@" &
  running=$((running + 1))
  if (( running >= MAX_JOBS )); then
    wait -n || status=1
    running=$((running - 1))
  fi
}

if [[ "$MODE" == normal || "$MODE" == both ]]; then
  for trace in $TRACES; do
    for pf in $NORMAL_PREFETCHERS; do
      launch run_normal "$trace" "$pf"
    done
  done
fi
if [[ "$MODE" == lstm || "$MODE" == both ]]; then
  IFS=';' read -r -a variants <<< "$NN_VARIANTS"
  for spec in "${variants[@]}"; do
    [[ -n "$spec" ]] || continue
    label="${spec%%=*}"
    art_dir="${spec#*=}"
    [[ "$label" != "$art_dir" ]] || { echo "[error] NN_VARIANTS entry must be label=dir" >&2; exit 2; }
    for trace in $TRACES; do
      launch run_lstm "$trace" "$label" "$art_dir"
    done
  done
fi
while (( running > 0 )); do
  wait -n || status=1
  running=$((running - 1))
done
(( status == 0 )) || exit "$status"

if [[ "$MODE" == normal || "$MODE" == both ]]; then
  python3 "$NORMAL_PARSER" \
    --log-root "$OUT_ROOT/normal/logs" \
    --out "$OUT_ROOT/normal/summary.csv" \
    --traces "$TRACES" \
    --prefetchers "$NORMAL_PREFETCHERS" \
    --nodup
fi

cat > "$OUT_ROOT/RUN_INFO.txt" <<EOF
RUN_KIND=prefetch_event_explainability
TRACES=$TRACES
NORMAL_PREFETCHERS=$NORMAL_PREFETCHERS
NN_VARIANTS=$NN_VARIANTS
WARMUP=$WARMUP
SIM=$SIM
CHUNK_LEN=$CHUNK_LEN
EXPORT_SUFFIX=$EXPORT_SUFFIX
NORMAL_BIN=$NORMAL_BIN
REPLAY_BIN=$REPLAY_BIN
NORMAL_SUMMARY=$OUT_ROOT/normal/summary.csv
EOF
echo "[done] $OUT_ROOT"
