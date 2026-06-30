#!/usr/bin/env bash
# Re-run normal and standalone policies with per-event L2C logging.
# Analysis-only: normal outcomes never become standalone-NN labels or inputs.
#
# Legacy NN_VARIANTS mode consumes one artifact directory per label.  Plan mode
# (REPLAY_PLAN=...) consumes every current-run v3.9 list directly from its plan.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TRACES="${TRACES:-602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B 620.omnetpp_s-874B 623.xalancbmk_s-700B}"
NORMAL_PREFETCHERS="${NORMAL_PREFETCHERS:-no_pref stride streamer ampm spp ipcp sms sandbox power7}"
# Semicolon-separated LABEL=ARTIFACT_DIRECTORY entries; legacy mode only.
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

# Optional v3.9 replay-plan mode.
REPLAY_PLAN="${REPLAY_PLAN:-}"
PLAN_ROOT="${PLAN_ROOT:-}"

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
PLAN_ENTRIES="$OUT_ROOT/replay_plan_entries.tsv"

mkdir -p "$OUT_ROOT/normal/events" "$OUT_ROOT/normal/logs" "$OUT_ROOT/normal/configs" \
         "$OUT_ROOT/lstm" "$OUT_ROOT/replay_inputs"
[[ "$MODE" == normal || "$MODE" == lstm || "$MODE" == both ]] || {
  echo "[error] MODE must be normal, lstm, or both" >&2; exit 2; }
[[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "[error] MAX_JOBS must be a positive integer" >&2; exit 2; }

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

plan_entries() {
  local plan="$1" root="$2" out="$3"
  python3 - "$plan" "$root" "$out" <<'PY'
from __future__ import print_function
import csv
import re
import sys
from pathlib import Path
plan = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
out = Path(sys.argv[3])
required = {"tag", "trace", "source_rel"}
safe = re.compile(r"^[A-Za-z0-9_.-]+$")
if not plan.is_file():
    raise SystemExit("missing replay plan: {}".format(plan))
if not root.is_dir():
    raise SystemExit("missing plan root: {}".format(root))
seen, rows = set(), []
with plan.open(newline="") as handle:
    reader = csv.DictReader(handle)
    missing = sorted(required - set(reader.fieldnames or []))
    if missing:
        raise SystemExit("replay plan missing columns: {}".format(missing))
    for line_no, row in enumerate(reader, start=2):
        tag = (row.get("tag") or "").strip()
        trace = (row.get("trace") or "").strip()
        source_rel = (row.get("source_rel") or "").strip()
        if not tag or not trace or not source_rel:
            raise SystemExit("blank tag/trace/source_rel at replay plan row {}".format(line_no))
        if not safe.match(tag) or not safe.match(trace):
            raise SystemExit("unsafe tag or trace at replay plan row {}".format(line_no))
        if tag in seen:
            raise SystemExit("duplicate plan tag: {}".format(tag))
        seen.add(tag)
        source = Path(source_rel)
        source = source if source.is_absolute() else root / source
        source = source.resolve()
        if not source.is_file() or not source.stat().st_size:
            raise SystemExit("missing/nonempty rich list for {}: {}".format(tag, source))
        rows.append((tag, trace, str(source)))
if not rows:
    raise SystemExit("replay plan has no candidates")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as handle:
    for tag, trace, source in rows:
        handle.write("{}\t{}\t{}\n".format(tag, trace, source))
print("[plan] {} candidates from {}".format(len(rows), plan))
PY
}

build_all() {
  [[ "$BUILD" == 1 ]] || return 0
  RESET_PATCH="$RESET_PATCH" CHAMP_DIR="$CHAMP_DIR" bash "$PATCH"
  if [[ "$MODE" == normal || "$MODE" == both ]]; then
    ( cd "$CHAMP_DIR" && bash ./build_champsim.sh no multi no 1 )
  fi
  if [[ "$MODE" == lstm || "$MODE" == both ]]; then
    CHAMP_DIR="$CHAMP_DIR" bash "$REPLAYER_BUILD"
  fi
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

run_lstm_common() {
  local trace="$1" label="$2" rich="$3"
  local variant_root="$OUT_ROOT/lstm/$label"
  local trace_file="$TRACE_DIR/$trace.champsimtrace.xz"
  local oracle="$ORACLE_DIR/$trace.oracle.csv.gz"
  local keyed="$OUT_ROOT/replay_inputs/$label/$trace.pc_line_occ.csv"
  local raw="$variant_root/events/$trace.events.csv"
  local out="$raw.gz"
  local log="$variant_root/logs/$trace.standalone_lstm.log"
  mkdir -p "$variant_root/events" "$variant_root/logs" "$(dirname "$keyed")"
  [[ -s "$trace_file" && -s "$rich" && -s "$oracle" ]] || {
    echo "[error] missing trace, rich export, or oracle for $label/$trace" >&2; return 1; }
  [[ "$FORCE" == 1 || ! -s "$out" || ! -s "$log" || ! -s "$keyed" ]] || {
    echo "[skip lstm] $label $trace"; return 0; }
  python3 "$PREP" --rich-list "$rich" --oracle "$oracle" --out "$keyed" > "$variant_root/logs/$trace.prepare.log" 2>&1
  echo "[lstm] $label $trace"
  PFETCH_LIST_PATH="$keyed" DEMAND_EVENT_LOG="$raw" "$REPLAY_BIN" \
    --l2c_prefetcher_types=list_replayer \
    --warmup_instructions="$WARMUP" --simulation_instructions="$SIM" \
    -traces "$trace_file" > "$log" 2>&1
  [[ -s "$raw" ]] && grep -Fq 'key=pc_line_occ' "$log" || {
    echo "[error] replay failed: $label $trace" >&2; return 1; }
  gzip -f "$raw"
}

run_lstm_legacy() {
  local trace="$1" label="$2" art_dir="$3"
  local rich="$art_dir/prefetch_list_${trace}_cl${CHUNK_LEN}_${EXPORT_SUFFIX}.csv"
  run_lstm_common "$trace" "$label" "$rich"
}

run_lstm_plan() {
  local tag="$1" trace="$2" rich="$3"
  run_lstm_common "$trace" "$tag" "$rich"
}

if [[ -n "$REPLAY_PLAN" ]]; then
  [[ -z "$PLAN_ROOT" ]] && PLAN_ROOT="$(cd "$(dirname "$REPLAY_PLAN")" && pwd)"
  plan_entries "$REPLAY_PLAN" "$PLAN_ROOT" "$PLAN_ENTRIES"
  cp -f "$REPLAY_PLAN" "$OUT_ROOT/v3_9_replay_plan.csv"
fi

build_all
if [[ "$MODE" == normal || "$MODE" == both ]]; then
  [[ -x "$NORMAL_BIN" ]] || { echo "[error] expected normal binary missing: $NORMAL_BIN" >&2; exit 2; }
fi
if [[ "$MODE" == lstm || "$MODE" == both ]]; then
  [[ -x "$REPLAY_BIN" ]] || { echo "[error] expected replayer binary missing: $REPLAY_BIN" >&2; exit 2; }
fi
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
  if [[ -n "$REPLAY_PLAN" ]]; then
    while IFS=$'\t' read -r tag trace rich; do
      launch run_lstm_plan "$tag" "$trace" "$rich"
    done < "$PLAN_ENTRIES"
  else
    IFS=';' read -r -a variants <<< "$NN_VARIANTS"
    for spec in "${variants[@]}"; do
      [[ -n "$spec" ]] || continue
      label="${spec%%=*}"
      art_dir="${spec#*=}"
      [[ "$label" != "$art_dir" ]] || {
        echo "[error] NN_VARIANTS entry must be label=dir" >&2; exit 2; }
      for trace in $TRACES; do
        launch run_lstm_legacy "$trace" "$label" "$art_dir"
      done
    done
  fi
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
REPLAY_PLAN=$REPLAY_PLAN
PLAN_ROOT=$PLAN_ROOT
WARMUP=$WARMUP
SIM=$SIM
CHUNK_LEN=$CHUNK_LEN
EXPORT_SUFFIX=$EXPORT_SUFFIX
NORMAL_BIN=$NORMAL_BIN
REPLAY_BIN=$REPLAY_BIN
NORMAL_SUMMARY=$OUT_ROOT/normal/summary.csv
MAX_JOBS=$MAX_JOBS
FORCE=$FORCE
EOF
echo "[done] $OUT_ROOT"
