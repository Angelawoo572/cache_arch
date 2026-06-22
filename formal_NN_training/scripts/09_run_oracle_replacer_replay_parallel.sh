#!/usr/bin/env bash
# Replay base-independent oracle-LSTM prefetch lists in Pythia with strict
# ROI-L2-LOAD alignment and write one simulator-result summary.csv.
#
# Contract:
#   1) Notebook rich export: cycle/pc/line + prefetch_addr (diagnostic format).
#   2) Script 10 converts it to strict idx,prefetch_addr where idx is the
#      no-prefetch post-warmup ROI L2 LOAD ordinal.
#   3) The Pythia ListReplayer validates every runtime (pc,line) callback
#      against the dense oracle reference before it emits a candidate.
#
# The default rich filename matches the notebook's current default:
#   prefetch_list_<trace>_cl128_fair_dedup_lru2048.csv
#
# Use ART_DIR to select a frozen threshold/sweep directory. Do not overwrite a
# previously replayed list with a new notebook run.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MAX_JOBS=${MAX_JOBS:-3}
WARMUP=${WARMUP:-25000000}
SIM=${SIM:-25000000}
CHUNK_LEN=${CHUNK_LEN:-128}
DEDUP_CAPACITY=${DEDUP_CAPACITY:-2048}
RICH_SUFFIX=${RICH_SUFFIX:-"fair_dedup_lru${DEDUP_CAPACITY}"}
RUN_TAG=${RUN_TAG:-"oracle_lstm_cl${CHUNK_LEN}_${RICH_SUFFIX}"}

BIN=${BIN:-"$ROOT/external/ChampSim/bin/champsim.oracle_l2_replayer"}
L2_REPLAYER_KNOB=${L2_REPLAYER_KNOB:---l2c_prefetcher_types=list_replayer}
# Maximum terminal callback-count drift after signature validation. This is a
# safety bound, not a performance knob.
MAX_TAIL_SLACK=${MAX_TAIL_SLACK:-64}

OUT_DIR=${OUT_DIR:-"formal_NN_training/results/oracle_replacer_replay/${RUN_TAG}"}
LOG_DIR="$OUT_DIR/logs"
REPLAY_DIR="$OUT_DIR/replay_inputs"
ART_DIR=${ART_DIR:-formal_NN_training/artifacts/oracle_replacer}
ORACLE_DIR=${ORACLE_DIR:-formal_NN_training/results/base_prefetcher_zoo/oracle_event_table_pc_line_occ}
PREP=${PREP:-formal_NN_training/scripts/10_prepare_oracle_replacer_replay_input.py}
PARSER=${PARSER:-formal_NN_training/scripts/12_parse_oracle_replacer_replay.py}
PARSE_RESULTS=${PARSE_RESULTS:-1}
SUMMARY_OUT=${SUMMARY_OUT:-"$OUT_DIR/summary.csv"}
BASELINE_METRICS=${BASELINE_METRICS:-formal_NN_training/results/base_prefetcher_zoo/normal_prefetcher_metrics.csv}
OFFLINE_SUMMARY=${OFFLINE_SUMMARY:-"$ART_DIR/oracle_replacer_sweep.csv"}

mkdir -p "$LOG_DIR" "$REPLAY_DIR"

if [[ ! -x "$BIN" ]]; then
  cat >&2 <<EOF
[error] replay binary is not executable: $BIN
Build it first:
  bash formal_NN_training/scripts/11_install_oracle_l2_replayer.sh
EOF
  exit 2
fi
if [[ ! -f "$PREP" ]]; then
  echo "[error] missing strict-list converter: $PREP" >&2
  exit 2
fi
if (( PARSE_RESULTS )) && [[ ! -f "$PARSER" ]]; then
  echo "[error] missing replay-summary parser: $PARSER" >&2
  exit 2
fi

if [[ -n "${TRACES:-}" ]]; then
  read -r -a TRACE_LIST <<< "$TRACES"
else
  TRACE_LIST=(
    "602.gcc_s-734B"
    "619.lbm_s-4268B"
    "605.mcf_s-994B"
    "620.omnetpp_s-874B"
    "623.xalancbmk_s-700B"
  )
fi

expected_roi_rows() {
  local oracle="$1"
  python3 - "$oracle" <<'PY'
import csv, gzip, sys
p = sys.argv[1]
op = gzip.open if p.endswith('.gz') else open
with op(p, 'rt', newline='') as f:
    r = csv.DictReader(f)
    n = 0
    for row in r:
        i = int(float(row['demand_idx']))
        if i != n:
            raise SystemExit("non-contiguous demand_idx: expected {}, saw {}".format(n, i))
        n += 1
print(n)
PY
}

# Print: <strict candidates whose idx < observed> <unique trigger idxs < observed>
strict_prefix_stats() {
  local strict="$1" observed="$2"
  python3 - "$strict" "$observed" <<'PY'
import csv, sys
p, observed = sys.argv[1], int(sys.argv[2])
entries = 0
indices = set()
prev = -1
with open(p, newline='') as f:
    r = csv.DictReader(f)
    if r.fieldnames != ['idx', 'prefetch_addr']:
        raise SystemExit('strict list schema must be exactly idx,prefetch_addr')
    for row in r:
        idx = int(row['idx'])
        if idx < prev:
            raise SystemExit('strict list is not sorted by idx')
        prev = idx
        if idx < observed:
            entries += 1
            indices.add(idx)
print(entries, len(indices))
PY
}

validate_log() {
  local trace="$1" log="$2" expected="$3" strict="$4"
  local final actual matched emitted sig_mismatch ref_tail prefix_entries prefix_indices delta abs_delta

  if ! grep -q "adding L2C_PREFETCHER: list_replayer" "$log"; then
    echo "[error] $trace: binary did not instantiate list_replayer at L2; inspect $log" >&2
    return 1
  fi
  if ! grep -q "\[list_replayer\] loaded .* dense ROI L2 LOAD signatures" "$log"; then
    echo "[error] $trace: oracle signature reference was not loaded; rebuild/re-run with current scripts" >&2
    return 1
  fi

  final="$(grep "\[list_replayer\].*over .*ROI L2 LOAD accesses" "$log" | tail -1 || true)"
  if [[ -z "$final" ]]; then
    echo "[error] $trace: no list_replayer final-stat line; inspect $log" >&2
    return 1
  fi

  actual="$(sed -nE 's/.*over ([0-9]+) ROI L2 LOAD accesses.*/\1/p' <<<"$final")"
  emitted="$(sed -nE 's/.*\] emitted ([0-9]+) candidates.*/\1/p' <<<"$final")"
  matched="$(sed -nE 's/.*\(([0-9]+) matched access indices;.*/\1/p' <<<"$final")"
  sig_mismatch="$(sed -nE 's/.*; ([0-9]+) signature mismatches;.*/\1/p' <<<"$final")"
  ref_tail="$(sed -nE 's/.*; ([0-9]+) post-reference tail accesses;.*/\1/p' <<<"$final")"

  if [[ -z "$actual" || -z "$matched" || -z "$emitted" || -z "$sig_mismatch" || -z "$ref_tail" ]]; then
    echo "[error] $trace: could not parse current list_replayer final line: $final" >&2
    return 1
  fi
  if [[ "$sig_mismatch" != "0" ]]; then
    cat >&2 <<EOF
[error] $trace: runtime L2 callback stream diverged from the oracle stream.
  signature mismatches : $sig_mismatch
  final: $final
The index replay is invalid; candidate emission was suppressed at mismatches.
EOF
    return 1
  fi

  read -r prefix_entries prefix_indices < <(strict_prefix_stats "$strict" "$actual")
  if [[ "$emitted" != "$prefix_entries" || "$matched" != "$prefix_indices" ]]; then
    cat >&2 <<EOF
[error] $trace: strict-list prefix did not replay exactly.
  observed L2 LOAD prefix : [0, $((actual - 1))]
  strict prefix candidates: $prefix_entries; replayer emitted: $emitted
  strict prefix trigger ids: $prefix_indices; replayer matched: $matched
  final: $final
EOF
    return 1
  fi

  delta=$((actual - expected))
  abs_delta=$delta
  if (( abs_delta < 0 )); then abs_delta=$((-abs_delta)); fi
  if (( abs_delta > MAX_TAIL_SLACK )); then
    cat >&2 <<EOF
[error] $trace: terminal callback-count drift exceeds safety bound.
  oracle ROI L2 LOAD rows : $expected
  replay L2 LOAD rows     : $actual
  absolute tail drift     : $abs_delta (limit $MAX_TAIL_SLACK)
  final: $final
EOF
    return 1
  fi

  if (( delta < 0 )); then
    echo "[warn] $trace: validated prefix ends $((-delta)) callbacks before oracle tail; all observed callbacks signature-match."
  elif (( delta > 0 )); then
    echo "[warn] $trace: validated replay has $delta post-reference tail callbacks; all oracle-prefix callbacks signature-match."
  fi
  echo "[ok] $trace: signature-validated ROI-L2 replay; $final"
}

run_one() {
  local trace="$1"
  local trace_file="traces/${trace}.champsimtrace.xz"
  local rich="$ART_DIR/prefetch_list_${trace}_cl${CHUNK_LEN}_${RICH_SUFFIX}.csv"
  local oracle="$ORACLE_DIR/${trace}.oracle.csv.gz"
  local strict="$REPLAY_DIR/${trace}.l2roi.idx_addr.csv"
  local reference="$REPLAY_DIR/${trace}.l2roi.reference.csv"
  local log="$LOG_DIR/${trace}.oracle_replacer.log"
  local expected

  echo "============================================================"
  echo "[run] $trace"
  echo "[tag] $RUN_TAG"
  echo "[binary] $BIN"
  echo "[L2 knob] $L2_REPLAYER_KNOB"
  echo "[rich] $rich"
  echo "[oracle] $oracle"
  echo "[strict] $strict"
  echo "[reference] $reference"
  echo "[log] $log"
  echo "============================================================"

  [[ -f "$trace_file" ]] || { echo "[error] missing trace: $trace_file" >&2; return 1; }
  [[ -f "$rich" ]] || { echo "[error] missing rich export: $rich" >&2; return 1; }
  [[ -f "$oracle" ]] || { echo "[error] missing oracle: $oracle" >&2; return 1; }

  python3 "$PREP" --rich-list "$rich" --oracle "$oracle" --out "$strict" --reference-out "$reference" \
    > "$LOG_DIR/${trace}.prepare.log" 2>&1
  expected="$(expected_roi_rows "$oracle")"

  # -traces MUST be last: Pythia considers every following argument a trace.
  PFETCH_LIST_PATH="$strict" \
  PFETCH_REF_PATH="$reference" \
  "$BIN" \
    "$L2_REPLAYER_KNOB" \
    --warmup_instructions="$WARMUP" \
    --simulation_instructions="$SIM" \
    -traces "$trace_file" \
    > "$log" 2>&1

  if ! validate_log "$trace" "$log" "$expected" "$strict"; then
    echo "[oracle-replay-validation] status=fail" >> "$log"
    return 1
  fi
  echo "[oracle-replay-validation] status=pass" >> "$log"
  echo "[done] $trace"
}

running=0
status=0
for trace in "${TRACE_LIST[@]}"; do
  run_one "$trace" &
  running=$((running + 1))
  if (( running >= MAX_JOBS )); then
    if ! wait -n; then status=1; fi
    running=$((running - 1))
  fi
done
while (( running > 0 )); do
  if ! wait -n; then status=1; fi
  running=$((running - 1))
done

if (( status != 0 )); then
  echo "[failed] one or more replays failed validation; see $LOG_DIR" >&2
  exit "$status"
fi

if (( PARSE_RESULTS )); then
  parse_args=(
    --log-root "$LOG_DIR"
    --replay-input-root "$REPLAY_DIR"
    --out "$SUMMARY_OUT"
    --traces "${TRACE_LIST[*]}"
  )
  if [[ -f "$BASELINE_METRICS" ]]; then
    parse_args+=(--baseline-metrics "$BASELINE_METRICS")
  else
    echo "[warn] no normal-prefetcher baseline table: $BASELINE_METRICS" >&2
  fi
  if [[ -f "$OFFLINE_SUMMARY" ]]; then
    parse_args+=(--offline-summary "$OFFLINE_SUMMARY")
  else
    echo "[warn] no offline notebook summary to join: $OFFLINE_SUMMARY" >&2
  fi
  python3 "$PARSER" "${parse_args[@]}"
fi

echo "[all done] signature-validated ROI-L2 replay"
echo "[summary] $SUMMARY_OUT"
