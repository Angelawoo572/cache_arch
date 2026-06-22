#!/usr/bin/env bash
# Replay base-independent LSTM prefetch lists in Pythia using a PC-line-occurrence
# trigger key rather than a no-prefetch global L2 callback ordinal.
#
# Why: once a prefetch changes memory timing, independent L2 callbacks can reorder.
# A global callback index from the no-prefetch run is therefore not invariant under
# the intervention. Script 10 maps each rich notebook event to:
#
#   pc,line,occ,prefetch_addr
#
# where occ is the no-prefetch occurrence count of that (pc,line) pair. The
# ListReplayer maintains this local counter at runtime. This is keyed offline-policy
# replay, not in-simulator PyTorch inference.

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
[[ -f "$PREP" ]] || { echo "[error] missing keyed-list converter: $PREP" >&2; exit 2; }
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

meta_counts() {
  local meta="$1"
  python3 - "$meta" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    x = json.load(f)
print(int(x.get("entries", 0)), int(x.get("unique_trigger_keys", 0)), int(x.get("unmatched_rows", -1)))
PY
}

validate_log() {
  local trace="$1" log="$2" meta="$3"
  local final entries expected_keys unmatched emitted observed matched loaded_keys

  if ! grep -q "adding L2C_PREFETCHER: list_replayer" "$log"; then
    echo "[error] $trace: binary did not instantiate list_replayer at L2; inspect $log" >&2
    return 1
  fi
  if ! grep -q "PC-line-occ triggers" "$log"; then
    echo "[error] $trace: keyed PC-line-occ list was not loaded; rebuild with current Pythia" >&2
    return 1
  fi

  read -r entries expected_keys unmatched < <(meta_counts "$meta")
  if [[ "$unmatched" != "0" || "$entries" == "0" || "$expected_keys" == "0" ]]; then
    echo "[error] $trace: converter metadata is not a complete keyed list: entries=$entries keys=$expected_keys unmatched=$unmatched" >&2
    return 1
  fi

  final="$(grep "\[list_replayer\].*runtime ROI L2 LOAD accesses" "$log" | tail -1 || true)"
  if [[ -z "$final" ]]; then
    echo "[error] $trace: no keyed ListReplayer final-stat line; inspect $log" >&2
    return 1
  fi

  emitted="$(sed -nE 's/.*\] emitted ([0-9]+) candidates over.*/\1/p' <<<"$final")"
  observed="$(sed -nE 's/.*over ([0-9]+) runtime ROI L2 LOAD accesses.*/\1/p' <<<"$final")"
  matched="$(sed -nE 's/.*\(([0-9]+) matched PC-line-occ triggers;.*/\1/p' <<<"$final")"
  loaded_keys="$(sed -nE 's/.*; ([0-9]+) loaded trigger keys; key=pc_line_occ\).*/\1/p' <<<"$final")"

  if [[ -z "$emitted" || -z "$observed" || -z "$matched" || -z "$loaded_keys" ]]; then
    echo "[error] $trace: could not parse current keyed final line: $final" >&2
    return 1
  fi
  if [[ "$loaded_keys" != "$expected_keys" ]]; then
    echo "[error] $trace: runtime loaded $loaded_keys keys, converter produced $expected_keys keys" >&2
    return 1
  fi
  if (( emitted > entries || matched > expected_keys || observed == 0 )); then
    echo "[error] $trace: impossible keyed replay counters: $final" >&2
    return 1
  fi

  echo "[ok] $trace: keyed replay transport passed; emitted=$emitted/$entries entries, matched=$matched/$expected_keys PC-line-occ triggers, runtime_l2_loads=$observed"
}

run_one() {
  local trace="$1"
  local trace_file="traces/${trace}.champsimtrace.xz"
  local rich="$ART_DIR/prefetch_list_${trace}_cl${CHUNK_LEN}_${RICH_SUFFIX}.csv"
  local oracle="$ORACLE_DIR/${trace}.oracle.csv.gz"
  local keyed="$REPLAY_DIR/${trace}.pc_line_occ.csv"
  local meta="$REPLAY_DIR/${trace}.pc_line_occ.csv.meta.json"
  local log="$LOG_DIR/${trace}.oracle_replacer.log"

  echo "============================================================"
  echo "[run] $trace"
  echo "[tag] $RUN_TAG"
  echo "[binary] $BIN"
  echo "[L2 knob] $L2_REPLAYER_KNOB"
  echo "[rich] $rich"
  echo "[oracle] $oracle"
  echo "[keyed] $keyed"
  echo "[log] $log"
  echo "============================================================"

  [[ -f "$trace_file" ]] || { echo "[error] missing trace: $trace_file" >&2; return 1; }
  [[ -f "$rich" ]] || { echo "[error] missing rich export: $rich" >&2; return 1; }
  [[ -f "$oracle" ]] || { echo "[error] missing oracle: $oracle" >&2; return 1; }

  python3 "$PREP" --rich-list "$rich" --oracle "$oracle" --out "$keyed" \
    > "$LOG_DIR/${trace}.prepare.log" 2>&1

  # -traces MUST be last: Pythia treats every following argument as a trace.
  PFETCH_LIST_PATH="$keyed" \
  "$BIN" \
    "$L2_REPLAYER_KNOB" \
    --warmup_instructions="$WARMUP" \
    --simulation_instructions="$SIM" \
    -traces "$trace_file" \
    > "$log" 2>&1

  if ! validate_log "$trace" "$log" "$meta"; then
    echo "[oracle-replay-validation] status=keyed_transport_fail" >> "$log"
    return 1
  fi
  echo "[oracle-replay-validation] status=keyed_transport_pass" >> "$log"
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
  echo "[failed] one or more keyed replays failed transport validation; see $LOG_DIR" >&2
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

echo "[all done] PC-line-occ keyed LSTM replay"
echo "[summary] $SUMMARY_OUT"
