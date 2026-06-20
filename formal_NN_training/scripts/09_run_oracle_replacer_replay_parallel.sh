#!/usr/bin/env bash
# Replay the base-independent oracle-LSTM prefetch lists in parallel.
#
# IMPORTANT CONTRACT
# ------------------
# The rich notebook CSV is NOT list-replayer input. Its historical first two
# columns are `order` (= cycle) and `pc`, whereas list_replayer expects
# `idx,0xprefetch_addr`. This runner first converts rich lists to strict inputs
# indexed by the no-prefetch ROI L2-LOAD ordinal (demand_idx).
#
# The binary must be the ROI-aligned L2 ListReplayer built by script 11. Its
# final counter must equal the number of rows in the corresponding oracle table.
# A mismatch means the prefetcher is attached at L1D, includes warmup/RFO, or
# otherwise sees a different access stream. The script fails rather than
# reporting no-prefetch IPC as an invalid neural result.
#
# This simulator is Pythia, not newer ChampSim. Pythia requires:
#   --warmup_instructions=<N>
#   --simulation_instructions=<N>
#   -traces <trace>
# and treats every argument after -traces as a trace filename.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MAX_JOBS=${MAX_JOBS:-3}
WARMUP=${WARMUP:-25000000}
SIM=${SIM:-25000000}
BIN=${BIN:-"$ROOT/external/ChampSim/bin/champsim.oracle_l2_replayer"}
# Pythia parses this as the runtime L2 prefetcher list.
L2_REPLAYER_KNOB=${L2_REPLAYER_KNOB:---l2c_prefetcher_types=list_replayer}

OUT_DIR=${OUT_DIR:-formal_NN_training/results/oracle_replacer_replay}
LOG_DIR="$OUT_DIR/logs"
REPLAY_DIR="$OUT_DIR/replay_inputs"
ART_DIR=${ART_DIR:-formal_NN_training/artifacts/oracle_replacer}
ORACLE_DIR=${ORACLE_DIR:-formal_NN_training/results/base_prefetcher_zoo/oracle_event_table_pc_line_occ}
PREP=${PREP:-formal_NN_training/scripts/10_prepare_oracle_replacer_replay_input.py}
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

if [[ -n "${TRACES:-}" ]]; then
  # Example: TRACES="619.lbm_s-4268B 602.gcc_s-734B"
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
            raise SystemExit(f"non-contiguous demand_idx: expected {n}, saw {i}")
        n += 1
print(n)
PY
}

validate_log() {
  local trace="$1" log="$2" expected="$3"
  local final actual matched emitted

  if ! grep -q "adding L2C_PREFETCHER: list_replayer" "$log"; then
    echo "[error] $trace: binary did not instantiate list_replayer at L2; inspect $log" >&2
    return 1
  fi
  if ! grep -q "\[list_replayer\] loaded" "$log"; then
    echo "[error] $trace: list_replayer did not load a strict list; inspect $log" >&2
    return 1
  fi

  final="$(grep "\[list_replayer\].*over .*accesses" "$log" | tail -1 || true)"
  if [[ -z "$final" ]]; then
    echo "[error] $trace: no list_replayer final-stat line; inspect $log" >&2
    return 1
  fi

  # Supports both the new Pythia-native line and the older legacy line:
  #   emitted X candidates over Y ROI L2 LOAD accesses (Z matched access indices)
  #   issued  X prefetches over Y accesses (A attempted, Z matched access indices)
  actual="$(sed -nE 's/.*over ([0-9]+) (ROI L2 LOAD )?accesses.*/\1/p' <<<"$final")"
  matched="$(sed -nE 's/.*\(([0-9]+) matched access indices\).*/\1/p' <<<"$final")"
  emitted="$(sed -nE 's/.*\] (issued|emitted) ([0-9]+) .*/\2/p' <<<"$final")"

  if [[ -z "$actual" || -z "$matched" || -z "$emitted" ]]; then
    echo "[error] $trace: could not parse final list_replayer line: $final" >&2
    return 1
  fi
  if [[ "$actual" != "$expected" ]]; then
    cat >&2 <<EOF
[error] $trace: replay access-domain mismatch.
  oracle ROI L2 LOAD rows : $expected
  list_replayer counter   : $actual
The binary is not counting the same post-warmup L2 LOAD stream as the oracle.
Do NOT use this IPC. Re-run script 11, then use the rebuilt binary.
final: $final
EOF
    return 1
  fi
  if [[ "$matched" == "0" || "$emitted" == "0" ]]; then
    echo "[error] $trace: no list entries matched/emitted despite a non-empty strict input: $final" >&2
    return 1
  fi
  echo "[ok] $trace: ROI-L2 domain aligned; $final"
}

run_one() {
  local trace="$1"
  local trace_file="traces/${trace}.champsimtrace.xz"
  local rich="$ART_DIR/prefetch_list_${trace}_cl128_fair_dedup_lru2048.csv"
  local oracle="$ORACLE_DIR/${trace}.oracle.csv.gz"
  local strict="$REPLAY_DIR/${trace}.l2roi.idx_addr.csv"
  local log="$LOG_DIR/${trace}.oracle_replacer.log"
  local expected

  echo "============================================================"
  echo "[run] $trace"
  echo "[binary] $BIN"
  echo "[L2 knob] $L2_REPLAYER_KNOB"
  echo "[rich] $rich"
  echo "[oracle] $oracle"
  echo "[strict] $strict"
  echo "[log] $log"
  echo "============================================================"

  [[ -f "$trace_file" ]] || { echo "[error] missing trace: $trace_file" >&2; return 1; }
  [[ -f "$rich" ]] || { echo "[error] missing rich export: $rich" >&2; return 1; }
  [[ -f "$oracle" ]] || { echo "[error] missing oracle: $oracle" >&2; return 1; }

  python3 "$PREP" --rich-list "$rich" --oracle "$oracle" --out "$strict" \
    > "$LOG_DIR/${trace}.prepare.log" 2>&1
  expected="$(expected_roi_rows "$oracle")"

  # -traces MUST be last: Pythia considers every following argument a trace.
  PFETCH_LIST_PATH="$strict" \
  "$BIN" \
    "$L2_REPLAYER_KNOB" \
    --warmup_instructions="$WARMUP" \
    --simulation_instructions="$SIM" \
    -traces "$trace_file" \
    > "$log" 2>&1

  validate_log "$trace" "$log" "$expected"
  echo "[done] $trace"
}

running=0
status=0
for trace in "${TRACE_LIST[@]}"; do
  run_one "$trace" &
  ((running+=1))
  if (( running >= MAX_JOBS )); then
    if ! wait -n; then status=1; fi
    ((running-=1))
  fi
done
while (( running > 0 )); do
  if ! wait -n; then status=1; fi
  ((running-=1))
done

if (( status != 0 )); then
  echo "[failed] one or more replays failed validation; see $LOG_DIR" >&2
  exit "$status"
fi

echo "[all done] validated ROI-L2 oracle replays"
