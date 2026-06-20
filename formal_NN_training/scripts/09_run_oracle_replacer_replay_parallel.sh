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
# The binary must also be an ROI-aligned L2 replayer: its final list_replayer
# counter must equal the number of rows in the corresponding oracle table. A
# mismatch means it is attached at L1D, includes warmup/RFO, or otherwise sees
# a different access stream. The script fails rather than reporting no-prefetch
# IPC as an invalid neural result.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MAX_JOBS=${MAX_JOBS:-3}
WARMUP=${WARMUP:-25000000}
SIM=${SIM:-25000000}
BIN=${BIN:-"$ROOT/external/ChampSim/bin/champsim.l2_replayer"}

OUT_DIR=${OUT_DIR:-formal_NN_training/results/oracle_replacer_replay}
LOG_DIR="$OUT_DIR/logs"
REPLAY_DIR="$OUT_DIR/replay_inputs"
ART_DIR=${ART_DIR:-formal_NN_training/artifacts/oracle_replacer}
ORACLE_DIR=${ORACLE_DIR:-formal_NN_training/results/base_prefetcher_zoo/oracle_event_table_pc_line_occ}
PREP=${PREP:-formal_NN_training/scripts/10_prepare_oracle_replacer_replay_input.py}
mkdir -p "$LOG_DIR" "$REPLAY_DIR"

if [[ ! -x "$BIN" ]]; then
  echo "[error] replay binary is not executable: $BIN" >&2
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
    last = -1
    for row in r:
        i = int(float(row['demand_idx']))
        if i != n:
            raise SystemExit(f"non-contiguous demand_idx: expected {n}, saw {i}")
        last = i
        n += 1
print(n)
PY
}

validate_log() {
  local trace="$1" log="$2" expected="$3"
  local final actual matched attempted

  if ! grep -q "\[list_replayer\] loaded" "$log"; then
    echo "[error] $trace: list_replayer did not load a list; inspect $log" >&2
    return 1
  fi

  final="$(grep "\[list_replayer\].*over .* accesses" "$log" | tail -1 || true)"
  if [[ -z "$final" ]]; then
    echo "[error] $trace: no list_replayer final-stat line; inspect $log" >&2
    return 1
  fi

  # Expected legacy format:
  # [list_replayer] issued X prefetches over Y accesses (Z attempted, W matched access indices)
  actual="$(sed -nE 's/.*over ([0-9]+) accesses.*/\1/p' <<<"$final")"
  attempted="$(sed -nE 's/.*\(([0-9]+) attempted,.*/\1/p' <<<"$final")"
  matched="$(sed -nE 's/.*, ([0-9]+) matched access indices\).*/\1/p' <<<"$final")"

  if [[ -z "$actual" ]]; then
    echo "[error] $trace: could not parse final list_replayer line: $final" >&2
    return 1
  fi
  if [[ "$actual" != "$expected" ]]; then
    cat >&2 <<EOF
[error] $trace: replay access-domain mismatch.
  oracle ROI L2 LOAD rows : $expected
  list_replayer counter   : $actual
The current binary is not counting the same ROI L2 LOAD stream as the oracle
(e.g., it is attached at L1D, includes warmup/RFO, or uses a different hook).
Do NOT use this IPC. Rebuild the ROI-aligned L2 list_replayer before replaying.
final: $final
EOF
    return 1
  fi
  if [[ "${matched:-0}" == "0" || "${attempted:-0}" == "0" ]]; then
    echo "[error] $trace: no list entries matched/attempted despite a non-empty strict input: $final" >&2
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

  PFETCH_LIST_PATH="$strict" \
  "$BIN" \
    --warmup-instructions "$WARMUP" \
    --simulation-instructions "$SIM" \
    "$trace_file" \
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
