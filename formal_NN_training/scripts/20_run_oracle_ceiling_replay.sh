#!/usr/bin/env bash
# Build oracle ceiling lists and replay them through the existing keyed driver.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
TRACES="${TRACES:-602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B 620.omnetpp_s-874B 623.xalancbmk_s-700B}"
ORACLE_DIR="${ORACLE_DIR:-$ROOT/formal_NN_training/results/standalone_nn_data/oracle}"
CEILING_MODE="${CEILING_MODE:-omniscient}"
LEDGER_DIR="${LEDGER_DIR:-}"
MIN_LEAD_EVENTS="${MIN_LEAD_EVENTS:-8}"
MIN_LEAD_BIN="${MIN_LEAD_BIN:-2}"
MAX_DEGREE="${MAX_DEGREE:-1}"
REQUIRE_FULL_LEDGER="${REQUIRE_FULL_LEDGER:-0}"
WARMUP="${WARMUP:-25000000}"
SIM="${SIM:-25000000}"
MAX_JOBS="${MAX_JOBS:-2}"
FORCE="${FORCE:-0}"
RUN_TAG="${RUN_TAG:-ceiling_${CEILING_MODE}_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/formal_NN_training/results/oracle_ceiling/$RUN_TAG}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-$ROOT/formal_NN_training/results/prefetcher_baselines/summary.csv}"
BASELINE_REFERENCE_JSON="${BASELINE_REFERENCE_JSON:-$ROOT/formal_NN_training/_cfg/replay_same_binary_no_pref_reference_v4_0.json}"

BUILDER="$ROOT/formal_NN_training/scripts/19_build_oracle_ceiling_lists.py"
REPLAY="$ROOT/formal_NN_training/scripts/08_run_standalone_lstm_replay.sh"
[[ "$CEILING_MODE" == "omniscient" || "$CEILING_MODE" == "bank" ]] || { echo "invalid CEILING_MODE" >&2; exit 2; }
[[ -f "$BUILDER" && -f "$REPLAY" ]] || { echo "missing ceiling builder or replay driver" >&2; exit 2; }
[[ "$CEILING_MODE" != "bank" || -n "$LEDGER_DIR" ]] || { echo "LEDGER_DIR required for bank mode" >&2; exit 2; }
mkdir -p "$OUT_ROOT/lists" "$OUT_ROOT/meta"
PLAN="$OUT_ROOT/${CEILING_MODE}_ceiling_replay_plan.csv"

for trace in $TRACES; do
  oracle="$ORACLE_DIR/${trace}.oracle.csv.gz"
  list="$OUT_ROOT/lists/${trace}.${CEILING_MODE}.csv"
  meta="$OUT_ROOT/meta/${trace}.${CEILING_MODE}.json"
  [[ -s "$oracle" ]] || { echo "missing oracle: $oracle" >&2; exit 2; }
  args=(--oracle "$oracle" --out "$list" --mode "$CEILING_MODE" --max-degree "$MAX_DEGREE" --meta-out "$meta")
  if [[ "$CEILING_MODE" == "omniscient" ]]; then
    args+=(--min-lead-events "$MIN_LEAD_EVENTS")
  else
    ledger=$(ls "$LEDGER_DIR"/decision_ledger_${trace}_*_full_candidates.csv.gz 2>/dev/null | head -n 1 || true)
    [[ -n "$ledger" ]] || ledger=$(ls "$LEDGER_DIR"/decision_ledger_${trace}_*_val_candidates.csv.gz 2>/dev/null | head -n 1 || true)
    [[ -n "$ledger" ]] || { echo "missing candidate ledger for $trace" >&2; exit 2; }
    args+=(--ledger-candidates "$ledger" --min-lead-bin "$MIN_LEAD_BIN")
    [[ "$REQUIRE_FULL_LEDGER" == "1" ]] && args+=(--require-full-coverage)
  fi
  python3 "$BUILDER" "${args[@]}"
done

python3 - "$PLAN" "$OUT_ROOT" "$CEILING_MODE" $TRACES <<'PY'
import csv, sys
from pathlib import Path
plan, root, mode, *traces = sys.argv[1:]
root = Path(root)
rows = []
for trace in traces:
    rows.append({"tag":"ceiling_{}_{}".format(mode, trace.split(".",1)[0]),"trace":trace,"source_rel":str((root / "lists" / "{}.{}.csv".format(trace,mode)).resolve()),"candidate_role":"oracle_ceiling_not_nn_candidate","replay_kind":"{}_ceiling".format(mode),"model_family":"oracle_ceiling","policy_tag":mode})
with open(plan, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
PY

REPLAY_OUT="$OUT_ROOT/replay"
RUN_TAG="$RUN_TAG" OUT_DIR="$REPLAY_OUT" REPLAY_PLAN="$PLAN" PLAN_ROOT="$OUT_ROOT" ORACLE_DIR="$ORACLE_DIR" BASELINE_SUMMARY="$BASELINE_SUMMARY" WARMUP="$WARMUP" SIM="$SIM" MAX_JOBS="$MAX_JOBS" FORCE="$FORCE" BASELINE_REFERENCE_JSON="$BASELINE_REFERENCE_JSON" bash "$REPLAY"
echo "[done] $OUT_ROOT"
