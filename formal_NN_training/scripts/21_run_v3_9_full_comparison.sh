#!/usr/bin/env bash
# Run all current v3.9 candidates, refresh the normal-prefetcher zoo, then
# rewrite the replay comparison against that fresh normal baseline summary.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PLAN="${PLAN:-$ROOT/formal_NN_training/artifacts/v3_9/v3_9_replay_plan.csv}"
RUN_TAG="${RUN_TAG:-v3_9_campaign_$(date +%Y%m%d_%H%M%S)}"
REPLAY_OUT="${REPLAY_OUT:-$ROOT/formal_NN_training/results/v3_9_replay/$RUN_TAG}"
NORMAL_OUT="${NORMAL_OUT:-$ROOT/formal_NN_training/results/v3_9_normal_baselines/$RUN_TAG}"
WARMUP="${WARMUP:-25000000}"
SIM="${SIM:-25000000}"
REPLAY_MAX_JOBS="${REPLAY_MAX_JOBS:-2}"
NORMAL_MAX_JOBS="${NORMAL_MAX_JOBS:-2}"
FORCE="${FORCE:-0}"
RUN_FULL_NORMAL_SWEEP="${RUN_FULL_NORMAL_SWEEP:-1}"
TRACES="${TRACES:-602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B 620.omnetpp_s-874B 623.xalancbmk_s-700B}"
PREFETCHERS="${PREFETCHERS:-no_pref stride streamer ampm spp ipcp sms sandbox power7}"

REPLAY_SCRIPT="$ROOT/formal_NN_training/scripts/19_run_v3_9_replay_plan.sh"
NORMAL_SCRIPT="$ROOT/formal_NN_training/scripts/04_run_normal_prefetcher_sweep.sh"
PARSER="$ROOT/formal_NN_training/scripts/20_parse_v3_9_replay_plan.py"
HISTORICAL_BASELINE="${BASELINE_SUMMARY:-$ROOT/formal_NN_training/results/prefetcher_baselines/summary.csv}"
NORMAL_BIN="${NORMAL_BIN:-$ROOT/external/ChampSim/bin/perceptron-no-multi-no-ship-1core}"
NORMAL_BUILD="${NORMAL_BUILD:-0}"

[[ -f "$REPLAY_SCRIPT" && -f "$PARSER" && -s "$PLAN" ]] || { echo "[error] missing plan or v3.9 scripts" >&2; exit 2; }

PLAN="$PLAN" RUN_TAG="$RUN_TAG" OUT_DIR="$REPLAY_OUT" WARMUP="$WARMUP" SIM="$SIM" \
  MAX_JOBS="$REPLAY_MAX_JOBS" FORCE="$FORCE" BASELINE_SUMMARY="$HISTORICAL_BASELINE" \
  bash "$REPLAY_SCRIPT"

if [[ "$RUN_FULL_NORMAL_SWEEP" == "1" ]]; then
  TRACES="$TRACES" PREFETCHERS="$PREFETCHERS" WARMUP="$WARMUP" SIM="$SIM" \
    MAX_JOBS="$NORMAL_MAX_JOBS" OUT_ROOT="$NORMAL_OUT" BIN="$NORMAL_BIN" \
    BUILD="$NORMAL_BUILD" FORCE="$FORCE" bash "$NORMAL_SCRIPT"

  python3 "$PARSER" --plan "$PLAN" --log-root "$REPLAY_OUT/logs" \
    --replay-input-root "$REPLAY_OUT/replay_inputs" --same-binary-log-root "$REPLAY_OUT/logs" \
    --baseline-summary "$NORMAL_OUT/summary.csv" --out-dir "$REPLAY_OUT" \
    --no-pref-ipc-tolerance "${NO_PREF_IPC_TOLERANCE:-0.002}"
fi

cat > "$REPLAY_OUT/CAMPAIGN_INFO.txt" <<EOF
RUN_KIND=v3_9_replay_plus_normal_comparison
PLAN=$PLAN
REPLAY_OUT=$REPLAY_OUT
NORMAL_OUT=$NORMAL_OUT
RUN_FULL_NORMAL_SWEEP=$RUN_FULL_NORMAL_SWEEP
WARMUP=$WARMUP
SIM=$SIM
REPLAY_MAX_JOBS=$REPLAY_MAX_JOBS
NORMAL_MAX_JOBS=$NORMAL_MAX_JOBS
NORMAL_BIN=$NORMAL_BIN
PREFETCHERS=$PREFETCHERS
EOF

echo "[done] $REPLAY_OUT/v3_9_comparison.md"
