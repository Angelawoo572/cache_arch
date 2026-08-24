#!/usr/bin/env bash
# Validate existing offline artifacts, run the complete live curve in two
# disjoint h8/h16 processes, and package a LaTeX-only Overleaf project.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/formal_NN_training/experiments/602_lstm_stride_live_inference"
RUN_DIR="${RUN_DIR:-$EXP/runs/602_gcc_stride_prefix_seed7}"
OLD_RUN_DIR="${OLD_RUN_DIR:-}"
KEYED_REPLAYER_BINARY="${KEYED_REPLAYER_BINARY:-$ROOT/external/ChampSim/bin/champsim.602_offline_replay}"
STATE_MODE="${STATE_MODE:-parity}"
RUN_MODE="${RUN_MODE:-live}"
WARMUP_INSTRUCTIONS="${WARMUP_INSTRUCTIONS:-25000000}"
SIMULATION_INSTRUCTIONS="${SIMULATION_INSTRUCTIONS:-25000000}"
COMMAND="${1:-run}"

usage() {
  cat <<'EOF'
Usage: run_complete_live_overleaf.sh [run|status]

Required for run:
  RUN_DIR=PATH
  OLD_RUN_DIR=PATH

The run command never trains, never launches offline ChampSim, and never
changes an existing valid live log. It validates existing offline artifacts,
runs every exported h8/h16 live checkpoint in two disjoint concurrent workers,
aggregates measured results, generates PGFPlots LaTeX, and writes the
LaTeX-only Overleaf ZIP.
EOF
}

[[ "$COMMAND" != "-h" && "$COMMAND" != "--help" ]] || { usage; exit 0; }
[[ "$COMMAND" == "run" || "$COMMAND" == "status" ]] || { usage >&2; exit 2; }

if [[ "$COMMAND" == "status" ]]; then
  python3 "$EXP/validation/validate_complete_live.py" \
    --run-dir "$RUN_DIR" --state-mode "$STATE_MODE" --run-mode "$RUN_MODE"
  exit $?
fi

[[ -d "$RUN_DIR" ]] || { echo "[error] missing RUN_DIR: $RUN_DIR" >&2; exit 3; }
[[ -n "$OLD_RUN_DIR" && -d "$OLD_RUN_DIR" ]] || {
  echo "[error] set OLD_RUN_DIR to the original 20M run" >&2
  exit 3
}
[[ -x "$RUN_DIR/bin/champsim.602_stride_lstm_live" ]] || {
  echo "[error] missing live ChampSim binary" >&2
  exit 3
}
[[ -x "$KEYED_REPLAYER_BINARY" ]] || {
  echo "[error] missing keyed-replayer binary" >&2
  exit 3
}
[[ -s "$ROOT/traces/602.gcc_s-734B.champsimtrace.xz" ]] || {
  echo "[error] missing 602 trace" >&2
  exit 3
}

lock="$RUN_DIR/.complete_live_overleaf.lock"
if ! mkdir "$lock" 2>/dev/null; then
  echo "[error] another complete-live workflow may be running: $lock" >&2
  exit 4
fi
trap 'rmdir "$lock" 2>/dev/null || true' EXIT

job_tag="$(date -u +%Y%m%dT%H%M%SZ)"
job_dir="$RUN_DIR/logs/complete_live_overleaf_$job_tag"
mkdir -p "$job_dir" "$RUN_DIR/report"
printf '%s\n' "$job_dir" > "$RUN_DIR/logs/latest_complete_live_job_dir.txt"
echo "[job] $job_dir"

python3 "$EXP/validation/validate_training_sweep.py" \
  --run-dir "$RUN_DIR" --require-all
python3 "$EXP/validation/validate_offline_fairness.py" \
  --run-dir "$RUN_DIR" \
  --keyed-replayer-binary "$KEYED_REPLAYER_BINARY" --require-all
python3 "$EXP/validation/validate_20m_regression.py" \
  --old-run-dir "$OLD_RUN_DIR" --new-run-dir "$RUN_DIR"

python3 "$EXP/python/summarize_training_prefixes.py" \
  --prefix-root "$RUN_DIR/training_prefixes" --out-dir "$RUN_DIR"
python3 "$EXP/python/aggregate_offline_results.py" --run-dir "$RUN_DIR"

echo "[live] launching complete h8 and h16 curves concurrently"
env LIVE_ALL_VALID=1 HIDDEN_SIZES=8 RUN_DIR="$RUN_DIR" \
  STATE_MODE="$STATE_MODE" RUN_MODE="$RUN_MODE" \
  WARMUP_INSTRUCTIONS="$WARMUP_INSTRUCTIONS" \
  SIMULATION_INSTRUCTIONS="$SIMULATION_INSTRUCTIONS" \
  bash "$EXP/linux/run_live_sweep.sh" run >"$job_dir/h8.log" 2>&1 &
h8_pid=$!
printf '%s\n' "$h8_pid" > "$job_dir/h8.pid"

env LIVE_ALL_VALID=1 HIDDEN_SIZES=16 RUN_DIR="$RUN_DIR" \
  STATE_MODE="$STATE_MODE" RUN_MODE="$RUN_MODE" \
  WARMUP_INSTRUCTIONS="$WARMUP_INSTRUCTIONS" \
  SIMULATION_INSTRUCTIONS="$SIMULATION_INSTRUCTIONS" \
  bash "$EXP/linux/run_live_sweep.sh" run >"$job_dir/h16.log" 2>&1 &
h16_pid=$!
printf '%s\n' "$h16_pid" > "$job_dir/h16.pid"
printf '[live] h8 PID=%s; h16 PID=%s\n' "$h8_pid" "$h16_pid"

set +e
wait "$h8_pid"
h8_status=$?
wait "$h16_pid"
h16_status=$?
set -e
printf '%s\n' "$h8_status" > "$job_dir/h8.status"
printf '%s\n' "$h16_status" > "$job_dir/h16.status"
if [[ "$h8_status" -ne 0 || "$h16_status" -ne 0 ]]; then
  echo "[error] live workers failed: h8=$h8_status h16=$h16_status" >&2
  echo "[inspect] $job_dir/h8.log and $job_dir/h16.log" >&2
  exit 5
fi

python3 "$EXP/validation/validate_complete_live.py" \
  --run-dir "$RUN_DIR" --state-mode "$STATE_MODE" --run-mode "$RUN_MODE"
python3 "$EXP/python/aggregate_live_results.py" --run-dir "$RUN_DIR"
python3 "$EXP/python/compare_offline_live.py" --run-dir "$RUN_DIR"
python3 "$EXP/python/generate_overleaf_figures.py" --run-dir "$RUN_DIR"
RUN_DIR="$RUN_DIR" FORCE=1 bash "$EXP/linux/package_overleaf_report.sh"

archive="$RUN_DIR/602_stride_offline_sufficiency_overleaf.zip"
[[ -s "$archive" ]] || { echo "[error] Overleaf ZIP missing" >&2; exit 6; }
echo "[PASS] complete live curve and LaTeX-only Overleaf package"
echo "[ready] $archive"
