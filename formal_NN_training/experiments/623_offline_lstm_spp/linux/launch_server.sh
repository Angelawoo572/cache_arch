#!/usr/bin/env bash
# Safe nohup launcher.  It creates RUN_DIR before the shell opens the log.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/formal_NN_training/experiments/623_offline_lstm_spp"
MODEL_POINTS_SCRIPT="$EXP/python/model_contract.py"
DEFAULT_RUN_ID="$(python3 "$MODEL_POINTS_SCRIPT" --field run_id)"
RUN_ID="${RUN_ID:-$DEFAULT_RUN_ID}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "[error] RUN_ID must be one safe path token: $RUN_ID" >&2
  exit 2
fi
STAGE="${1:-${STAGE:-replay}}"
CANONICAL_RUN_DIR="$EXP/runs/$RUN_ID"
if [[ -n "${RUN_DIR:-}" && "$RUN_DIR" != "$CANONICAL_RUN_DIR" ]]; then
  echo "[isolation] ignoring foreign RUN_DIR=$RUN_DIR" >&2
fi
RUN_DIR="$CANONICAL_RUN_DIR"
LOG="$RUN_DIR/$STAGE.nohup.log"
PID_FILE="$RUN_DIR/$STAGE.pid"

case "$STAGE" in
  reuse-input|build|collect|replay|analyze) ;;
  *) echo "[error] stage must be reuse-input, build, collect, replay, or analyze" >&2; exit 2 ;;
esac

mkdir -p "$RUN_DIR"
if [[ -s "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE")"
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "[error] $STAGE is already running as PID $old_pid" >&2
    exit 2
  fi
fi

nohup env \
  RUN_ID="$RUN_ID" \
  RUN_DIR="$RUN_DIR" \
  STAGE="$STAGE" \
  FORCE="${FORCE:-1}" \
  BUILD="${BUILD:-1}" \
  JOBS="${JOBS:-8}" \
  RESET_PATCH="${RESET_PATCH:-1}" \
  bash "$EXP/linux/run_server.sh" \
  > "$LOG" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$PID_FILE"

echo "[started] stage=$STAGE pid=$pid"
echo "[log] $LOG"
echo "tail -f \"$LOG\""
