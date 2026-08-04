#!/usr/bin/env bash
# Safe nohup launcher.  It creates RUN_DIR before the shell opens the log.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/formal_NN_training/experiments/623_offline_lstm_spp"
DEFAULT_RUN_ID="623_offline_lstm_spp_factorized_fill_v17_seed7"
RUN_ID="${RUN_ID:-$DEFAULT_RUN_ID}"
STAGE="${1:-${STAGE:-replay}}"
CANONICAL_RUN_DIR="$EXP/runs/$RUN_ID"
if [[ -n "${RUN_DIR:-}" && "$RUN_DIR" != "$CANONICAL_RUN_DIR" ]]; then
  echo "[isolation] ignoring foreign RUN_DIR=$RUN_DIR" >&2
fi
RUN_DIR="$CANONICAL_RUN_DIR"
LOG="$RUN_DIR/$STAGE.nohup.log"
PID_FILE="$RUN_DIR/$STAGE.pid"

case "$STAGE" in
  build|collect|replay|analyze) ;;
  *) echo "[error] stage must be build, collect, replay, or analyze" >&2; exit 2 ;;
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
