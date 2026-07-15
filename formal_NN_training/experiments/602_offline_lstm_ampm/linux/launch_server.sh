#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/formal_NN_training/experiments/602_offline_lstm_ampm"
RUN_ID="${RUN_ID:-602_offline_lstm_ampm_direct_v3_seed7}"
STAGE="${1:-${STAGE:-replay}}"
RUN_DIR="${RUN_DIR:-$EXP/runs/$RUN_ID}"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/$STAGE.nohup.log"; PID_FILE="$RUN_DIR/$STAGE.pid"
case "$STAGE" in build|collect|replay|analyze) ;; *) echo "[error] invalid stage $STAGE" >&2; exit 2;; esac
if [[ -s "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then echo "[error] already running PID $(cat "$PID_FILE")" >&2; exit 2; fi
nohup env -u COLAB_SOURCE_INPUT_DIR RUN_ID="$RUN_ID" RUN_DIR="$RUN_DIR" STAGE="$STAGE" FORCE="${FORCE:-1}" BUILD="${BUILD:-1}" JOBS="${JOBS:-8}" bash "$EXP/linux/run_server.sh" >"$LOG" 2>&1 </dev/null &
printf '%s\n' "$!" >"$PID_FILE"
echo "[started] $STAGE pid=$(cat "$PID_FILE")"; echo "[log] $LOG"; echo "tail -f \"$LOG\""
