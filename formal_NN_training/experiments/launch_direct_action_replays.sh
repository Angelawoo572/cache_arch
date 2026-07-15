#!/usr/bin/env bash
set -euo pipefail

# Start all independent direct-action replay/analyze pipelines concurrently.
# Each per-experiment launcher uses nohup.  ChampSim builds remain serialized by
# the repository-wide build lock; completed builds then replay concurrently.

CACHE_ROOT="${CACHE_ROOT:-$HOME/cache}"
STAGE="${1:-replay}"
case "$STAGE" in replay|analyze) ;; *) echo "usage: $0 [replay|analyze]" >&2; exit 2 ;; esac

launch() {
  local experiment="$1"
  local run_id="$2"
  local exp="$CACHE_ROOT/formal_NN_training/experiments/$experiment"
  [[ -x "$exp/linux/launch_server.sh" ]] || {
    echo "[error] missing launcher: $exp/linux/launch_server.sh" >&2
    return 1
  }
  echo "[launch] $experiment :: $run_id :: $STAGE"
  RUN_ID="$run_id" FORCE="${FORCE:-1}" BUILD="${BUILD:-1}" JOBS="${JOBS:-8}" \
    bash "$exp/linux/launch_server.sh" "$STAGE"
}

launch 602_offline_lstm_stride \
  602_offline_lstm_stride_direct_v3_seed7
launch 602_offline_lstm_streamer \
  602_offline_lstm_streamer_direct_v3_seed7
launch 602_offline_lstm_ampm \
  602_offline_lstm_ampm_direct_v3_seed7
launch 623_offline_lstm_cnn_stride \
  623_offline_lstm_cnn_stride_direct_v3_seed7
launch 623_offline_lstm_cnn_spp \
  623_offline_lstm_cnn_spp_direct_v4_seed7

echo "[started] five nohup pipelines; inspect each RUN_DIR/<stage>.nohup.log"

