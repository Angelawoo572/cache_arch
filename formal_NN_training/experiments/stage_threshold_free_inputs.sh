#!/usr/bin/env bash
set -euo pipefail

# Reuse the exact previously collected stream bytes for the three 602 reruns.
# The two 623 policies are freshly collected in their LSTM directories and
# then copied byte-for-byte into their separate CNN directories by
# stage_623_split_inputs.sh.
# Source IDs may be overridden when a server uses different archival names.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGER="$ROOT/formal_NN_training/experiments/stage_direct_action_inputs.sh"

stage() {
  bash "$STAGER" "$1" "$2" "$3"
}

stage 602-stride \
  "${SRC_602_STRIDE:-602_offline_lstm_stride_stateful_v2_seed7}" \
  602_offline_lstm_stride_variable_delta_free_running_v7_seed7

stage 602-streamer \
  "${SRC_602_STREAMER:-602_offline_lstm_streamer_stateful_v2_seed7}" \
  602_offline_lstm_streamer_variable_delta_free_running_v7_seed7

stage 602-ampm \
  "${SRC_602_AMPM:-602_offline_lstm_ampm_stateful_v2_seed7}" \
  602_offline_lstm_ampm_variable_delta_free_running_v7_seed7

echo "[READY] three unchanged-input 602 Colab archives"
echo "[NEXT] collect 623 Stride and SPP once in the standalone LSTM tracks"
echo "[NEXT] run stage_623_split_inputs.sh stride and ... spp"
