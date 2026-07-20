#!/usr/bin/env bash
set -euo pipefail

# Historical v7/v9/v11 seven-track installer retained for reproducibility only.
# Install the seven Colab output archives into their matching, revisioned run
# directories.  Existing nonempty output directories fail closed so an old
# model cannot be mixed with a new revision.

CACHE_ROOT="${CACHE_ROOT:-$HOME/cache}"
UPLOAD_DIR="${1:-$HOME/direct_action_uploads}"

install_one() {
  local experiment="$1"
  local run_id="$2"
  local archive="$UPLOAD_DIR/$run_id.colab_output.tar.gz"
  local run_dir="$CACHE_ROOT/formal_NN_training/experiments/$experiment/runs/$run_id"
  local output_dir="$run_dir/colab_output"

  [[ -s "$archive" ]] || {
    echo "[error] missing output archive: $archive" >&2
    return 1
  }
  if [[ -d "$output_dir" ]] && find "$output_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "[error] refusing to mix outputs in nonempty $output_dir" >&2
    return 1
  fi
  mkdir -p "$output_dir"
  tar -xzf "$archive" -C "$output_dir"
  [[ -s "$output_dir/sweep_manifest.json" ]] || {
    echo "[error] $archive did not contain sweep_manifest.json" >&2
    return 1
  }
  if ! find "$output_dir" -mindepth 2 -maxdepth 2 -name run_metadata.json -print -quit | grep -q .; then
    echo "[error] $archive did not contain model metadata" >&2
    return 1
  fi
  echo "[INSTALLED] $experiment :: $run_id"
}

install_one 602_offline_lstm_stride \
  602_offline_lstm_stride_variable_delta_free_running_v7_seed7
install_one 602_offline_lstm_streamer \
  602_offline_lstm_streamer_variable_delta_free_running_v7_seed7
install_one 602_offline_lstm_ampm \
  602_offline_lstm_ampm_variable_delta_free_running_v7_seed7
install_one 623_offline_lstm_stride \
  623_offline_lstm_stride_variable_delta_free_running_v9_seed7
install_one 623_offline_cnn_stride \
  623_offline_cnn_stride_variable_delta_free_running_v9_seed7
install_one 623_offline_lstm_spp \
  623_offline_lstm_spp_variable_delta_free_running_v11_seed7
install_one 623_offline_cnn_spp \
  623_offline_cnn_spp_variable_delta_free_running_v11_seed7

echo "[PASS] seven revision-matched Colab outputs installed"
