#!/usr/bin/env bash
# Short, dynamic workflow for the two active 623 v25 experiments.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STRIDE_EXP="$ROOT/formal_NN_training/experiments/623_offline_lstm_stride"
SPP_EXP="$ROOT/formal_NN_training/experiments/623_offline_lstm_spp"
STRIDE_CONTRACT="$STRIDE_EXP/python/model_contract.py"
SPP_CONTRACT="$SPP_EXP/python/model_contract.py"
STRIDE_RUN="$(python3 "$STRIDE_CONTRACT" --field run_id)"
SPP_RUN="$(python3 "$SPP_CONTRACT" --field run_id)"
STRIDE_DIR="$STRIDE_EXP/runs/$STRIDE_RUN"
SPP_DIR="$SPP_EXP/runs/$SPP_RUN"
STRIDE_TRACE="$(python3 "$STRIDE_CONTRACT" --field trace)"
STRIDE_POLICY="$(python3 "$STRIDE_CONTRACT" --field policy)"
STRIDE_TAGS="$(python3 "$STRIDE_CONTRACT" --tags-csv)"
STRIDE_OUTPUT_REPAIR="$STRIDE_EXP/python/repair_colab_output_manifest.py"
INSTALL_COLAB_OUTPUT="$ROOT/formal_NN_training/common/install_colab_output.py"

for value in "$STRIDE_RUN" "$SPP_RUN"; do
  [[ "$value" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "[error] unsafe run_id: $value" >&2
    exit 2
  }
done

prepare() {
  STAGE=reuse-input BUILD=0 FORCE=0 \
    bash "$STRIDE_EXP/linux/run_server.sh"
  STAGE=reuse-input BUILD=0 FORCE=0 \
    bash "$SPP_EXP/linux/run_server.sh"

  local stride_input="$STRIDE_DIR/$STRIDE_RUN.colab_input.tar.gz"
  local spp_input="$SPP_DIR/$SPP_RUN.colab_input.tar.gz"
  local bundle="$ROOT/623_v25_bundle.colab_input.tar.gz"
  gzip -t "$stride_input"
  gzip -t "$spp_input"
  (
    local staging temporary
    staging="$(mktemp -d /tmp/623_v25_bundle.XXXXXX)"
    temporary="$(mktemp "$ROOT/.623_v25_bundle.XXXXXX.colab_input.tar.gz")"
    trap 'rm -rf -- "$staging"; rm -f -- "$temporary"' EXIT
    cp -p "$STRIDE_EXP/colab/623_offline_lstm_stride_A100.ipynb" \
      "$staging/stride.ipynb"
    cp -p "$SPP_EXP/colab/623_offline_lstm_spp_A100.ipynb" \
      "$staging/spp.ipynb"
    cp -p "$stride_input" "$staging/stride.colab_input.tar.gz"
    cp -p "$spp_input" "$staging/spp.colab_input.tar.gz"
    tar -czf "$temporary" -C "$staging" \
      stride.ipynb spp.ipynb \
      stride.colab_input.tar.gz spp.colab_input.tar.gz
    gzip -t "$temporary"
    mv -f "$temporary" "$bundle"
  )
  gzip -t "$bundle"
  echo "[ready] $bundle"
}

install_outputs() {
  local stride_source="$ROOT/stride.colab_output.tar.gz"
  local spp_source="$ROOT/spp.colab_output.tar.gz"
  local stride_target="$STRIDE_DIR/$STRIDE_RUN.colab_output.tar.gz"
  local spp_target="$SPP_DIR/$SPP_RUN.colab_output.tar.gz"
  local stride_temporary spp_temporary
  gzip -t "$stride_source"
  gzip -t "$spp_source"
  mkdir -p "$STRIDE_DIR" "$SPP_DIR"
  stride_temporary="$(mktemp "$STRIDE_DIR/.colab_output.XXXXXX.tar.gz")"
  spp_temporary="$(mktemp "$SPP_DIR/.colab_output.XXXXXX.tar.gz")"
  trap 'rm -f -- "$stride_temporary" "$spp_temporary"' RETURN
  cp -p "$stride_source" "$stride_temporary"
  cp -p "$spp_source" "$spp_temporary"
  gzip -t "$stride_temporary"
  gzip -t "$spp_temporary"
  mv -f "$stride_temporary" "$stride_target"
  mv -f "$spp_temporary" "$spp_target"
  trap - RETURN
  echo "[installed] $stride_target"
  echo "[installed] $spp_target"
}

pid_is_running() {
  local pid_file="$1" expected_command="$2" pid command
  [[ -s "$pid_file" ]] || return 1
  pid="$(cat "$pid_file")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  command="$(ps -p "$pid" -o args= 2>/dev/null || true)"
  [[ "$command" == *"$expected_command"* ]]
}

repair_stride_output() {
  local archive="$STRIDE_DIR/$STRIDE_RUN.colab_output.tar.gz"
  local installed_manifest="$STRIDE_DIR/colab_output/sweep_manifest.json"
  python3 "$STRIDE_OUTPUT_REPAIR" \
    --archive "$archive" \
    --installed-manifest "$installed_manifest" \
    --run-id "$STRIDE_RUN" \
    --trace "$STRIDE_TRACE" \
    --policy "$STRIDE_POLICY"
  gzip -t "$archive"
  python3 "$INSTALL_COLAB_OUTPUT" \
    --archive "$archive" \
    --output-dir "$STRIDE_DIR/colab_output" \
    --model-tags "$STRIDE_TAGS"
}

replay_stride() {
  if pid_is_running "$SPP_DIR/replay.pid" \
    "$SPP_EXP/linux/run_server.sh"; then
    echo "[wait] SPP replay is still running; leave it alone and start Stride after it finishes" >&2
    return 2
  fi
  repair_stride_output
  BUILD=0 FORCE=0 RESET_PATCH=0 JOBS="${JOBS:-8}" \
    bash "$STRIDE_EXP/linux/launch_server.sh" replay
}

replay_spp() {
  if pid_is_running "$STRIDE_DIR/replay.pid" \
    "$STRIDE_EXP/linux/run_server.sh"; then
    echo "[wait] Stride replay is still running; do not start SPP concurrently" >&2
    return 2
  fi
  BUILD=0 FORCE=0 RESET_PATCH=0 JOBS="${JOBS:-8}" \
    bash "$SPP_EXP/linux/launch_server.sh" replay
}

analyze_spp() {
  if pid_is_running "$SPP_DIR/replay.pid" \
    "$SPP_EXP/linux/run_server.sh"; then
    echo "[wait] SPP replay is still running; analyze only after it finishes" >&2
    return 2
  fi
  STAGE=analyze BUILD=0 FORCE=0 RESET_PATCH=0 JOBS="${JOBS:-8}" \
    bash "$SPP_EXP/linux/run_server.sh"
}

replay() {
  echo "[error] run the tracks separately: replay-spp, then replay-stride after SPP finishes" >&2
  return 2
}

show_one_status() {
  local label="$1" run_dir="$2" pid_file="$2/replay.pid"
  local log="$run_dir/replay.nohup.log" result="$run_dir/matched_comparison.json"
  local pid running=0
  echo "[$label]"
  if [[ -s "$pid_file" ]]; then
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      running=1
      ps -fp "$pid" || true
    else
      echo "replay PID $pid is not running"
    fi
  else
    echo "replay has not been launched"
  fi
  if [[ -s "$result" ]]; then
    python3 - "$result" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text())
except (OSError, ValueError) as error:
    print("[result] invalid {}: {}".format(path, error))
else:
    print("[result] status={} {}".format(payload.get("status"), path))
PY
  else
    echo "[result] no matched_comparison.json yet"
  fi
  if [[ -f "$log" ]]; then
    if [[ "$running" == 0 ]]; then
      echo "[retained replay log tail; this may show an earlier failure]"
    fi
    tail -n 25 "$log"
  else
    echo "no replay log yet: $log"
  fi
}

status() {
  show_one_status STRIDE "$STRIDE_DIR"
  show_one_status SPP "$SPP_DIR"
}

package() {
  bash "$ROOT/formal_NN_training/experiments/package_623_v25_evidence.sh"

  local result_dir="$ROOT/formal_NN_training/results/623_v25_evidence"
  local combined="$ROOT/formal_NN_training/results/623_v25_evidence.tar.gz"
  local temporary
  mkdir -p "$result_dir"
  cp -f "$STRIDE_DIR/$STRIDE_RUN.evidence.tar.gz" \
    "$result_dir/stride.evidence.tar.gz"
  cp -f "$SPP_DIR/$SPP_RUN.evidence.tar.gz" \
    "$result_dir/spp.evidence.tar.gz"
  gzip -t "$result_dir/stride.evidence.tar.gz"
  gzip -t "$result_dir/spp.evidence.tar.gz"
  temporary="$(mktemp "$ROOT/formal_NN_training/results/.623_v25_evidence.XXXXXX.tar.gz")"
  trap 'rm -f -- "$temporary"' RETURN
  tar -czf "$temporary" -C "$result_dir" \
    stride.evidence.tar.gz spp.evidence.tar.gz
  gzip -t "$temporary"
  mv -f "$temporary" "$combined"
  trap - RETURN
  echo "[ready] $combined"
}

case "${1:-}" in
  prepare) prepare ;;
  install-outputs) install_outputs ;;
  repair-stride-output) repair_stride_output ;;
  replay-spp) replay_spp ;;
  replay-stride) replay_stride ;;
  analyze-spp) analyze_spp ;;
  replay) replay ;;
  status) status ;;
  package) package ;;
  *)
    echo "usage: $0 {prepare|install-outputs|repair-stride-output|replay-spp|replay-stride|analyze-spp|status|package}" >&2
    exit 2
    ;;
esac
