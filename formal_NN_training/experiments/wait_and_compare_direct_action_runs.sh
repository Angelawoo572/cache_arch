#!/usr/bin/env bash
# Wait for the seven direct-action result contracts, then compare 623 LSTM/CNN.
# Compatible with the server's Python 3.6 and intentionally uses no pandas.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
WAIT_INTERVAL="${WAIT_INTERVAL:-60}"

results=(
  "formal_NN_training/experiments/602_offline_lstm_stride/runs/602_offline_lstm_stride_variable_delta_free_running_v7_seed7/matched_comparison.json"
  "formal_NN_training/experiments/602_offline_lstm_streamer/runs/602_offline_lstm_streamer_variable_delta_free_running_v7_seed7/matched_comparison.json"
  "formal_NN_training/experiments/602_offline_lstm_ampm/runs/602_offline_lstm_ampm_variable_delta_free_running_v7_seed7/matched_comparison.json"
  "formal_NN_training/experiments/623_offline_lstm_stride/runs/623_offline_lstm_stride_variable_delta_free_running_v9_seed7/matched_comparison.json"
  "formal_NN_training/experiments/623_offline_cnn_stride/runs/623_offline_cnn_stride_variable_delta_free_running_v9_seed7/matched_comparison.json"
  "formal_NN_training/experiments/623_offline_lstm_spp/runs/623_offline_lstm_spp_variable_delta_free_running_v11_seed7/matched_comparison.json"
  "formal_NN_training/experiments/623_offline_cnn_spp/runs/623_offline_cnn_spp_variable_delta_free_running_v11_seed7/matched_comparison.json"
)

result_is_pass() {
  python3 - "$1" <<'PY'
import json
import sys

try:
    payload = json.load(open(sys.argv[1]))
except Exception:
    raise SystemExit(1)
ok = (
    payload.get("status") == "PASS"
    and payload.get("fair_comparison_claim_allowed") is True
)
raise SystemExit(0 if ok else 1)
PY
}

result_summary() {
  python3 - "$1" <<'PY'
import json
import os
import sys

path = sys.argv[1]
if not os.path.isfile(path):
    print("MISSING")
    raise SystemExit(0)
try:
    payload = json.load(open(path))
except Exception as exc:
    print("INVALID: {}".format(exc))
    raise SystemExit(0)
failures = payload.get("failures") or []
print("{} fair={} failures={}".format(
    payload.get("status"),
    payload.get("fair_comparison_claim_allowed"),
    " | ".join(str(item) for item in failures) if failures else "none",
))
PY
}

for result in "${results[@]}"; do
  while ! result_is_pass "$result"; do
    echo "[wait] $(result_summary "$result") :: $result"
    sleep "$WAIT_INTERVAL"
  done
  echo "[PASS] $result"
done

python3 formal_NN_training/experiments/compare_623_split_architectures.py \
  --policy stride \
  --lstm-run-dir formal_NN_training/experiments/623_offline_lstm_stride/runs/623_offline_lstm_stride_variable_delta_free_running_v9_seed7 \
  --cnn-run-dir formal_NN_training/experiments/623_offline_cnn_stride/runs/623_offline_cnn_stride_variable_delta_free_running_v9_seed7 \
  --out-dir formal_NN_training/results/623_stride_lstm_vs_cnn_free_running &
stride_compare=$!

python3 formal_NN_training/experiments/compare_623_split_architectures.py \
  --policy spp \
  --lstm-run-dir formal_NN_training/experiments/623_offline_lstm_spp/runs/623_offline_lstm_spp_variable_delta_free_running_v11_seed7 \
  --cnn-run-dir formal_NN_training/experiments/623_offline_cnn_spp/runs/623_offline_cnn_spp_variable_delta_free_running_v11_seed7 \
  --out-dir formal_NN_training/results/623_spp_lstm_vs_cnn_free_running &
spp_compare=$!

wait "$stride_compare"
wait "$spp_compare"
echo "[PASS] all seven runs and both comparisons completed"
