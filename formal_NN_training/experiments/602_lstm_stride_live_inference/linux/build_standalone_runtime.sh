#!/usr/bin/env bash
# Build the dependency-free C++11 model loader, runtime, and parity runner.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/formal_NN_training/experiments/602_lstm_stride_live_inference"
RUN_DIR="${RUN_DIR:-$EXP/runs/602_gcc_stride_prefix_seed7}"
OUT="${OUT:-$RUN_DIR/bin/stride_lstm_standalone}"
CXX="${CXX:-g++}"
FORCE="${FORCE:-0}"

usage() {
  cat <<'EOF'
Usage: build_standalone_runtime.sh

Environment:
  OUT=PATH     Output binary below an ignored run directory.
  CXX=g++      C++11 compiler.
  FORCE=1      Rebuild an existing binary.
EOF
}

[[ "${1:-}" != "-h" && "${1:-}" != "--help" ]] || { usage; exit 0; }
if [[ -x "$OUT" && "$FORCE" != 1 ]]; then
  echo "[skip valid] $OUT"
  exit 0
fi
mkdir -p "$(dirname "$OUT")"
"$CXX" -std=c++11 -Wall -Wextra -Werror -pedantic -O2 -I "$EXP/runtime" "$EXP/runtime/stride_lstm_model_loader.cc" "$EXP/runtime/stride_lstm_runtime.cc" "$EXP/runtime/standalone_runner.cc" -o "$OUT"
"$OUT" --help >/dev/null
echo "[ok] $OUT"

