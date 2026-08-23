#!/usr/bin/env bash
# Install deterministic sources and build one frozen-live ChampSim binary.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/formal_NN_training/experiments/602_lstm_stride_live_inference"
CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
RUN_DIR="${RUN_DIR:-$EXP/runs/602_gcc_stride_prefix_seed7}"
OUT="${OUT:-$RUN_DIR/bin/champsim.602_stride_lstm_live}"
FORCE="${FORCE:-0}"
INSTALLER="$EXP/runtime/champsim/install_live_prefetcher.sh"
BUILT="$CHAMP_DIR/bin/perceptron-no-stride_lstm_live-no-ship-1core"

usage() {
  cat <<'EOF'
Usage: build_live_champsim.sh

Environment:
  CHAMP_DIR=PATH   ChampSim checkout.
  RUN_DIR=PATH     Ignored experiment run directory.
  OUT=PATH         Preserved live binary path.
  FORCE=1          Archive and rebuild an existing OUT.

Requires ChampSim's pinned libbf build. Existing binaries are archived or
left intact; this script never runs git clean.
EOF
}

[[ "${1:-}" != "-h" && "${1:-}" != "--help" ]] || { usage; exit 0; }
[[ -x "$CHAMP_DIR/build_champsim.sh" ]] || {
  echo "[error] missing ChampSim build script: $CHAMP_DIR" >&2
  exit 2
}
[[ -f "$CHAMP_DIR/libbf/build/lib/libbf.a" ]] || {
  echo "[error] pinned libbf is not built; follow COMMANDS.md first" >&2
  exit 2
}
if [[ -x "$OUT" && "$FORCE" != 1 ]]; then
  echo "[skip valid] $OUT"
  exit 0
fi

CHAMP_DIR="$CHAMP_DIR" FORCE="$FORCE" bash "$INSTALLER" install
archive="$RUN_DIR/replaced/$(date -u +%Y%m%dT%H%M%SZ)/bin"
if [[ -e "$OUT" ]]; then
  mkdir -p "$archive"
  mv "$OUT" "$archive/"
fi
if [[ -e "$BUILT" ]]; then
  mkdir -p "$archive"
  mv "$BUILT" "$archive/"
fi

(cd "$CHAMP_DIR" && ./build_champsim.sh no stride_lstm_live no 1)
[[ -x "$BUILT" ]] || { echo "[error] expected binary absent: $BUILT" >&2; exit 3; }
mkdir -p "$(dirname "$OUT")"
cp "$BUILT" "$OUT"
python3 - "$ROOT" "$CHAMP_DIR" "$EXP" "$OUT" <<'PY'
import json
import sys
from pathlib import Path

root, champ, experiment, binary = map(Path, sys.argv[1:])
sources = [
    experiment / "runtime/stride_lstm_model_loader.h",
    experiment / "runtime/stride_lstm_model_loader.cc",
    experiment / "runtime/stride_lstm_runtime.h",
    experiment / "runtime/stride_lstm_runtime.cc",
    experiment / "runtime/champsim/stride_lstm_live.h",
    experiment / "runtime/champsim/stride_lstm_live.cc",
]
payload = {
    "schema_version": 1,
    "binary": str(binary),
    "sources": [str(path.relative_to(root)) for path in sources],
}
path = binary.with_suffix(binary.suffix + ".build_metadata.json")
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print("[ok] {}".format(path))
PY
git -C "$CHAMP_DIR" status --short
echo "[ok] $OUT"
