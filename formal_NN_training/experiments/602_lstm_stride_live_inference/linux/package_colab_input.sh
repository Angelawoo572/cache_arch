#!/usr/bin/env bash
# Package validated, shared prefix streams and the fixed evaluation stream.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/formal_NN_training/experiments/602_lstm_stride_live_inference"
RUN_ID="${RUN_ID:-602_gcc_stride_prefix_seed7}"
RUN_DIR="${RUN_DIR:-$EXP/runs/$RUN_ID}"
OUTPUT="${OUTPUT:-$RUN_DIR/602_stride_live_colab_input.tar.gz}"
FORCE="${FORCE:-0}"

usage() {
  cat <<'EOF'
Usage: package_colab_input.sh

Environment:
  RUN_DIR=PATH    Collected ignored run directory.
  OUTPUT=PATH     Output archive (default: RUN_DIR/602_stride_live_colab_input.tar.gz).
  FORCE=1         Replace an existing archive after validation.

The archive contains no checkpoints or generated replay lists.
EOF
}

[[ "${1:-}" != "-h" && "${1:-}" != "--help" ]] || { usage; exit 0; }
[[ -d "$RUN_DIR/training_prefixes" ]] || { echo "[error] no collected prefixes" >&2; exit 2; }
[[ -s "$RUN_DIR/evaluation/602.gcc_s-734B.eval_stream.csv.gz" ]] || {
  echo "[error] fixed evaluation stream is missing" >&2
  exit 2
}

python3 "$EXP/python/summarize_training_prefixes.py" --prefix-root "$RUN_DIR/training_prefixes" --out-dir "$RUN_DIR"

python3 - "$EXP/config/training_prefix_sweep.json" "$RUN_DIR" <<'PY'
import gzip
import hashlib
import json
import sys
from pathlib import Path

config = json.load(open(sys.argv[1]))
run_dir = Path(sys.argv[2])
errors = []
for point in config["instruction_budgets"]:
    base = run_dir / "training_prefixes" / point["tag"]
    manifest_path = base / "training_manifest.json"
    if not manifest_path.is_file():
        errors.append("missing {}".format(manifest_path))
        continue
    metadata = json.loads(manifest_path.read_text())
    if metadata.get("instruction_budget") != point["instructions"]:
        errors.append("budget mismatch {}".format(manifest_path))
    stream = Path(metadata["training_stream"])
    if not stream.is_absolute():
        candidate = base / stream.name
        stream = candidate if candidate.exists() else stream
    if not stream.is_file():
        errors.append("missing stream for {}".format(point["tag"]))
        continue
    digest = hashlib.sha256(stream.read_bytes()).hexdigest()
    if digest != metadata.get("training_stream_sha256"):
        errors.append("SHA256 mismatch {}".format(stream))
    try:
        with gzip.open(stream, "rt") as handle:
            next(handle)
    except Exception as exc:
        errors.append("invalid gzip {}: {}".format(stream, exc))
if errors:
    raise SystemExit("\n".join(errors))
print("PASS: 17 exact-prefix manifests and streams validated")
PY

if [[ -e "$OUTPUT" && "$FORCE" != 1 ]]; then
  echo "[error] archive exists; set FORCE=1 to replace: $OUTPUT" >&2
  exit 3
fi
mkdir -p "$(dirname "$OUTPUT")"
tmp="$OUTPUT.tmp"
tar -C "$RUN_DIR" -czf "$tmp" training_prefixes evaluation training_prefix_data_summary.csv training_prefix_data_summary.json
mv "$tmp" "$OUTPUT"
sha256sum "$OUTPUT" | tee "$OUTPUT.sha256"
tar -tzf "$OUTPUT"
echo "[ready] $OUTPUT"
