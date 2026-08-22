#!/usr/bin/env bash
# Validate and install the Colab output archive into the ignored run tree.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/formal_NN_training/experiments/602_lstm_stride_live_inference"
RUN_DIR="${RUN_DIR:-$EXP/runs/602_gcc_stride_prefix_seed7}"
ARCHIVE="${ARCHIVE:-$RUN_DIR/incoming/602_stride_live_colab_output.tar.gz}"
SHA_FILE="${SHA_FILE:-$ARCHIVE.sha256}"
FORCE="${FORCE:-0}"
ACTION="${1:-install}"

usage() {
  cat <<'EOF'
Usage: install_colab_output.sh [install|status]

Environment:
  ARCHIVE=PATH    602_stride_live_colab_output.tar.gz.
  SHA_FILE=PATH   Optional sidecar with SHA256 as its first token.
  RUN_DIR=PATH    Existing ignored Sacramento run directory.
  FORCE=1         Archive an existing points tree before replacement.

The installer rejects absolute paths, '..', links, missing h8/h16 20M anchors,
and parameter-count mismatches.
EOF
}

[[ "$ACTION" != "-h" && "$ACTION" != "--help" ]] || { usage; exit 0; }
[[ "$ACTION" == "install" || "$ACTION" == "status" ]] || { usage >&2; exit 2; }

show_status() {
  python3 - "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for path in root.glob("points/h*/*/seed*/point_metadata.json"):
    item = json.loads(path.read_text())
    checkpoint = path.parent / "offline/model.pt"
    rows.append((
        item.get("hidden_size"), item.get("budget_tag"), item.get("seed"),
        item.get("parameter_count"), item.get("status"),
        "yes" if checkpoint.is_file() else "no",
    ))
print("hidden budget seed parameters status checkpoint")
for row in sorted(rows):
    print(*row)
PY
}

if [[ "$ACTION" == "status" ]]; then
  show_status
  exit 0
fi
[[ -s "$ARCHIVE" ]] || { echo "[error] missing archive: $ARCHIVE" >&2; exit 3; }
if [[ -s "$SHA_FILE" ]]; then
  expected="$(awk 'NR==1 {print $1}' "$SHA_FILE")"
  observed="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
  [[ "$expected" == "$observed" ]] || {
    echo "[error] Colab archive SHA256 mismatch" >&2
    exit 3
  }
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
python3 - "$ARCHIVE" "$tmp" <<'PY'
import json
import tarfile
import sys
from pathlib import Path, PurePosixPath

archive = Path(sys.argv[1])
destination = Path(sys.argv[2])
with tarfile.open(archive, "r:gz") as handle:
    for member in handle.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
            raise SystemExit("unsafe archive member: {}".format(member.name))
    handle.extractall(destination)
points = destination / "points"
if not points.is_dir():
    raise SystemExit("archive has no points/ tree")
anchors = {}
for path in points.glob("h*/i20m/seed7/point_metadata.json"):
    row = json.loads(path.read_text())
    anchors[row.get("hidden_size")] = row
for hidden, parameters in ((8, 1908), (16, 5220)):
    row = anchors.get(hidden)
    if row is None:
        raise SystemExit("missing h{} i20m seed7 anchor metadata".format(hidden))
    if row.get("parameter_count") != parameters:
        raise SystemExit("h{} parameter mismatch".format(hidden))
    checkpoint = path = points / "h{}".format(hidden) / "i20m/seed7/offline/model.pt"
    if not checkpoint.is_file():
        raise SystemExit("missing anchor checkpoint {}".format(checkpoint))
print("PASS: h8/h16 20M anchor metadata and checkpoints present")
PY

mkdir -p "$RUN_DIR"
if [[ -e "$RUN_DIR/points" && "$FORCE" != 1 ]]; then
  echo "[error] points tree exists; inspect it or set FORCE=1" >&2
  exit 4
fi
if [[ -e "$RUN_DIR/points" ]]; then
  archive_dir="$RUN_DIR/replaced/$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$archive_dir"
  mv "$RUN_DIR/points" "$archive_dir/"
fi
mv "$tmp/points" "$RUN_DIR/points"
if [[ -f "$tmp/training_sweep_status.json" ]]; then
  cp "$tmp/training_sweep_status.json" "$RUN_DIR/training_sweep_status.json"
fi
sha256sum "$ARCHIVE" > "$RUN_DIR/colab_output_archive.sha256"
show_status
echo "[installed] $RUN_DIR/points"
