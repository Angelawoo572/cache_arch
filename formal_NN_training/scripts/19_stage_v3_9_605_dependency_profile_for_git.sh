#!/usr/bin/env bash
# Copy the curated v3.9 605 dependency-profile payload into the Git-tracked
# Colab-input location. Do NOT stage the duplicate tar.gz, build log, raw trace,
# or whole artifacts directory.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/cache}"
SRC_DIR="${SRC_DIR:-$REPO_ROOT/formal_NN_training/artifacts/v3_9_dependency_sidecars}"
DST_DIR="$REPO_ROOT/formal_NN_training/data/upload/v3_9_dependency_sidecars"
PROFILE="605.mcf_s-994B.v3_9_dependency_profile.csv.gz"
META="605.mcf_s-994B.v3_9_dependency_profile.json"

[[ -d "$REPO_ROOT/.git" ]] || { echo "[error] REPO_ROOT is not a cache repo: $REPO_ROOT" >&2; exit 2; }
[[ -s "$SRC_DIR/$PROFILE" ]] || { echo "[error] missing $SRC_DIR/$PROFILE" >&2; exit 2; }
[[ -s "$SRC_DIR/$META" ]] || { echo "[error] missing $SRC_DIR/$META" >&2; exit 2; }

mkdir -p "$DST_DIR"
cp -f "$SRC_DIR/$PROFILE" "$DST_DIR/$PROFILE"
cp -f "$SRC_DIR/$META" "$DST_DIR/$META"

python3 - "$DST_DIR/$META" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    meta = json.load(handle)
assert meta.get("schema") == "v3_9_pc_static_dependency_profile", meta
assert meta.get("profile_scope") == "raw-trace training prefix only", meta
assert meta.get("uses_oracle_alignment") is False, meta
print("[verified] unique_pcs={:,}, dependency_pcs={:,}, profile_records={:,}".format(
    int(meta["unique_pcs"]),
    int(meta["pcs_with_dependency_observations"]),
    int(meta["profile_records"]),
))
PY

cd "$REPO_ROOT"
if git check-ignore -q "$DST_DIR/$PROFILE"; then
  echo "[error] profile is still ignored; pull the latest .gitignore" >&2
  exit 3
fi

git status --short \
  formal_NN_training/data/upload/v3_9_dependency_sidecars \
  .gitignore

echo "[next] git add .gitignore formal_NN_training/scripts/19_stage_v3_9_605_dependency_profile_for_git.sh \\
  formal_NN_training/data/upload/v3_9_dependency_sidecars/$PROFILE \\
  formal_NN_training/data/upload/v3_9_dependency_sidecars/$META"
