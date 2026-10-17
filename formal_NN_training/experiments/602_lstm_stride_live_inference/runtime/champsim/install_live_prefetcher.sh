#!/usr/bin/env bash
# Deterministically stage the authoritative live runtime into ChampSim.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
EXP="$ROOT/formal_NN_training/experiments/602_lstm_stride_live_inference"
CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
ACTION="${1:-status}"
FORCE="${FORCE:-0}"
MANIFEST="$CHAMP_DIR/.stride_lstm_live_install_manifest.json"

usage() {
  cat <<'EOF'
Usage: install_live_prefetcher.sh [status|install|restore]

Environment:
  CHAMP_DIR=PATH   ChampSim checkout.
  FORCE=1          Replace a conflicting installer-owned file, or restore it.

The installer does not change the submodule pointer, delete binaries, or run
git clean. It generates stride_lstm_live.l2c_pref from the committed
multi.l2c_pref template and copies only named runtime files.
EOF
}

[[ "$ACTION" != "-h" && "$ACTION" != "--help" ]] || { usage; exit 0; }
[[ "$ACTION" == "status" || "$ACTION" == "install" || "$ACTION" == "restore" ]] || {
  usage >&2
  exit 2
}
git -C "$CHAMP_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "[error] not a ChampSim checkout: $CHAMP_DIR" >&2
  exit 2
}

targets=(
  "inc/stride_lstm_model_loader.h"
  "inc/stride_lstm_runtime.h"
  "inc/stride_lstm_live.h"
  "prefetcher/stride_lstm_model_loader.cc"
  "prefetcher/stride_lstm_runtime.cc"
  "prefetcher/stride_lstm_live.cc"
  "prefetcher/stride_lstm_live.l2c_pref"
)
sources=(
  "$EXP/runtime/stride_lstm_model_loader.h"
  "$EXP/runtime/stride_lstm_runtime.h"
  "$EXP/runtime/champsim/stride_lstm_live.h"
  "$EXP/runtime/stride_lstm_model_loader.cc"
  "$EXP/runtime/stride_lstm_runtime.cc"
  "$EXP/runtime/champsim/stride_lstm_live.cc"
)

show_status() {
  git -C "$CHAMP_DIR" status --short
  for target in "${targets[@]}"; do
    if [[ -f "$CHAMP_DIR/$target" ]]; then
      echo "present $target"
    else
      echo "absent $target"
    fi
  done
}

generate_registry() {
  local output="$1"
  local template="$CHAMP_DIR/prefetcher/multi.l2c_pref"
  [[ -f "$template" ]] || { echo "[error] missing $template" >&2; exit 3; }
  git -C "$CHAMP_DIR" show HEAD:prefetcher/multi.l2c_pref |
    grep -Fq '#include "stride.h"' || {
      echo "[error] expected include context absent in committed template" >&2
      exit 3
    }
  git -C "$CHAMP_DIR" show HEAD:prefetcher/multi.l2c_pref |
    grep -Fq 'compare("stride")' || {
      echo "[error] expected registry context absent in committed template" >&2
      exit 3
    }
  local base
  base="$(mktemp)"
  git -C "$CHAMP_DIR" show HEAD:prefetcher/multi.l2c_pref > "$base"
  python3 - "$base" "$output" <<'PY'
from pathlib import Path
import sys

source, output = map(Path, sys.argv[1:])
text = source.read_text()
include_anchor = '#include "stride.h"\n'
if text.count(include_anchor) != 1:
    raise SystemExit("[error] non-unique stride include anchor")
text = text.replace(
    include_anchor,
    include_anchor + '#include "stride_lstm_live.h"\n',
    1,
)
registry_anchor = '''\t\telse if(!knob::l2c_prefetcher_types[index].compare("stride"))
\t\t{
\t\t\tcout << "adding L2C_PREFETCHER: Stride" << endl;
\t\t\tStridePrefetcher *pref_stride = new StridePrefetcher(knob::l2c_prefetcher_types[index]);
\t\t\tprefetchers.push_back(pref_stride);
\t\t}
'''
if text.count(registry_anchor) != 1:
    raise SystemExit("[error] non-unique stride registry anchor")
live = registry_anchor + '''\t\telse if(!knob::l2c_prefetcher_types[index].compare("stride_lstm_live"))
\t\t{
\t\t\tcout << "adding L2C_PREFETCHER: stride_lstm_live" << endl;
\t\t\tStrideLSTMLive *pref_live = new StrideLSTMLive(knob::l2c_prefetcher_types[index], this);
\t\t\tprefetchers.push_back(pref_live);
\t\t}
'''
text = text.replace(registry_anchor, live, 1)
output.write_text(text)
PY
  rm -f "$base"
}

if [[ "$ACTION" == "status" ]]; then
  show_status
  exit 0
fi

if [[ "$ACTION" == "restore" ]]; then
  if [[ ! -f "$MANIFEST" && "$FORCE" != 1 ]]; then
    echo "[error] install manifest absent; refusing broad cleanup" >&2
    exit 3
  fi
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  generate_registry "$tmp/stride_lstm_live.l2c_pref"
  if [[ "$FORCE" != 1 ]]; then
    for index in "${!sources[@]}"; do
      target="$CHAMP_DIR/${targets[$index]}"
      [[ ! -e "$target" ]] || cmp -s "${sources[$index]}" "$target" || {
        echo "[error] modified installed file; inspect or set FORCE=1: $target" >&2
        exit 3
      }
    done
    generated="$CHAMP_DIR/${targets[6]}"
    [[ ! -e "$generated" ]] || cmp -s "$tmp/stride_lstm_live.l2c_pref" "$generated" || {
      echo "[error] modified installed registry; inspect or set FORCE=1: $generated" >&2
      exit 3
    }
  fi
  python3 - "$CHAMP_DIR" "$MANIFEST" "${targets[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
targets = sys.argv[3:]
recorded = json.loads(manifest_path.read_text()).get("installed_files", [])
if sorted(recorded) != sorted(targets):
    raise SystemExit("[error] install manifest target list changed")
for relative in targets:
    path = root / relative
    if path.exists():
        path.unlink()
if manifest_path.exists():
    manifest_path.unlink()
print("[restored] removed only installer-owned live source files")
PY
  show_status
  exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
generate_registry "$tmp/stride_lstm_live.l2c_pref"

for index in "${!sources[@]}"; do
  source="${sources[$index]}"
  target="$CHAMP_DIR/${targets[$index]}"
  if [[ -e "$target" ]] && ! cmp -s "$source" "$target" && [[ "$FORCE" != 1 ]]; then
    echo "[error] conflicting target; inspect or set FORCE=1: $target" >&2
    exit 4
  fi
  install -m 0644 "$source" "$target"
done
generated="$CHAMP_DIR/${targets[6]}"
if [[ -e "$generated" ]] && ! cmp -s "$tmp/stride_lstm_live.l2c_pref" "$generated" && [[ "$FORCE" != 1 ]]; then
  echo "[error] conflicting generated registry: $generated" >&2
  exit 4
fi
install -m 0644 "$tmp/stride_lstm_live.l2c_pref" "$generated"

python3 - "$CHAMP_DIR" "$MANIFEST" "${targets[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = Path(sys.argv[2])
targets = sys.argv[3:]
payload = {
    "schema_version": 1,
    "installed_files": targets,
}
manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print("[installed] deterministic stride_lstm_live sources")
PY
show_status
