#!/usr/bin/env bash
# Package the measured report inputs as a self-contained Overleaf project.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/formal_NN_training/experiments/602_lstm_stride_live_inference"
RUN_DIR="${RUN_DIR:-$EXP/runs/602_gcc_stride_prefix_seed7}"
OUTPUT="${OUTPUT:-$RUN_DIR/602_stride_live_overleaf.zip}"
FORCE="${FORCE:-0}"
SOURCE_TEX="$EXP/report/602_stride_training_budget_live_inference.tex"
CONCLUSIONS="$RUN_DIR/report/generated_conclusions.tex"
PLOT_DIR="$RUN_DIR/plots"

usage() {
  cat <<'EOF'
Usage: package_overleaf_report.sh

Environment:
  RUN_DIR=PATH   Completed ignored experiment run directory.
  OUTPUT=PATH    Output Overleaf ZIP.
  FORCE=1        Replace an existing ZIP after validation.

The ZIP contains main.tex, report/generated_conclusions.tex, and exactly
fourteen PNG plots. It excludes checkpoints, logs, event streams, venvs, and
all other run artifacts.
EOF
}

[[ "${1:-}" != "-h" && "${1:-}" != "--help" ]] || { usage; exit 0; }
[[ -s "$SOURCE_TEX" ]] || { echo "[error] missing tracked report TeX" >&2; exit 2; }
[[ -s "$CONCLUSIONS" ]] || { echo "[error] missing generated conclusions: $CONCLUSIONS" >&2; exit 2; }
[[ -d "$PLOT_DIR" ]] || { echo "[error] missing plots directory: $PLOT_DIR" >&2; exit 2; }

mapfile -t PLOTS < <(find "$PLOT_DIR" -maxdepth 1 -type f -name '*.png' -size +0c -print | sort)
[[ "${#PLOTS[@]}" -eq 14 ]] || {
  echo "[error] expected 14 plots, found ${#PLOTS[@]}" >&2
  exit 2
}
if [[ -e "$OUTPUT" && "$FORCE" != 1 ]]; then
  echo "[error] archive exists; inspect it or set FORCE=1: $OUTPUT" >&2
  exit 3
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
project="$tmp/602_stride_live_overleaf"
mkdir -p "$project/plots" "$project/report"
cp "$SOURCE_TEX" "$project/main.tex"
cp "$CONCLUSIONS" "$project/report/generated_conclusions.tex"
for plot in "${PLOTS[@]}"; do
  cp "$plot" "$project/plots/"
done

python3 - "$project/main.tex" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
old = r"\providecommand{\RunDir}{../runs/602_gcc_stride_prefix_seed7}"
new = r"\providecommand{\RunDir}{.}"
if old not in text:
    raise SystemExit("[error] expected RunDir declaration not found")
path.write_text(text.replace(old, new, 1))
PY

mkdir -p "$(dirname "$OUTPUT")"
python3 - "$project" "$OUTPUT" <<'PY'
import sys
import zipfile
from pathlib import Path

project = Path(sys.argv[1])
output = Path(sys.argv[2])
temporary = output.with_suffix(output.suffix + ".tmp")
with zipfile.ZipFile(str(temporary), "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(project.rglob("*")):
        if path.is_file():
            archive.write(str(path), str(path.relative_to(project)))
temporary.replace(output)
PY

python3 - "$OUTPUT" <<'PY'
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as archive:
    names = archive.namelist()
required = {"main.tex", "report/generated_conclusions.tex"}
missing = required.difference(names)
plots = [name for name in names if name.startswith("plots/") and name.endswith(".png")]
if missing or len(plots) != 14:
    raise SystemExit("[error] invalid Overleaf archive")
print("PASS: Overleaf project has main.tex, conclusions, and 14 plots")
for name in names:
    print(name)
PY
echo "[ready] $OUTPUT"
