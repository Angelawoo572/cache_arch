#!/usr/bin/env bash
# Package the measured report inputs as a self-contained Overleaf project.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/formal_NN_training/experiments/602_lstm_stride_live_inference"
RUN_DIR="${RUN_DIR:-$EXP/runs/602_gcc_stride_prefix_seed7}"
OUTPUT="${OUTPUT:-$RUN_DIR/602_stride_offline_sufficiency_overleaf.zip}"
FORCE="${FORCE:-0}"
SOURCE_TEX="$EXP/report/602_stride_training_budget_live_inference.tex"
REPORT_DIR="$RUN_DIR/report"

usage() {
  cat <<'EOF'
Usage: package_overleaf_report.sh

Environment:
  RUN_DIR=PATH   Completed ignored experiment run directory.
  OUTPUT=PATH    Output Overleaf ZIP.
  FORCE=1        Replace an existing ZIP after validation.

The ZIP contains main.tex and the generated report/*.tex inputs, including
three PGFPlots figure fragments. Overleaf renders the figures. Sacramento does
not need matplotlib or pdflatex. The archive excludes checkpoints, logs, event
streams, replay lists, venvs, binaries, and all other run artifacts.
EOF
}

[[ "${1:-}" != "-h" && "${1:-}" != "--help" ]] || { usage; exit 0; }
[[ -s "$SOURCE_TEX" ]] || { echo "[error] missing tracked report TeX" >&2; exit 2; }
REQUIRED_TEX=(
  generated_conclusions.tex
  generated_h8_key_budgets.tex
  generated_h16_key_budgets.tex
  generated_behavior_differences.tex
  generated_live_validation.tex
  offline_fairness_report.tex
  regression_20m_report.tex
  generated_figure1_offline_ipc.tex
  generated_figure2_same_h_differences.tex
  generated_figure3_training_supervision.tex
)
for name in "${REQUIRED_TEX[@]}"; do
  [[ -s "$REPORT_DIR/$name" ]] || {
    echo "[error] missing generated report input: $REPORT_DIR/$name" >&2
    exit 2
  }
done
if [[ -e "$OUTPUT" && "$FORCE" != 1 ]]; then
  echo "[error] archive exists; inspect it or set FORCE=1: $OUTPUT" >&2
  exit 3
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
project="$tmp/602_stride_offline_sufficiency_overleaf"
mkdir -p "$project/report"
cp "$SOURCE_TEX" "$project/main.tex"
for name in "${REQUIRED_TEX[@]}"; do
  cp "$REPORT_DIR/$name" "$project/report/$name"
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
required = {
    "main.tex",
    "report/generated_conclusions.tex",
    "report/generated_h8_key_budgets.tex",
    "report/generated_h16_key_budgets.tex",
    "report/generated_behavior_differences.tex",
    "report/generated_live_validation.tex",
    "report/offline_fairness_report.tex",
    "report/regression_20m_report.tex",
    "report/generated_figure1_offline_ipc.tex",
    "report/generated_figure2_same_h_differences.tex",
    "report/generated_figure3_training_supervision.tex",
}
missing = required.difference(names)
forbidden_suffixes = (
    ".png", ".pdf", ".log", ".csv", ".json", ".bin", ".pt", ".gz"
)
forbidden = [name for name in names if name.endswith(forbidden_suffixes)]
if missing or forbidden:
    raise SystemExit("[error] invalid Overleaf archive")
print("PASS: LaTeX-only Overleaf project has all required report inputs")
for name in names:
    print(name)
PY
echo "[ready] $OUTPUT"
