#!/bin/bash
# Sacramento entry point. Every protocol path comes from its saved config.
set -euo pipefail
experiment_dir="$(cd -- "$(dirname -- "$0")/.." && pwd)"
repository_dir="$(cd -- "$experiment_dir/../../.." && pwd)"
python_bin=/home/qianruw/venvs/cache-few-nn-cpu/bin/python
config_path="$experiment_dir/config.json"
config_value() {
    "$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$config_path" "$1"
}
raw_dir="$(config_value raw)"
runtime_dir="$(config_value runtime)"
analysis_dir="$(config_value analysis)"
protocol="$(config_value protocol)"
export TMPDIR="$runtime_dir/tmp"
export PYTHONPYCACHEPREFIX="$runtime_dir/pycache"
export MPLCONFIGDIR="$runtime_dir/mpl"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$TMPDIR" "$PYTHONPYCACHEPREFIX" "$MPLCONFIGDIR" "$raw_dir"
cd -- "$repository_dir"
exec 9>"$raw_dir/controller.lock"
flock -n 9 || { echo 'This protocol already has an active controller.' >&2; exit 1; }
export FOUR602_EXPERIMENT_DIR="$experiment_dir"
"$python_bin" - <<'PYTHON'
import importlib.util,os
from pathlib import Path
p=Path(os.environ['FOUR602_EXPERIMENT_DIR'])/'formal.py'
s=importlib.util.spec_from_file_location('four_formal',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
if not (m.RAW/'build.json').exists():m.build()
else:m.current_build()
if not (m.RAW/'correctness_gate.json').exists():m.certify()
else:m.require_correctness()
PYTHON
"$python_bin" "$experiment_dir/formal.py" validation
"$python_bin" "$experiment_dir/formal.py" select
"$python_bin" "$experiment_dir/formal.py" test
"$python_bin" "$experiment_dir/analyze.py" --raw "$raw_dir" --out "$analysis_dir" --route-manifest "$experiment_dir/route_manifest.json"
export FOUR602_ANALYSIS_DIR="$analysis_dir" FOUR602_RAW_DIR="$raw_dir" FOUR602_PROTOCOL="$protocol"
"$python_bin" - <<'PYTHON'
import json,os
from pathlib import Path
p=Path(os.environ['FOUR602_RAW_DIR'])/'latest_analysis.json'
p.write_text(json.dumps(dict(status='complete',analysis=os.environ['FOUR602_ANALYSIS_DIR'],
    protocol=os.environ['FOUR602_PROTOCOL'],expected_validation_runs=8,expected_test_runs=56),indent=2)+'\n')
PYTHON
