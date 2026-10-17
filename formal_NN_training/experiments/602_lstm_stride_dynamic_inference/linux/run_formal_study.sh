#!/usr/bin/env bash
# Rebuild in a new isolated directory, verify correctness, then run real study.
set -euo pipefail
FORMAL_EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORMAL_PY="${FORMAL_PYTHON:-/home/qianruw/venvs/cache-few-nn-cpu/bin/python}"
FORMAL_ROOT="$(cd "$FORMAL_EXP/../../.." && pwd)"
FORMAL_RAW="$("$FORMAL_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["raw"])' "$FORMAL_EXP/formal_config.json")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$FORMAL_RAW"
FORMAL_ATTEMPT="$FORMAL_RAW/launcher_attempts/$(date +%Y%m%d_%H%M%S)"
for FORMAL_LOG in correctness.log training_identity_tests.log pipeline.log; do
  if [[ -e "$FORMAL_RAW/$FORMAL_LOG" ]]; then
    mkdir -p "$FORMAL_ATTEMPT"
    cp -p "$FORMAL_RAW/$FORMAL_LOG" "$FORMAL_ATTEMPT/$FORMAL_LOG"
  fi
done
cd "$FORMAL_ROOT"
if [[ ! -s "$FORMAL_RAW/build.json" ]]; then
  "$FORMAL_PY" "$FORMAL_EXP/formal.py" build > "$FORMAL_RAW/build_driver.log" 2>&1
fi
DYNAMIC602_SIMULATOR=/home/qianruw/cache/external/ChampSim "$FORMAL_PY" -m unittest discover -s "$FORMAL_EXP/validation" -p 'test_*.py' -v > "$FORMAL_RAW/correctness.log" 2>&1
"$FORMAL_PY" "$FORMAL_ROOT/formal_NN_training/experiments/602_lstm_stride_live_inference/validation/test_formal_training_identity.py" > "$FORMAL_RAW/training_identity_tests.log" 2>&1
"$FORMAL_PY" "$FORMAL_EXP/formal.py" pipeline > "$FORMAL_RAW/pipeline.log" 2>&1
