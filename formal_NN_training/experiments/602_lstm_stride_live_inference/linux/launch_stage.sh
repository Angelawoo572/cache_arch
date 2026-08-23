#!/usr/bin/env bash
# Enforce h16-first, h8-second, offline-curve, then recommendation staging.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/formal_NN_training/experiments/602_lstm_stride_live_inference"
RUN_DIR="${RUN_DIR:-$EXP/runs/602_gcc_stride_prefix_seed7}"
EVAL="${EVAL:-$RUN_DIR/evaluation/602.gcc_s-734B.eval_stream.csv.gz}"
STAGE="${1:-status}"
FORCE="${FORCE:-0}"
PARITY_MAX_EVENTS="${PARITY_MAX_EVENTS:-0}"

usage() {
  cat <<'EOF'
Usage: launch_stage.sh stage0|stage1|stage2|stage3|status

stage0  h16 i20m export, validation, standalone parity, build, live smoke
stage1  h8 i20m through the identical infrastructure
stage2  full h8/h16 offline training/inference sweep (Colab/GPU)
stage3  aggregate offline results and recommend; does not launch live runs
status  print training/export/parity/live status table

Optional old anchors:
  OLD_H16_CHECKPOINT=/path/model.pt
  OLD_H8_CHECKPOINT=/path/model.pt

FORCE=1 is forwarded only to explicit rebuild/export/live operations.
EOF
}

[[ "$STAGE" != "-h" && "$STAGE" != "--help" ]] || { usage; exit 0; }

export_and_validate() {
  local hidden="$1" old_variable="$2"
  local point="$RUN_DIR/points/h$hidden/i20m/seed7"
  local checkpoint="$point/offline/model.pt"
  local export_dir="$point/export"
  local old_checkpoint
  old_checkpoint="$(printenv "$old_variable" 2>/dev/null || true)"
  [[ -s "$checkpoint" ]] || { echo "[error] missing h$hidden 20M checkpoint" >&2; exit 3; }
  export_args=(
    --checkpoint "$checkpoint"
    --run-metadata "$point/offline/run_metadata.json"
    --out-dir "$export_dir"
    --hidden-size "$hidden"
    --instruction-budget 20000000
    --seed 7
  )
  [[ "$FORCE" != 1 ]] || export_args+=(--force)
  if [[ ! -s "$export_dir/model.bin" || "$FORCE" == 1 ]]; then
    python3 "$EXP/python/export_live_model.py" "${export_args[@]}"
  fi
  validate_args=(
    --model-bin "$export_dir/model.bin"
    --metadata "$export_dir/model_metadata.json"
    --checkpoint "$checkpoint"
  )
  if [[ -n "$old_checkpoint" ]]; then
    validate_args+=(--reference-checkpoint "$old_checkpoint" --evaluation-stream "$EVAL")
  fi
  python3 "$EXP/python/validate_export.py" "${validate_args[@]}"
  RUN_DIR="$RUN_DIR" FORCE="$FORCE" bash "$EXP/linux/build_standalone_runtime.sh"
  runner="$RUN_DIR/bin/stride_lstm_standalone"
  python3 "$EXP/validation/compare_python_cpp_outputs.py" --cpp-runner "$runner" --work-dir "$RUN_DIR/parity/synthetic" --synthetic --output "$RUN_DIR/parity/synthetic/summary.json"
  recorded_args=(
    --cpp-runner "$runner"
    --work-dir "$point/parity/recorded"
    --model-bin "$export_dir/model.bin"
    --checkpoint "$checkpoint"
    --stream "$EVAL"
    --point-metadata "$point/point_metadata.json"
    --output "$point/parity/recorded/summary.json"
  )
  [[ "$PARITY_MAX_EVENTS" == 0 ]] || recorded_args+=(--max-events "$PARITY_MAX_EVENTS")
  python3 "$EXP/validation/compare_python_cpp_outputs.py" "${recorded_args[@]}"
  RUN_DIR="$RUN_DIR" FORCE="$FORCE" bash "$EXP/linux/build_live_champsim.sh"
  RUN_DIR="$RUN_DIR" HIDDEN_SIZES="$hidden" BUDGETS=i20m SEEDS=7 RUN_MODE=smoke STATE_MODE=parity WARMUP_INSTRUCTIONS=25000000 SIMULATION_INSTRUCTIONS=20000000 FORCE="$FORCE" bash "$EXP/linux/run_live_sweep.sh" run
}

case "$STAGE" in
  stage0)
    export_and_validate 16 OLD_H16_CHECKPOINT
    ;;
  stage1)
    export_and_validate 8 OLD_H8_CHECKPOINT
    ;;
  stage2)
    args=(--run-dir "$RUN_DIR" --evaluation-stream "$EVAL" --hidden-sizes 8,16 --budgets all --seeds 7 --device auto --resume)
    [[ "$FORCE" != 1 ]] || args+=(--force)
    python3 "$EXP/python/run_training_prefix_sweep.py" "${args[@]}"
    ;;
  stage3)
    python3 "$EXP/python/aggregate_offline_results.py" --run-dir "$RUN_DIR"
    python3 "$EXP/python/select_live_budgets.py" --offline-results "$RUN_DIR/offline_sweep_results.json" --output "$RUN_DIR/selected_live_budgets.json" --selection-config "$EXP/config/live_selection.json"
    echo "[recommendation only] inspect and approve config/live_selection.json"
    ;;
  status)
    bash "$EXP/linux/install_colab_output.sh" status
    bash "$EXP/linux/run_offline_sweep.sh" status || true
    bash "$EXP/linux/run_live_sweep.sh" status || true
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
