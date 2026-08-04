# 623 Stride — natural-frequency compact LSTM v18

This directory is the active matched-input Stride experiment for
`623.xalancbmk_s-700B`. Normal Stride and the standalone NN receive the same
source-visible `pc` and current aligned `addr`. Captured Stride actions are
supervised labels and the offline-normal replay; they are never runtime inputs.

## Frozen experiment boundary

The NN input remains a lossless 64-bit PC plus 58-bit cache-line number. A
dynamic exact-PC map routes one single-layer LSTM state per observed PC without
copying Stride's fixed tracker capacity. Train, guard, and evaluation remain
chronological; guard is recurrent warm-up/audit only. The positive log-count,
scalar signed-log direct-delta decoder, and emitted-coordinate free-running
feedback are unchanged.

Normal actions schedule the supervised action ranks during training. At each
rank, however, the value fed back to the action decoder is its own prediction,
never the teacher delta. Inference uses the same self-action feedback.

## Why v17 did not solve the gate

v17 correctly removed the artificial balanced prior at decode, but the measured
result shows that post-hoc correction did not calibrate the learned positive
tail:

- all five v17 neural points requested more than normal Stride: 188,344--271,250
  requests versus 166,147, or about 1.13--1.63 times normal;
- false-positive trigger callbacks remained larger than false negatives;
- increasing hidden size from 8 to 128 did not produce monotonic IPC gains;
- every v17 point remained below offline Stride by 0.00001--0.00009 IPC.

This evidence rules out another capacity increase as the next controlled test.
v18 removes inverse-frequency weighting and the matching post-hoc correction.
The gate now uses natural-frequency, unweighted two-class cross-entropy. Its
bias is initialized to the log empirical zero/positive prior of the training
split, the model is moved to the selected device before that initialization,
and Adam is constructed only afterward. Inference deterministically takes the
raw two-logit argmax.

There is still no probability threshold, request budget, candidate table,
same-page rule, fixed page-offset class, Stride degree cap, or normal-policy
private state. v18 is a gate-objective test, not a claim that its IPC must win.

Input revision: `stride_source_input_variable_delta_free_running_v9`  
Model revision: `compact_pc_keyed_natural_hurdle_scalar_v18`  
Default run: `623_offline_lstm_stride_natural_hurdle_v18_seed7`

| Tag suffix | Hidden size | Parameters |
|---|---:|---:|
| `h8` | 8 | 1,860 |
| `h16` | 16 | 5,124 |
| `h32` | 32 | 15,876 |
| `h64` | 64 | 54,276 |
| `h128` | 128 | 198,660 |

The exact formula remains `11H^2 + 144H + 4`.

## Reuse the validated v17 input; do not recollect

The data, labels, split, chronology, and replay transport are intentionally
unchanged. Reuse the completed v17 input byte-for-byte under the v18 run ID:

```bash
cd ~/cache

export EXP=formal_NN_training/experiments/623_offline_lstm_stride
export SOURCE_RUN=623_offline_lstm_stride_prior_corrected_hurdle_v17_seed7
export RUN_ID=623_offline_lstm_stride_natural_hurdle_v18_seed7
export SOURCE_DIR="$EXP/runs/$SOURCE_RUN"
export RUN_DIR="$EXP/runs/$RUN_ID"

test -d "$SOURCE_DIR/colab_input"
test -s "$SOURCE_DIR/$SOURCE_RUN.colab_input.tar.gz"
test ! -e "$RUN_DIR"

mkdir -p "$RUN_DIR"
cp -a "$SOURCE_DIR/colab_input" "$RUN_DIR/colab_input"
cp -p "$SOURCE_DIR/$SOURCE_RUN.colab_input.tar.gz" \
  "$RUN_DIR/$RUN_ID.colab_input.tar.gz"

cmp "$SOURCE_DIR/$SOURCE_RUN.colab_input.tar.gz" \
  "$RUN_DIR/$RUN_ID.colab_input.tar.gz"
diff -qr "$SOURCE_DIR/colab_input" "$RUN_DIR/colab_input"

python3 "$EXP/python/validate_collected_inputs.py" \
  --input-dir "$RUN_DIR/colab_input" \
  --manifest-out "$RUN_DIR/colab_input/collection_manifest.json"
```

The reused `collection_manifest.json` records input-package provenance under
the unchanged v9 input revision. Its historical decoder wording is not the v18
model-output contract; `run_metadata.json` and `data/stream_contract.json` are.

Upload the renamed input archive to
`colab/623_offline_lstm_stride_A100.ipynb`. Put the downloaded output archive
at `$RUN_DIR/$RUN_ID.colab_output.tar.gz`, then run:

```bash
BUILD=0 RUN_ID="$RUN_ID" bash "$EXP/linux/launch_server.sh" replay
tail -F "$RUN_DIR/replay.nohup.log"

python3 "$EXP/python/check_matched_comparison.py" --run-id "$RUN_ID"
python3 "$EXP/python/diagnose_completed_run.py" --run-id "$RUN_ID"
```

Do not run `collect`, and do not launch `analyze` concurrently with replay.
The analyzer preserves v15, v16, and v17 metadata support; defaults and the
current contract are v18.
