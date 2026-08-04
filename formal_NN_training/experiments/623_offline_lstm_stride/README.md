# 623 Stride — prior-corrected compact LSTM v17

This directory is the active matched-input Stride experiment for
`623.xalancbmk_s-700B`. Normal Stride and the standalone NN receive the same
source-visible `pc` and current aligned `addr`. Captured Stride actions are
supervised labels and the offline-normal replay; they are never runtime inputs.

## What 602 actually contributes

602's independent-student method is behavior cloning under a matched external
input contract. The normal policy supplies labels, but its actions, tracker
state, candidates, constants, and budget are absent at NN inference. 623 keeps
that method while using its own trace-specific input distribution and compact
exact-PC recurrent-state organization.

The NN input remains a lossless 64-bit PC plus 58-bit cache-line number. A
dynamic exact-PC map routes one single-layer LSTM state per observed PC without
copying Stride's fixed tracker capacity. Train, guard, and evaluation remain
chronological; guard is recurrent warm-up/audit only.

## v16 diagnosis and v17 correction

v16's root comparison passed, but its gate decoder was wrong for this class
prior. The weighted cross-entropy gives the zero and positive classes equal
aggregate training mass, then v16 applies raw-logit argmax. That argmax decides
under the artificial balanced prior:

- normal Stride requested 166,147 prefetches;
- v16 NNs requested 268,118--328,316, or 1.61--1.98 times normal;
- the best v16 IPC difference was only -0.000020 and is not evidence that the
  over-emitting policy is correct.

If weighted CE uses weights `w_y`, its learned scores are proportional to
`w_y p(y|x)`. v17 therefore decodes
`argmax(logit_y - log(w_y))`. The weights are computed from the training split
and saved in metadata, so this is deterministic and data-derived. It introduces
no selected threshold, degree cap, request budget, candidate table, page rule,
or guard tuning.

Everything else is unchanged: balanced gate training, positive log-count,
scalar signed-log direct delta, free-running emitted-coordinate feedback,
lossless input encoder, labels, split, and replay transport.

Input revision: `stride_source_input_variable_delta_free_running_v9`  
Model revision: `compact_pc_keyed_prior_corrected_hurdle_scalar_v17`  
Default run: `623_offline_lstm_stride_prior_corrected_hurdle_v17_seed7`

| Tag suffix | Hidden size | Parameters |
|---|---:|---:|
| `h8` | 8 | 1,860 |
| `h16` | 16 | 5,124 |
| `h32` | 32 | 15,876 |
| `h64` | 64 | 54,276 |
| `h128` | 128 | 198,660 |

The exact formula is `11H^2 + 144H + 4`.

## Reuse the validated input; do not recollect

Use the completed v16 input byte-for-byte under a new run ID:

```bash
cd ~/cache

export EXP=formal_NN_training/experiments/623_offline_lstm_stride
export SOURCE_RUN=623_offline_lstm_stride_compact_hurdle_v16_seed7
export RUN_ID=623_offline_lstm_stride_prior_corrected_hurdle_v17_seed7
export SOURCE_DIR="$EXP/runs/$SOURCE_RUN"
export RUN_DIR="$EXP/runs/$RUN_ID"

test -d "$SOURCE_DIR/colab_input"
test -s "$SOURCE_DIR/$SOURCE_RUN.colab_input.tar.gz"
test ! -e "$RUN_DIR"

mkdir -p "$RUN_DIR"
cp -a "$SOURCE_DIR/colab_input" "$RUN_DIR/colab_input"
cp -p "$SOURCE_DIR/$SOURCE_RUN.colab_input.tar.gz"   "$RUN_DIR/$RUN_ID.colab_input.tar.gz"

cmp "$SOURCE_DIR/$SOURCE_RUN.colab_input.tar.gz"   "$RUN_DIR/$RUN_ID.colab_input.tar.gz"
diff -qr "$SOURCE_DIR/colab_input" "$RUN_DIR/colab_input"

python3 "$EXP/python/validate_collected_inputs.py"   --input-dir "$RUN_DIR/colab_input"   --manifest-out "$RUN_DIR/colab_input/collection_manifest.json"
```

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
The analyzer still recognizes preserved v15/v16 outputs, but the defaults and
current contract are v17.
