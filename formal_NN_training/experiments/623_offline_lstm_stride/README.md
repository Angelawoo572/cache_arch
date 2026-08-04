# 623 Stride — chronological global/local action-grammar LSTM v19

This directory is the active matched-input Stride experiment for
`623.xalancbmk_s-700B`. Normal Stride and the standalone NN receive the same
source-visible `pc` and current aligned `addr`. Captured Stride actions are
supervised sequence labels and the offline-normal replay only; they never enter
the runtime encoder, main rollout state, sampler key, or inference decoder.
Teacher codec prefixes recurrently advance only the isolated loss-branch
likelihood state described below.

## What v19 changes

v18's per-PC-only encoder and separate hurdle / rounded positive count /
scalar signed-log delta heads did not reproduce Stride's actions even when IPC
rounded to the same six decimals. v19 changes the learning problem rather than
adding a selected threshold:

1. A global LSTM processes every callback in original chronology, so phase and
   intervening-PC activity remain visible.
2. A second LSTM is dynamically routed by exact PC. Its causal input adds
   lossless same-PC signed delta, same-PC reuse age, and a history-valid bit,
   all derived from the same `pc+addr` history.
3. A learned sigmoid validity gate softly controls how much PC-local context is
   fused with the global context. It does not reproduce Stride's 64-entry
   tracker, replacement rule, confidence, or degree.
4. A learned rank-wise `STOP/EMIT` grammar determines both zero requests and
   sequence length. There is no hurdle, Poisson count, rounded mean,
   probability threshold, request budget, or degree cap.
5. Each emitted target is an increment: rank 1 is relative to the current
   demand line and each later rank is relative to the prior emitted line. The
   exact signed 58-bit integer is ZigZag encoded and then generated as canonical
   LEB128. Small strides normally require one byte; the full address domain is
   representable in at most nine bytes. There is no GMM or scalar rounding.
6. Every categorical choice uses stateless event/rank/field-keyed inverse-CDF
   sampling. The key excludes model capacity and teacher values, giving strict
   common random numbers across the two sizes. A loss-only teacher-prefix
   branch computes the full canonical autoregressive codec NLL from the actual
   rank state/origin. Teacher tokens recurrently advance only that isolated
   likelihood state and cannot mutate the main rollout; main recurrent
   feedback and the next-rank origin always use the model's own hard sampled
   `STOP/EMIT`, payload, and continuation tokens.
7. The float64 inverse-CDF sampler fails closed if `STOP` has no representable
   53-bit interval. A nontermination watchdog derived from that grid precision
   raises without producing a replay; it never truncates a sequence or forces
   `STOP`, and is not a policy degree cap.

Input revision: `stride_source_input_variable_delta_free_running_v9`  
Model revision: `chronological_global_pc_local_stop_emit_leb128_v19`  
Default run: `623_offline_lstm_stride_global_local_grammar_v19_seed7`

The two pinned points are `h8/p0` and `h16/p1`, both below 10,000 trainable
parameters. Their tags, projection widths, exact counts, revisions, runtime
feature widths, and derived formula have one source of truth:

```bash
python3 formal_NN_training/experiments/623_offline_lstm_stride/python/model_contract.py
```

The formula uses `E=H/2` and derives the projection term from the runtime
feature widths. Metadata also reports the dynamic recurrent-state footprint:
one global `(h,c)` pair plus one local `(h,c)` pair per observed PC.

## Reuse v18 input byte-for-byte; do not recollect

The raw streams, captured labels, splits, chronology, hashes, and replay
transport are unchanged. Reuse the completed v18 package under the v19 run ID:

```bash
cd ~/cache

export EXP=formal_NN_training/experiments/623_offline_lstm_stride
export SOURCE_RUN=623_offline_lstm_stride_natural_hurdle_v18_seed7
export RUN_ID=623_offline_lstm_stride_global_local_grammar_v19_seed7
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
```

The reused `collection_manifest.json` intentionally retains the v9 input
revision and its historical decoder wording. The v19 output contract is pinned
by `data/stream_contract.json`, the Colab assertions, and each
`run_metadata.json`.

Run `colab/623_offline_lstm_stride_A100.ipynb` on one A100. It trains only the
two pinned points with `chunk_len=1024`, gradient accumulation over 16 chunks,
and ten epochs. Accumulated gradients are weighted by the exact number of
categorical atoms, so each optimizer window matches the global natural
sequence NLL. Put the downloaded archive at:

```text
$RUN_DIR/$RUN_ID.colab_output.tar.gz
```

Then replay and diagnose:

```bash
BUILD=0 RUN_ID="$RUN_ID" bash "$EXP/linux/launch_server.sh" replay
tail -F "$RUN_DIR/replay.nohup.log"

python3 "$EXP/python/check_matched_comparison.py" --run-id "$RUN_ID"
python3 "$EXP/python/diagnose_completed_run.py" --run-id "$RUN_ID"
```

Do not run `collect`, and do not launch `analyze` concurrently with replay.
PASS certifies input fairness, metadata, and replay accounting; IPC and action
quality still decide whether the redesign succeeds.
