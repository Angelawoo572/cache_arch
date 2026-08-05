# 623 Stride — independent exact-PC LSTM v20

This is the active matched-input Stride experiment for
`623.xalancbmk_s-700B`. Normal Stride and the standalone NN receive exactly
the same source-visible `pc` and aligned `addr`. Captured Stride actions are
offline-normal replay entries and training labels only. The NN never receives
the normal tracker, confidence, last stride, candidate addresses, degree,
request rate, or action outcome at runtime.

Input revision: `stride_source_input_variable_delta_free_running_v9`  
Model revision: `pc_keyed_independent_rank_delta_v20`  
Default run: `623_offline_lstm_stride_independent_delta_v20_seed7`

## Why v20 replaces v19

The failed 623 sequence of hurdle/GMM, scalar free-running, and finally v19's
sampled `STOP/EMIT` plus ZigZag/LEB128 decoder treated one cache-line target as
a long stochastic token trajectory. A wrong early token changed later target
origins, while the loss branch used teacher prefixes. More capacity could not
repair that training/inference and loss/action mismatch.

v20 keeps fairness at the **input boundary**, not by reproducing the normal
algorithm inside the NN:

1. The lossless causal encoder contains raw PC/line bits, current and prior
   same-PC signed deltas, exact distinct-PC reuse distance, and two validity
   bits. Every value is derived only from the observed `pc+addr` chronology.
2. One single-layer LSTM is dynamically keyed by exact PC. There is no global
   branch and no copied finite Stride tracker. Ordinary LSTM forget gates can
   learn how reuse distance changes useful memory.
3. A natural-frequency gate uses unweighted cross-entropy. Only its initial
   bias is the log TRAIN zero/positive prior. Inference is raw argmax; there is
   no probability threshold or guard calibration.
4. Positive cardinality is learned by smooth-L1 on log teacher count and is
   decoded by deterministic rounding. Its bias starts at the TRAIN positive
   mean log-count to avoid an arbitrary initial request explosion. There is no
   normal degree cap or request budget.
5. Every teacher target is supervised directly relative to the current demand.
   A shared head receives a generic sinusoidal rank code; it never receives a
   previous teacher or predicted action.
6. Up to 255 most frequent TRAIN signed deltas (frequency, then signed-integer
   tie break) receive exact classes. One `OTHER` class predicts a continuous
   signed-log delta, giving unseen values a broad bounded approximation. Only
   vocabulary deltas are integer-exact; float32 `OTHER` coordinates use a
   rounded inverse and do not guarantee domain endpoints. The continuous coordinate receives an auxiliary
   loss at every teacher rank, even when the exact class is available, so it
   cannot be left untrained when TRAIN has at most 255 unique deltas.
   The 256 class biases start from add-one-smoothed TRAIN exact/OTHER
   frequencies; this is label-derived initialization, not a policy template.
   `255+OTHER` is a byte-sized NN output alphabet,
   not a page-offset table, same-page rule, stride template, or degree cap.
7. After every epoch, guard-only behavior chooses the checkpoint
   lexicographically by target F1, trigger F1, then absolute request-ratio
   error. No threshold is selected. Evaluation is decoded once after the
   checkpoint is fixed.
8. Learned counts are never clipped. Generic host-resource watchdogs abort the
   role before a replay is written if one callback exceeds 4,096 actions or
   the role exceeds 10,000,000 actions. This fail-closed mechanism does not
   materialize a smaller count and is not a neural degree cap.

The pinned points are `h16/p0` (11,620 parameters) and `h32/p1` (27,076
parameters). The torch-free source of truth is:

```bash
python3 formal_NN_training/experiments/623_offline_lstm_stride/python/model_contract.py
```

The seed-7 run ID also pins `epochs=10`, `chunk_len=1024`, gradient
accumulation over 16 chunks, and `learning_rate=0.002`. The notebook derives
these values from the contract rather than repeating them. Each output records
and server/analyzer validation re-hashes the trainer, model contract, and
shared `threshold_free_policy.py`; results from different source bytes cannot
silently pass as this run.
The pinned accelerator is an A100. The trainer sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing torch, enables strict torch
deterministic algorithms, and uses deterministic/non-benchmark cuDNN; the
small model also pins float32 matmul precision to `highest`. The notebook and
server verify the recorded device and flags.

## Relation to the completed 602 Stride experiment

Both experiments use only public PC/address input, exact-PC recurrent state,
natural chronological TBPTT, teacher actions as losses only, deterministic
direct addresses, and the same replay/accounting path. v20 intentionally does
not copy 602's single scalar signed-log delta head. On 623, repeated continuous
or tokenized decoders failed to align integer-address loss with replay action
quality. v20 therefore keeps common TRAIN deltas exact, retains a generic
continuous escape, supervises all ranks without autoregressive feedback, and
uses guard behavior only for checkpoint selection. The network still learns
patterns independently; no normal Stride action form is hardcoded.

## Reuse the v19 input byte-for-byte

Do not recollect. The raw streams, labels, split boundaries, chronology, and
replay transport are unchanged:

```bash
cd ~/cache

export EXP=formal_NN_training/experiments/623_offline_lstm_stride
export SOURCE_RUN=623_offline_lstm_stride_global_local_grammar_v19_seed7
export RUN_ID=623_offline_lstm_stride_independent_delta_v20_seed7
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

The reused `collection_manifest.json` describes the historical v9 collection.
The v20 model/output semantics are pinned by `data/stream_contract.json`, the
torch-free model contract, the notebook assertions, and `run_metadata.json`.

Run `colab/623_offline_lstm_stride_A100.ipynb` on one A100. It trains only the
two contract points with `chunk_len=1024`, accumulation over 16 chunks, and ten
epochs. The upload cell accepts exactly one complete
`$RUN_ID.colab_input.tar.gz`; multipart input is intentionally unsupported. It
rejects path traversal, links, and non-file/archive directory members before
extraction. Put the downloaded archive at:

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

Contract PASS proves matched inputs, train-only label use, deterministic model
semantics, and exact reachable-intersection replay accounting. It does not
claim that v20 beats normal Stride; IPC, target/trigger F1, coverage, selected
accuracy, timeliness, and traffic remain separate reported outcomes.
