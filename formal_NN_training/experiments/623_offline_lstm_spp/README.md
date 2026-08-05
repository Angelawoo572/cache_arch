# 623 SPP — global LSTM with rankwise STOP/EMIT v21

This is the active matched-input SPP experiment for
`623.xalancbmk_s-700B`.  It is a clean redesign after v15–v20 rather than a
threshold patch.  The neural policy sees exactly the source-visible callback
stream already captured for v18, in chronological order:

- `DEMAND(invoke_prefetcher.addr)`
- `CACHE_FILL(cache_fill.evicted_addr)`

Each address is encoded losslessly as 58 cache-line bits and callback kind is
one bit.  PC is replay transport only.  Source-SPP targets, request counts,
fill levels, thresholds, tables, confidence, candidate actions, and private
state are not runtime inputs.  The recorded fill callbacks came from the
source-SPP run, so the scientific claim is a matched-input **open-loop**
comparison, not closed-loop live NN execution.

## Why v15–v20 failed

The failures were objective/decoder mismatches, not evidence that 623 needed
an SPP-shaped neural implementation.

- v15's keyed stochastic fill model improved some offline target statistics,
  but the best IPC was `0.353270` versus source SPP `0.353900`; h32 raised
  selected accuracy and timeliness while worsening miss rate and IPC.
- v16A collapsed the rare L2 lifecycle in four of five capacities.
- v17 over-produced L2 fills (3.56–5.31% versus about 2% in the teacher),
  duplicated later ranks, and peaked at `0.352960`.
- v18's separate trigger gate and count objective decoded almost every
  callback as positive but usually emitted one action.
- v19 serialized actions through autoregressive STOP/EMIT and LEB128 bytes.
  A prefix error corrupted the remaining address/action sequence.
- v20 removed that byte grammar, but still split the same learned decision
  into a gate and rounded log-count.  Those heads could disagree, and keyed
  fill sampling made checkpoint comparisons noisier than necessary.

v21 replaces the gate/count pair; it does not tune either one.

## v21 architecture and objectives

1. A single-layer global chronological LSTM consumes the complete 59-bit
   callback sequence.  Hidden sizes 8, 16, 32, 64, and 128 are all reported.
2. For rank `r`, the callback state and a generic four-component sinusoidal
   rank code feed an independent two-class `STOP/EMIT` head.  A callback with
   `N` teacher actions supervises `EMIT` at ranks `0..N-1` and one terminal
   `STOP` at rank `N`, including `STOP` at rank zero for an empty callback.
3. STOP/EMIT uses natural-frequency, unweighted cross-entropy.  The TRAIN
   token distribution contains one STOP per demand and one EMIT per action,
   so this is not an extremely sparse positive-label problem.  Inference uses
   categorical argmax at each rank and ends only when the model emits STOP.
4. Delta and fill losses exist only at teacher EMIT ranks.  Each delta is
   relative to the **current demand line**, never a preceding teacher or
   predicted action.  There is no autoregressive action/byte grammar.
5. The exact delta vocabulary is learned from TRAIN labels only: up to 255
   most-frequent signed deltas with signed-value tie breaking, plus one
   `OTHER` class.  A signed-log auxiliary is trained at every EMIT rank and
   supplies the bounded approximate value for `OTHER`.
6. Fill is conditioned on rank, delta class, and delta value.  During training
   it uses the teacher class/value and inverse-frequency TRAIN fill CE.  During
   inference it uses the actually decoded class/value, adds the log TRAIN
   natural prior to undo the inverse-frequency training reweighting, and takes
   deterministic argmax.  This paired correction is data-derived; it is not a
   threshold.  There is no keyed sampling, decoder seed, or fill cutoff.
7. Self targets and duplicate outputs are preserved.  ChampSim replay, queue
   merging, issue, delay, fill, usefulness, and eviction determine their
   effect; the offline decoder does not clean them up.
8. A fail-closed resource watchdog protects output materialization.  If an
   emitted sequence never reaches STOP, the run aborts without writing a
   replay; it never truncates, injects STOP, or acts as a neural degree cap.

Checkpoint selection is a strict lexicographic comparison, with no mean or
composite score:

1. joint `(target, fill)` action F1
2. target F1
3. L2 joint F1
4. trigger F1
5. exact-count rate
6. fill accuracy on matched targets
7. lower normalized TRAIN loss
8. earlier epoch

Evaluation is decoded once after the guard checkpoint is frozen.

There is no same-page inference rule, page dictionary, SPP
signature/pattern table, captured candidate list, probability cutoff, source
SPP threshold, fixed action count, neural degree cap, separate gate/count
regression, action GRU, or autoregressive byte/token prefix.

Input revision:
`spp_source_input_variable_delta_fill_feedback_free_running_v11`  
Model revision: `global_chronological_lstm_rank_stop_emit_v21`  
Decoder revision: `deterministic_stop_emit_vocab_other_fill_map_v21`  
Operation: `train-v21`  
Default run: `623_offline_lstm_spp_stop_emit_vocab_v21_seed7`

| Tag | Pair | H | Maximum parameters (255 exact deltas) |
|---|---|---:|---:|
| `stop_emit_vocab_spp_lstm_h8` | p0 | 8 | 4,551 |
| `stop_emit_vocab_spp_lstm_h16` | p1 | 16 | 8,951 |
| `stop_emit_vocab_spp_lstm_h32` | p2 | 32 | 22,239 |
| `stop_emit_vocab_spp_lstm_h64` | p3 | 64 | 62,639 |
| `stop_emit_vocab_spp_lstm_h128` | p4 | 128 | 198,735 |

The realized parameter count depends on TRAIN vocabulary size and is recorded
in each `run_metadata.json`.  The exact formula is
`9H² + 78H + 15 + (V+1)(H+1+E) + 2E`, where `E=max(4,H//4)` and
`0<V<=255`.  `python/model_contract.py --json` is the stable machine-readable
source for points, tags, revisions, objectives, and the formula.

The pinned optimization contract remains seed 7, 10 epochs, chronological
chunk length 1,024, 16 accumulated chunks per optimizer step, and Adam
learning rate 0.002.  v21 decoding is deterministic and has no decoder seed.

The pinned accelerator is an NVIDIA A100.  The trainer sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing Torch, disables cuDNN
benchmarking, enables cuDNN deterministic mode, calls strict
`torch.use_deterministic_algorithms(True)`, and pins float32 matmul precision
to `highest`.  Missing deterministic support or a non-A100 device aborts.

## What is reused from 602, and what differs

Like 602 SPP, v21 uses one global causal recurrent history, only the normal
policy's public inputs, teacher actions only as output supervision,
chronological TBPTT, a fresh train→guard→eval inference history, and the same
fill-preserving list replay/system metrics.  The complete H8–H128 sweep is
also retained because 602 capacity results were non-monotone.

Unlike 602's four-component continuous delta mixture, autoregressive
own-action feedback, and modal fill, v21 uses a TRAIN-derived exact categorical
delta vocabulary plus an `OTHER` signed-log auxiliary, independent rank
conditioning, learned terminal STOP, and target-conditioned deterministic
fill.  These differences address the observed 623 failures: hard action
decoding did not agree with continuous/count objectives, autoregressive
prefixes amplified errors, and majority fill behavior hid L2 collapse.

## Reuse the v18 input byte-for-byte

Do not recollect.  On Sacramento:

```bash
cd ~/cache

export EXP=formal_NN_training/experiments/623_offline_lstm_spp
export SOURCE_RUN=623_offline_lstm_spp_hard_distinct_v18_seed7
export RUN_ID=623_offline_lstm_spp_stop_emit_vocab_v21_seed7
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

python3 formal_NN_training/common/split_colab_archive.py split \
  "$RUN_DIR/$RUN_ID.colab_input.tar.gz" --output-dir "$RUN_DIR" \
  --max-part-mib 90 --overwrite
python3 formal_NN_training/common/split_colab_archive.py verify \
  "$RUN_DIR/$RUN_ID.colab_input.tar.gz.parts.json" --parts-dir "$RUN_DIR"
```

The copied `collection_manifest.json` and `spp_source_contract.json` are
historical input/source provenance.  Any legacy decoder description inside
that immutable package does not define v21; the current contract is pinned by
`data/stream_contract.json`, `python/model_contract.py`, and each model's
`run_metadata.json`.

In `colab/623_offline_lstm_spp_A100.ipynb`, select the input `.parts.json`
manifest and every numbered part together.  The notebook persists them to the
run-specific Google Drive directory, verifies and rejoins them, then performs
safe extraction.  Large output is split the same way.  Copy either the single
output archive or its manifest and every part into `$RUN_DIR`; `run_server.sh`
rejoins verified multipart output automatically before replay.  Do not run
`collect`, do not pass a parent checkpoint, and do not launch analysis while
replay is still writing results.

```bash
BUILD=1 FORCE=0 RESET_PATCH=0 JOBS=8 RUN_ID="$RUN_ID" \
  bash "$EXP/linux/launch_server.sh" replay
tail -F "$RUN_DIR/replay.nohup.log"

python3 "$EXP/python/diagnose_completed_run.py" --run-id "$RUN_ID"
python3 "$EXP/python/check_matched_comparison.py" --run-id "$RUN_ID"
```

A root `PASS` validates the v21 input, metadata, replay, and accounting
contracts.  It does not assert an IPC win; target/fill F1, request count,
miss rate, coverage, timeliness, traffic, and IPC remain separate outcomes.
