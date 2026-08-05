# 623 Stride — raw exact-PC rank STOP/EMIT LSTM v21

This is the active matched-input Stride experiment for
`623.xalancbmk_s-700B`. Normal Stride and the standalone NN receive the same
source-visible `pc` and aligned `addr`. Captured Stride actions are
offline-normal replay entries and supervised labels only. They are never NN
runtime inputs, candidates, prefixes, budgets, degree hints, or templates.

Input revision: `stride_source_input_variable_delta_free_running_v9`  
Model revision: `pc_keyed_raw_rank_stop_emit_v21`  
Decoder revision: `deterministic_rank_stop_emit_train_vocab_v21`  
Default run: `623_offline_lstm_stride_rank_stop_emit_v21_seed7`

## The v21 choice

v21 resets the model around the actual learning problem instead of encoding a
partial version of normal Stride.

1. The runtime tensor is exactly 122 lossless raw bits: `pc64+line58`. There is
   no supplied same-PC delta, prior delta, reuse distance, validity flag,
   tracker state, confidence, or future/teacher-derived feature. The recurrent
   network must learn any useful history representation itself.
2. A single-layer LSTM is dynamically keyed by exact PC. Chronological TBPTT
   carries one `(h,c)` pair per observed PC and detaches it only at chunk
   boundaries. There is no global recurrent branch and no finite Stride table
   copied into the model.
3. At every rank, a shared binary head independently chooses `STOP` or `EMIT`.
   A teacher sequence with `k` actions contributes `k` EMIT decisions and one
   terminal STOP. There is no separate global gate, log-count head, or decoded
   count rounding: cardinality is simply the number of EMIT decisions before
   the first STOP.
4. STOP/EMIT cross-entropy weights are computed only from TRAIN labels as
   `N/(2*N_class)`. Therefore all TRAIN STOP labels and all TRAIN EMIT labels
   have equal aggregate loss mass. These are data-derived loss weights, not a
   hand-chosen probability threshold or operating point. The decision-head
   bias starts at zero.
5. Each emitted rank predicts a signed cache-line delta relative to the
   **current demand**, never relative to a teacher or predicted earlier action.
   The decoder receives only the LSTM context and a generic eight-value
   sinusoidal rank code, so later ranks have no teacher-forcing/free-running
   prefix mismatch.
6. Up to 255 most frequent TRAIN deltas, ordered by descending frequency and
   then signed-integer value, receive exact categorical rows. One final dynamic
   `OTHER` row predicts a rounded inverse signed-log coordinate. The coordinate
   receives auxiliary smooth-L1 loss on every emitted teacher rank, including
   exact-class ranks. This is an output alphabet learned from labels, not a
   page-offset table, same-page rule, stride template, or degree cap.
7. Inference is a deterministic argmax rank loop until STOP. It uses no
   sampling, probability threshold, normal request rate, action budget, or
   learned-count clipping. Generic host-resource watchdogs abort the whole role
   before replay if a callback would exceed 4,096 actions or a role would
   exceed 10,000,000. They never convert EMIT to STOP or accept a truncated
   prefix, so they are fail-closed resource checks rather than neural policy
   caps.
8. After every epoch, guard-only behavior selects a checkpoint
   lexicographically by target F1, trigger F1, count exact-match rate, negative
   absolute request-ratio error, negative TRAIN loss, and finally earlier
   epoch. Evaluation is decoded once after the checkpoint is fixed. No
   threshold is calibrated on guard or evaluation.

## What is reused from completed 602 Stride

The sound parts of 602 remain: the fair boundary is public PC/address input;
state is routed by exact PC; training and inference use natural chronology;
teacher actions enter losses only; outputs are direct deterministic addresses;
and replay/accounting compares reachable matched callbacks.

The cardinality and target decoder intentionally differ. Completed 602 had a
sparse trigger problem for which a balanced hurdle decision plus a positive
count objective worked, and its delta decoder could exploit the observed 602
distribution. On 623, the earlier global decision/count and stochastic/token
variants repeatedly separated “should emit,” “how many,” and “which integer
addresses” into error-prone stages. v21 instead trains the exact sequential
decision that inference executes: each action is EMIT, every sequence ends in
STOP, and each emitted rank has a directly supervised integer-aligned delta.
It retains 602's raw-input/exact-PC principle while removing engineered
same-PC history that would bias the NN toward a manually supplied Stride-like
representation.

## Sweep and parameter accounting

The contract sweep is `h8/p0`, `h16/p1`, `h32/p2`, `h64/p3`, and `h128/p4`.
Because `C = |TRAIN exact-delta vocabulary| + 1 OTHER`, the realized output
head and parameter count are data-dependent:

```text
parameters(H,C) = 8*H^2 + (122+8+C+13)*H + C + 3
```

Before TRAIN is loaded, the contract reports only `maximum_parameter_count` at
`C=256`: 3,963; 8,691; 21,219; 58,563; and 182,403 parameters respectively.
Every run records realized `C`, realized parameter count, the formula result,
and the point maximum. Validation must require exact agreement with realized
`C` and `realized <= maximum`; it must not assume all data sets realize 256
classes.

The torch-free, Python-3.6-compatible source of truth and self-test are:

```bash
python3 formal_NN_training/experiments/623_offline_lstm_stride/python/model_contract.py \
  --describe-model-points
python3 formal_NN_training/experiments/623_offline_lstm_stride/python/model_contract.py \
  --self-test
python3 formal_NN_training/experiments/623_offline_lstm_stride/python/train_and_offline_infer.py \
  --self-test
```

The run ID pins seed 7, 10 epochs, `chunk_len=1024`, gradient accumulation over
16 chronological chunks, and learning rate `0.002`. The A100 determinism
contract pins `CUBLAS_WORKSPACE_CONFIG=:4096:8`, strict deterministic torch
algorithms, deterministic/non-benchmark cuDNN, and float32 matmul precision
`highest`. Output metadata re-hashes the trainer, model contract, and shared
`threshold_free_policy.py`.

## Reuse the v19 input byte-for-byte

Do not recollect. The raw streams, labels, split boundaries, chronology, and
replay transport are unchanged:

```bash
cd ~/cache

export EXP=formal_NN_training/experiments/623_offline_lstm_stride
export SOURCE_RUN=623_offline_lstm_stride_global_local_grammar_v19_seed7
export RUN_ID=623_offline_lstm_stride_rank_stop_emit_v21_seed7
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

The reused `collection_manifest.json` correctly describes the historical v9
collection. v21 model/output semantics are pinned by
`data/stream_contract.json`, `python/model_contract.py`, the trainer metadata,
and their recorded hashes. Run archives and large data remain outside GitHub.

Run `colab/623_offline_lstm_stride_A100.ipynb` on one A100 after its contract
cell reports all five v21 points. In its upload chooser, select the input
`.parts.json` manifest and every numbered part together. The notebook persists
them under the run-specific Google Drive directory, verifies every part and
the whole archive, rejoins it, and safely extracts it. Large outputs use the
same at-most-90-MiB manifest/part format. Copy either the single output archive
or its manifest and every part into `$RUN_DIR`; `run_server.sh` rejoins and
verifies multipart output automatically.

```text
$RUN_DIR/$RUN_ID.colab_output.tar.gz
```

Then replay and diagnose with the server/analyzer versions that validate v21's
realized parameter formula and STOP/EMIT metadata:

```bash
BUILD=1 FORCE=0 RESET_PATCH=0 JOBS=8 RUN_ID="$RUN_ID" \
  bash "$EXP/linux/launch_server.sh" replay
tail -F "$RUN_DIR/replay.nohup.log"

python3 "$EXP/python/diagnose_completed_run.py" --run-id "$RUN_ID"
```

A contract PASS proves matched inputs, TRAIN-only label-derived vocabulary and
class weights, deterministic model semantics, and exact reachable-intersection
replay accounting. It does not by itself claim an IPC win; IPC, target and
trigger F1, count accuracy, coverage, timeliness, and traffic remain separate
reported outcomes.
