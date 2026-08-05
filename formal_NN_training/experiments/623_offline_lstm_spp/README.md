# 623 SPP — independent global LSTM + direct learned actions v20

This is the active matched-input SPP experiment for
`623.xalancbmk_s-700B`.  It replaces the v19 routed/page grammar from first
principles.  The neural policy sees exactly the source-visible chronological
callback stream already captured for v18:

- `DEMAND(invoke_prefetcher.addr)`
- `CACHE_FILL(cache_fill.evicted_addr)`

Each address is encoded losslessly as 58 cache-line bits and callback kind is
one bit.  PC is replay transport only.  Source-SPP targets, request counts,
fill levels, thresholds, tables, confidence, candidate actions, and private
state are not runtime inputs.  The recorded fill callbacks came from the
source-SPP run, so the claim is a matched-input **open-loop** comparison, not
closed-loop live NN execution.

## Why v19 was the wrong abstraction

v19 made a direct address pass through sampled `STOP/EMIT` decisions and up
to nine autoregressive LEB128 bytes.  One token error changed the next origin
and every later training state.  Its separate demand/fill/page LSTMs and
page-keyed dictionary also encoded a normal-prefetcher-shaped hypothesis
before the NN had learned it.  The model therefore spent capacity solving an
artificial stochastic serialization problem whose likelihood did not match
the hard addresses consumed by replay.

v20 removes the complete action grammar.  It does not patch its thresholds or
retain its page route.

## v20 architecture

1. One bounded global chronological LSTM learns from the complete 59-bit
   callback sequence.  The reported hidden sizes are 16 and 32.
2. A two-class gate is trained with natural-prior CE.  Its bias is initialized
   from TRAIN log priors and it is decoded by categorical MAP; there is no
   probability threshold.
3. Positive callbacks regress `log(count)` and decode by rounded `exp`.  A
   fail-closed output-resource watchdog aborts the whole run on a pathological
   count; it never truncates or changes the learned action count and is not a
   neural degree cap.
4. Every requested rank receives the same callback state and a generic
   sinusoidal rank code.  All teacher ranks bear loss.  There is no previous
   teacher action, previous sampled action, target, or fill recurrent feedback.
5. Signed target deltas are always relative to the callback line.  The exact
   vocabulary is learned from TRAIN labels only: at most the 255 most frequent
   deltas, with signed-value tie breaking.  A 256th `OTHER` class decodes via a
   signed-log scalar trained on every action. It gives unseen deltas a broad
   bounded approximation but does not guarantee either 58-bit endpoint; only
   the TRAIN vocabulary is integer-exact. The 255 value is a byte-sized representation
   budget, not a probability threshold, page class count, or source-SPP rule.
6. Fill is predicted after and conditioned on rank, decoded delta class, and
   the actual decoded signed-log delta.  Inverse-frequency CE prevents the
   rare L2 label from disappearing during representation learning.  Adding
   the TRAIN log prior recovers a natural posterior, then a stateless
   event/rank-keyed categorical draw selects L2 or LLC.  The same key and
   uniform are used for h16 and h32; there is no manual fill threshold.
7. Count and delta are deterministic.  Duplicate or self-target actions are
   preserved in the replay list so ChampSim queue merges, issue, delay, fill,
   usefulness, and eviction effects—not an offline cleanup rule—determine
   their system effect.
8. Checkpoint selection uses guard target F1, trigger F1, exact-count rate,
   matched-target fill accuracy, and L2 joint F1.  The last term prevents the
   98% LLC majority from rewarding rare-fill collapse; it is a guard metric,
   not a probability threshold.  Evaluation is decoded once after the guard
   checkpoint is frozen.

There is no same-page inference rule, page dictionary, SPP signature/pattern
table, captured candidate list, `STOP/EMIT`, LEB128, action GRU, normal SPP
threshold, neural degree cap, or probability cutoff.

Input revision:
`spp_source_input_variable_delta_fill_feedback_free_running_v11`  
Model revision: `global_chronological_lstm_independent_actions_v20`  
Decoder revision: `gate_logcount_rank_vocab_other_keyed_fill_v20`  
Operation: `train-v20`  
Default run: `623_offline_lstm_spp_independent_vocab_v20_seed7`

| Tag | H | Maximum parameters (255 exact deltas) |
|---|---:|---:|
| `independent_vocab_spp_lstm_h16` | 16 | 8,968 |
| `independent_vocab_spp_lstm_h32` | 32 | 22,272 |

The realized parameter count depends on TRAIN vocabulary size and is recorded
in each `run_metadata.json`.  The exact formula is
`9H² + 79H + 16 + (V+1)(H+1+E) + 2E`, where `E=max(4,H/4)` and
`0<V<=255`.  `python/model_contract.py --json` is the stable machine-readable
source for points, tags, revisions, and the formula.

The run name is also a pinned training contract, not just a label:

| Field | Pinned value |
|---|---:|
| seed | 7 |
| keyed fill decoder seed | 7 |
| epochs | 10 |
| chronological chunk length | 1,024 |
| accumulated chunks / optimizer step | 16 |
| Adam learning rate | 0.002 |

Trainer defaults are loaded from `model_contract.py`, the trainer rejects CLI
overrides that differ from these values, and Colab builds its CLI from the same
contract.  Metadata records both the structured training config and SHA-256 of
the trainer, stable model contract, shared behavior-metric source, and keyed
sampler source.  Colab,
Sacramento installation, analyzer, and diagnosis compare those hashes against
the current repository bytes, preventing stale code from passing by copying
revision strings.

The pinned accelerator is an NVIDIA A100.  The trainer sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing Torch, disables cuDNN
benchmarking, enables cuDNN deterministic mode, and calls strict
`torch.use_deterministic_algorithms(True)`.  Float32 matmul precision is pinned
to `highest` inside the trainer subprocess.  Missing deterministic support or
a non-A100 device aborts the run; nondeterministic warnings are not accepted.

## What is reused from 602, and what is intentionally different

Like 602 SPP, v20 uses one global causal recurrent history, the normal policy's
public inputs only, teacher actions only as output supervision, chronological
TBPTT, a fresh train→guard→eval inference history, and the same fill-preserving
list replayer and system metrics.

Unlike 602's four-component continuous delta mixture and modal fill, v20 uses
an exact TRAIN-derived categorical vocabulary with a rounded, bounded
approximate `OTHER` escape and prior-corrected keyed fill sampling. Unlike 602's autoregressive
own-action feedback, ranks are independent.  Those changes follow the 623
failure history: continuous likelihood versus hard decoding, sampled prefix
exposure, and rare-fill collapse were objective/decoder problems, not evidence
that a larger recurrent state was needed.

## Reuse v18 input byte-for-byte

Do not recollect.  On Sacramento:

```bash
cd ~/cache

export EXP=formal_NN_training/experiments/623_offline_lstm_spp
export SOURCE_RUN=623_offline_lstm_spp_hard_distinct_v18_seed7
export RUN_ID=623_offline_lstm_spp_independent_vocab_v20_seed7
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

The copied `collection_manifest.json` and `spp_source_contract.json` are
historical input/source provenance.  Any legacy student-decoder description
inside that immutable package does not define v20; the current decoder is
pinned by `data/stream_contract.json`, `python/model_contract.py`, and each
model's `run_metadata.json`.

Upload `$RUN_DIR/$RUN_ID.colab_input.tar.gz` to
`colab/623_offline_lstm_spp_A100.ipynb`.  Put the downloaded output at
`$RUN_DIR/$RUN_ID.colab_output.tar.gz`, then run:

```bash
BUILD=0 RUN_ID="$RUN_ID" bash "$EXP/linux/launch_server.sh" replay
tail -F "$RUN_DIR/replay.nohup.log"

python3 "$EXP/python/check_matched_comparison.py" --run-id "$RUN_ID"
python3 "$EXP/python/diagnose_completed_run.py" --run-id "$RUN_ID"
```

Do not run `collect`, do not pass a parent checkpoint, and do not launch
`analyze` concurrently with replay.
