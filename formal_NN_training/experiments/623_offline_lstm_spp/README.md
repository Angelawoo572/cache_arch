# 623 SPP — routed/page-local learned action grammar v19

This directory is the active matched-input SPP experiment for
`623.xalancbmk_s-700B`. The neural policy receives exactly the same
source-visible chronological callbacks as v18:

- `DEMAND(invoke_prefetcher.addr)`
- `CACHE_FILL(cache_fill.evicted_addr)`

PC remains replay transport only. Source-SPP request count, target, and fill
are supervised labels and the offline-normal comparator; they are never NN
runtime inputs. Recorded fill callbacks came from the source-SPP run, so the
claim remains a matched-input open-loop comparison, not live closed-loop NN
execution.

## Why v18 collapsed

v18 factorized a structured variable-length action into four marginal tasks.
That made the easiest likelihood solution the observed failure mode:

- a weak gate with a positive marginal above one half became almost-always
  positive under logit-sign decoding;
- the conditional excess mean was below one half, so rounding produced one
  action on almost every positive callback;
- continuous mixture NLL did not train the hard peak-density address selector;
- fill was predicted independently before the actual target, allowing wrong
  targets to receive harmful L2 placement.

Increasing hidden size cannot repair those objective/decoder mismatches.

## v19 architecture

v19 replaces the complete v18 model rather than calibrating it:

1. Lossless line58 + callback-kind bits are projected once.
2. DEMAND and FILL events update separate chronological LSTMs.
3. A page-keyed causal LSTM sees the same raw line-derived page identity and a
   causal reuse age. A learned vector validity gate decides how much of that
   local state should contribute.
4. Learned fusion combines demand, fill, page-local, and current-event state.
5. Each action rank samples a learned categorical `STOP` or `EMIT` token with
   a stateless event/rank key. The first `STOP` determines request count. There
   is no `0.5` rule, threshold, hurdle, Poisson, rounded mean, or degree cap.
6. `EMIT` generates an exact signed increment with a compact autoregressive
   byte GRU using ZigZag + canonical LEB128. Each actual hard byte conditions
   the next byte. Common small deltas need one byte and the complete signed
   58-bit cache-line domain needs at most nine. Rank zero is relative to the
   callback line; later ranks are relative to the actual previous target.
7. `delta=0` and a first-rank self target are legal, matching the captured SPP
   semantics. Only a target already emitted by this callback is masked from
   the learned categorical distribution and renormalized. Continuation bytes
   are masked when their complete subtree contains no legal target, so byte
   nine cannot dead-end. There is no backtracking, nearest `+/-1`, or other
   invented fallback.
8. Fill is predicted only after and conditioned on the actual hard target.
   Actual hard increment/target/fill values feed the next rank.

Training follows the sampled trajectory. STOP/EMIT CE exists only at states
and ranks actually reached by keyed model sampling. A strictly separate,
weight-shared teacher-prefix branch computes the full canonical target NLL;
the teacher prefix advances only that isolated likelihood branch's byte-GRU
state. It cannot change the main sampled action state, origin, duplicate
history, or runtime output. Fill CE is evaluated as the
conditional factor `p(fill | state, teacher target)`. Runtime fill is sampled
only from `p(fill | state, actual sampled target)`, and only actual hard
target/fill values feed the next rank. The summed NLL is divided by the total
number of categorical atoms in each gradient-accumulation group; there are no
manual per-head loss weights.

A 52-rank watchdog is derived from the 52-bit keyed-uniform sampler precision.
It is fail-closed: if a learned trajectory never samples STOP, the entire run
raises and emits no replay. It never truncates, forces STOP, or acts as a
neural degree cap. STOP probability must also be representable on that open
inverse-CDF grid.

Input revision:
`spp_source_input_variable_delta_fill_feedback_free_running_v11`  
Model revision: `routed_page_lstm_rank_grammar_leb128_v19`  
Decoder revision: `keyed_stop_emit_zigzag_leb128_target_fill_v19`  
Operation: `train-v19`  
Default run: `623_offline_lstm_spp_routed_grammar_v19_seed7`

| Tag | Fusion/action H | Routed LSTM R | Codec E | Trainable parameters |
|---|---:|---:|---:|---:|
| `routed_grammar_spp_lstm_h8` | 8 | 4 | 2 | 2,790 |
| `routed_grammar_spp_lstm_h16` | 16 | 8 | 4 | 7,032 |

The exact parameter formula is
`25R² + 90R + 4RH + 3H² + 7HE + 17H + 6E² + 589E + 260`, with
`R=H//2` and `E=H//4`. `python/model_points_v19.py --json` is the single
machine-readable source for sizes, pair IDs, derived widths, tags, revisions,
and counts. Run metadata also reports dynamic page-state bytes separately;
they are not hidden inside the weight count.

## Reuse v18 input byte-for-byte

Do not recollect. On Sacramento:

```bash
cd ~/cache

export EXP=formal_NN_training/experiments/623_offline_lstm_spp
export SOURCE_RUN=623_offline_lstm_spp_hard_distinct_v18_seed7
export RUN_ID=623_offline_lstm_spp_routed_grammar_v19_seed7
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

Upload `$RUN_DIR/$RUN_ID.colab_input.tar.gz` to
`colab/623_offline_lstm_spp_A100.ipynb`. Put the downloaded output at
`$RUN_DIR/$RUN_ID.colab_output.tar.gz`, then run:

```bash
BUILD=0 RUN_ID="$RUN_ID" bash "$EXP/linux/launch_server.sh" replay
tail -F "$RUN_DIR/replay.nohup.log"

python3 "$EXP/python/check_matched_comparison.py" --run-id "$RUN_ID"
python3 "$EXP/python/diagnose_completed_run.py" --run-id "$RUN_ID"
```

Do not run `collect`, do not pass a parent checkpoint, and do not launch
`analyze` concurrently with replay.
