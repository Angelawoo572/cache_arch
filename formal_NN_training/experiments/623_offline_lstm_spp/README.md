# 623 SPP — hard distinct-action feedback LSTM v18

This directory is the active matched-input SPP experiment for
`623.xalancbmk_s-700B`. The NN receives exactly the source-visible chronological
callbacks read by SPP: `DEMAND(addr)` and `CACHE_FILL(evicted_addr)`. PC is
replay transport only. Captured SPP request count, target, and fill are
supervised labels and the offline-normal replay; none is an NN runtime input.

Recorded fill callbacks were produced by the source-SPP run, so this remains a
matched-input open-loop comparison, not closed-loop live-NN execution.

## Why v17 failed

The v17 root `PASS` validated input/replay accounting, not model quality. All
five neural points lost to offline SPP. The best point was h128 at IPC 0.352960
versus 0.353900; its L2 miss rate rose from 0.360813 to 0.372343. Across the
sweep, neural L2 placement was 3.56--5.31% versus the teacher's 2.00%, turning
many wrong targets into harmful L2 fills.

The decoder also exposed a structural mismatch. Training fed the GRU mixture
expectation and fill probability, while replay executed a modal hard delta and
a keyed hard fill. Later ranks therefore did not condition on the action that
was actually emitted. In h16--h128, most second-or-later requests collapsed
onto an already requested address and were merged by the prefetch queue. Hidden
size was not the bottleneck.

## Minimal v18 repair

v18 retains the same 59-bit input, labels, split, chronology, one global LSTM,
four-component signed-log delta mixture, separate fill head, and exact model
sizes. Only count/action decoding and recurrent feedback change:

- count is deterministic: raw hurdle-logit sign, then one plus the rounded
  learned conditional excess mean; there is no Bernoulli or Poisson draw;
- the shared training/inference selector orders delta components by learned
  peak density (`log mixture mass - log scale`), hard-quantizes the chosen
  signed delta, and enforces only non-self and within-callback distinctness;
- `+2^57` and `-2^57` are canonicalized to the same signed delta before
  legality checks because they materialize the same 58-bit target;
- the actual hard emitted delta is the next-rank feedback; training uses a
  straight-through value so gradients still reach the selected mean;
- fill is drawn directly from the learned two-class posterior with the
  stateless keyed sampler; its uniform remains float64, and the actual hard
  one-hot fill is next-rank feedback through a straight-through value;
- no address-confidence multiplier, threshold, degree cap, request budget,
  normal candidate, page rule, or second recurrent state is added.

Teacher count schedules which action ranks receive training loss. It does not
mean that training count rollout equals inference. Teacher delta/fill values
remain loss-only; the action feedback is always the model's own hard action.

Input revision:
`spp_source_input_variable_delta_fill_feedback_free_running_v11`  
Model revision: `compact_crn_hard_distinct_delta_keyed_fill_v18`  
Decoder revision: `hard_distinct_delta_keyed_fill_v18`  
Operation: `train-v18`  
Default run: `623_offline_lstm_spp_hard_distinct_v18_seed7`

| Tag suffix | Hidden size | Parameters |
|---|---:|---:|
| `h8` | 8 | 2,664 |
| `h16` | 16 | 6,208 |
| `h32` | 32 | 15,984 |
| `h64` | 64 | 46,288 |
| `h128` | 128 | 149,904 |

For 59 input features, the exact formula remains `7H^2 + 275H + 16`.

## Reuse the validated input; do not recollect

The v17 input package is reused byte-for-byte. Its `collection_manifest.json`
is historical input-package provenance; old decoder fields in that manifest
are not the v18 decoder contract. The v18 contract is this source tree plus
each model's `run_metadata.json`.

```bash
cd ~/cache

export EXP=formal_NN_training/experiments/623_offline_lstm_spp
export SOURCE_RUN=623_offline_lstm_spp_factorized_fill_v17_seed7
export RUN_ID=623_offline_lstm_spp_hard_distinct_v18_seed7
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
  --manifest-out "$RUN_DIR/colab_input/collection_manifest.json" \
  --source-contract "$RUN_DIR/colab_input/spp_source_contract.json"
```

Upload the renamed input archive to
`colab/623_offline_lstm_spp_A100.ipynb`. Put the downloaded output archive at
`$RUN_DIR/$RUN_ID.colab_output.tar.gz`, then run:

```bash
BUILD=0 RUN_ID="$RUN_ID" bash "$EXP/linux/launch_server.sh" replay
tail -F "$RUN_DIR/replay.nohup.log"

python3 "$EXP/python/check_matched_comparison.py" --run-id "$RUN_ID"
python3 "$EXP/python/diagnose_completed_run.py" --run-id "$RUN_ID"
```

Do not run `collect`, do not pass a parent checkpoint, and do not launch
`analyze` concurrently with replay. The analyzer remains compatible with v15,
v16A, and v17, while defaults and current validation target v18.
