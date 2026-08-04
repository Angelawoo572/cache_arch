# 623 SPP — factorized delta and keyed rare-fill LSTM v17

This directory is the active matched-input SPP experiment for
`623.xalancbmk_s-700B`. The NN receives the same source-visible chronological
callbacks as SPP: `DEMAND(addr)` and `CACHE_FILL(evicted_addr)`. PC is replay
transport only. Captured SPP actions and fill choices are supervised labels and
the offline-normal replay; they never enter NN inference.

Recorded fill callbacks came from the source-SPP run, so the valid claim is a
matched-input open-loop comparison, not closed-loop live-NN execution.

## What 602 actually contributes

602 SPP is also a behavior-cloning student: normal actions are training labels,
while normal candidates/private state are not neural inputs. Its useful design
lesson is a compact causal recurrent student with separate action decisions,
not label independence. 623 keeps that experimental contract, its already
validated 59-feature input, split, chronology, count decoder, keyed CRN, and
fill-preserving replay.

## v16A diagnosis

The v16A result is a negative architecture/decoder result, even though its root
accounting passed. Normal SPP's captured list has only about two percent L2
placements. v16A selected a single joint `(delta component, fill)` class by
MAP. Four of five capacities then had hundreds of thousands of requested
actions but zero L2 lifecycle counters; h16 produced only a small effective L2
stream. The result shows no effective rare-L2 placement. Exact list L2/LLC
counts remain separately audited in each model's `run_metadata.json`; runtime
fill counters are not silently equated with list totals.

The request-count head is not the primary failure: action volume was already
close to normal. Adding a second recurrent state or more capacity would make
the experiment harder to interpret without evidence that the 59-feature global
LSTM is the bottleneck.

## Minimal v17 architecture

v17 retrains one global LSTM and changes only the action head:

- the existing unweighted Bernoulli hurdle and conditional Poisson excess keep
  the learned, unbounded request count and event-keyed common random numbers;
- a four-component signed-log direct-delta mixture is trained by mixture NLL
  and emits the modal component mean deterministically;
- a separate unweighted two-class fill head learns the empirical L2/LLC
  probability;
- fill is drawn with the existing stateless keyed categorical inverse CDF using
  event identity and action rank, so rare L2 probability cannot be erased by
  argmax;
- autoregressive feedback uses the factorized delta expectation and fill
  probabilities in both training and inference; teacher or sampled actions are
  never fed back.

There is no joint MAP, guard-selected decoder, threshold, degree cap, request
budget, candidate bank, page-offset class, or dual recurrent state.

Input revision:
`spp_source_input_variable_delta_fill_feedback_free_running_v11`  
Model revision: `compact_crn_factorized_delta_keyed_fill_v17`  
Operation: `train-v17`  
Default run: `623_offline_lstm_spp_factorized_fill_v17_seed7`

| Tag suffix | Hidden size | Parameters |
|---|---:|---:|
| `h8` | 8 | 2,664 |
| `h16` | 16 | 6,208 |
| `h32` | 32 | 15,984 |
| `h64` | 64 | 46,288 |
| `h128` | 128 | 149,904 |

For 59 input features, the exact formula is `7H^2 + 275H + 16`.

## Reuse the validated input; do not recollect

Use the clean completed v15 input byte-for-byte. v17 does not need v15
checkpoints:

```bash
cd ~/cache

export EXP=formal_NN_training/experiments/623_offline_lstm_spp
export SOURCE_RUN=623_offline_lstm_spp_keyed_crn_joint_fill_v15_seed7
export RUN_ID=623_offline_lstm_spp_factorized_fill_v17_seed7
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

python3 "$EXP/python/validate_collected_inputs.py"   --input-dir "$RUN_DIR/colab_input"   --manifest-out "$RUN_DIR/colab_input/collection_manifest.json"   --source-contract "$RUN_DIR/colab_input/spp_source_contract.json"
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
`analyze` concurrently with replay. The analyzer keeps legacy v15/v16A
support, but the defaults and current contract are v17.
