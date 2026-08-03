# 623 Stride — compact balanced deterministic LSTM v16

This track compares offline normal Stride with a standalone LSTM on
`623.xalancbmk_s-700B` through the same keyed replay transport.  v16 is a new,
controlled run; it does not overwrite the completed v15 negative checkpoint.

## Fair input contract

Both methods receive the source-visible current `pc` and aligned `addr` only.
The NN encodes the 64-bit PC and the 58-bit cache-line number losslessly (122
features); the six always-zero byte-offset bits are omitted.  Training and
inference call the same encoder.  Captured Stride requests are supervised
labels and the offline-normal comparator, never neural inputs.

The neural policy receives no Stride tracker state, candidate bank, request
budget, degree cap, selected probability threshold, fixed page-offset table,
same-page rule, handcrafted semantic feature, cache hit, queue state, cycle, or
future row.  A dynamic exact-PC map routes recurrent state without copying the
teacher's fixed tracker capacity.

## Why v16 changes the v15 heads

The completed v15 root comparison is `PASS`, so its input, transport, and
counter accounting reconcile.  It is nevertheless a valid negative model
result:

- offline normal Stride reaches IPC 0.353400; the best v15 neural point (h64)
  reaches 0.353230;
- h16/h32/h128 collapse to only 978/1,761/9,669 requests;
- h64 nearly matches normal request volume (164,128 versus 166,147) but reaches
  many more trigger rows, averaging 1.471 actions per reached trigger rather
  than normal's 1.973;
- h64 selected accuracy is 0.005589 versus 0.016943 and coverage is 0.001750
  versus 0.005134;
- v15 trains mixture means and scales, but replay emits only a rounded selected
  component mean, so learned scale cannot improve the emitted address.

Those diagnostics support a 602-style **head/objective/decoder** correction,
not a wholesale copy of the 602 network.  v16 retains the 623 lossless encoder,
exact-PC state routing, complete train→guard→evaluation chronology, and keyed
replay transport.  It replaces only the failed stochastic heads:

- a two-class zero/positive gate trained with inverse-frequency weights derived
  from the v16 training split, giving each observed class equal aggregate loss
  mass;
- deterministic two-class argmax, with no tuned probability threshold;
- a deterministic positive log-count regressor decoded by rounded exponent,
  with positive integer support and no degree cap;
- a scalar signed-log cache-line-delta regressor decoded by deterministic
  rounding;
- the emitted scalar coordinate is the autoregressive feedback in both
  training and inference; teacher deltas contribute loss only.

The guard split is causal recurrent-history warm-up and audit only.  It is not
called validation and does not select a checkpoint.

This is a diagnosis-backed hypothesis, not a promised IPC improvement.  The
normal Stride teacher itself is only 0.000190 IPC above no-prefetch on this
trace, so v16 must still be judged by replayed miss rate, target quality, and
IPC rather than offline loss alone.

## Architecture points and identifiers

The capacity tags remain unchanged because the run ID and model revision scope
the artifacts:

| Tag suffix | Hidden size | Parameters |
|---|---:|---:|
| `h8` | 8 | 1,860 |
| `h16` | 16 | 5,124 |
| `h32` | 32 | 15,876 |
| `h64` | 64 | 54,276 |
| `h128` | 128 | 198,660 |

For runtime feature count `F=122`, the exact compact-model formula is
`11H^2 + (F + 22)H + 4 = 11H^2 + 144H + 4`.  Training and analysis assert the
measured count at every point.

Input revision: `stride_source_input_variable_delta_free_running_v9`  
Model revision: `compact_pc_keyed_balanced_deterministic_scalar_v16`  
Default run: `623_offline_lstm_stride_compact_hurdle_v16_seed7`

The A100 notebook trains h8/h16/h32/h64/h128 with 12 epochs, chunk length 256,
PC batch size 128, learning rate 0.002, and seed 7.  Tags remain
`independent_delta_stride_lstm_h<size>`.

## Reuse the matched v15 inputs; do not recollect

v16 has the same input revision, trace split, logger schema, effective external
fields, and teacher labels as v15.  Reuse the already validated v15
`colab_input` byte-for-byte.  Before launching any v16 stage, create only the
new input copy and renamed upload archive below.  The existence checks prevent
an accidental merge into an old or partial v16 input directory; nothing under
the v15 run is modified.

```bash
cd ~/cache

export EXP=formal_NN_training/experiments/623_offline_lstm_stride
export OLD_RUN=623_offline_lstm_stride_keyed_crn_v15_seed7
export RUN_ID=623_offline_lstm_stride_compact_hurdle_v16_seed7
export OLD_RUN_DIR="$EXP/runs/$OLD_RUN"
export NEW_RUN_DIR="$EXP/runs/$RUN_ID"

test -d "$OLD_RUN_DIR/colab_input"
test -s "$OLD_RUN_DIR/$OLD_RUN.colab_input.tar.gz"
test ! -e "$NEW_RUN_DIR/colab_input"
test ! -e "$NEW_RUN_DIR/$RUN_ID.colab_input.tar.gz"

mkdir -p "$NEW_RUN_DIR"
cp -a "$OLD_RUN_DIR/colab_input" "$NEW_RUN_DIR/colab_input"
cp -p \
  "$OLD_RUN_DIR/$OLD_RUN.colab_input.tar.gz" \
  "$NEW_RUN_DIR/$RUN_ID.colab_input.tar.gz"

cmp "$OLD_RUN_DIR/$OLD_RUN.colab_input.tar.gz" \
  "$NEW_RUN_DIR/$RUN_ID.colab_input.tar.gz"
diff -qr "$OLD_RUN_DIR/colab_input" "$NEW_RUN_DIR/colab_input"

python3 "$EXP/python/validate_collected_inputs.py" \
  --input-dir "$NEW_RUN_DIR/colab_input" \
  --manifest-out "$NEW_RUN_DIR/colab_input/collection_manifest.json"
```

`cmp` and `diff` print nothing on success; the validator prints `[PASS]`.  Upload
`$NEW_RUN_DIR/$RUN_ID.colab_input.tar.gz` to
`colab/623_offline_lstm_stride_A100.ipynb`.  After training, place the downloaded
`$RUN_ID.colab_output.tar.gz` at `$NEW_RUN_DIR/$RUN_ID.colab_output.tar.gz`.
Then launch replay; do not run `collect`.  The replay stage installs only the
v16 model outputs, creates fresh v16 simulation logs/events, and invokes
analysis after every simulation finishes.

```bash
cd ~/cache

BUILD=0 RUN_ID="$RUN_ID" bash "$EXP/linux/launch_server.sh" replay
tail -f "$NEW_RUN_DIR/replay.nohup.log"
```

`launch_server.sh replay` installs the archive before simulation.
Sacramento-side replay/analyze remains Python 3.6 standard-library compatible.
The v16 launcher defaults to `replay`; `collect` remains available only when
named explicitly.
Do not launch `analyze` concurrently with the background replay.  Use the
standalone `analyze` stage only after replay has exited when regenerating
derived reports from already complete logs/events.

Read only the root result status and failure list with:

```bash
python3 "$EXP/python/check_matched_comparison.py" --run-id "$RUN_ID"
```

The checker exits 0 only for a root `PASS` with an empty failure list, 1 for a
structured root `FAIL`, 2 when analysis is not ready, and 3 for malformed or
inconsistent JSON.  This LSTM-only track produces `matched_comparison.json`,
`matched_comparison.csv`, `insight_summary.csv`, and `replay.nohup.log`; it does
not produce `architecture_pair_summary.csv`.

After a completed root-PASS v16 run, the default diagnosis command is:

```bash
python3 "$EXP/python/diagnose_completed_run.py"
```

To diagnose the preserved v15 checkpoint explicitly:

```bash
python3 "$EXP/python/diagnose_completed_run.py" \
  --run-id 623_offline_lstm_stride_keyed_crn_v15_seed7
```

The analyzer and diagnosis tool accept either supported model revision, but
fail closed if capacities are mixed within one run.  The diagnosis binds model
metadata and replay-list hashes to analyzer evidence and audits the **current
checkout** of ChampSim Stride.  Neither completed v15 nor newly generated v16
metadata records the historical Stride source blob SHA, so the current-source
audit is not proof of the source blob that produced an earlier collection.

The committed TeX results remain historical v9 evidence.  They must not be
relabeled as v15 or v16.  The v15 run remains under
`623_offline_lstm_stride_keyed_crn_v15_seed7`; v16 writes only under its new run
ID.
