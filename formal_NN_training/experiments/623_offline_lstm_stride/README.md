# 623 Stride — compact independent LSTM v15

This track compares offline normal Stride with a standalone LSTM on
`623.xalancbmk_s-700B` through the same keyed replay transport.

## Fair input contract

Both methods receive the source-visible current `pc` and aligned `addr` only.
The NN encodes the 64-bit PC and the 58-bit cache-line number losslessly (122
features); the six always-zero byte-offset bits are omitted input features.
Training and inference call the same encoder, and the server/analyzer fail
closed unless field lists and encoder hashes agree.  Captured Stride requests
are supervised labels and the offline-normal comparator, never neural inputs.
The NN receives no Stride tracker state, candidates, degree, request-rate
budget, probability cutoff, or future rows.

## v15 hypothesis and completed result

The successful 602 Stride model used a frequency-balanced two-class hurdle,
deterministic positive count, and scalar signed-log delta.  v15 deliberately
tested a different hypothesis: an unweighted Bernoulli hurdle, conditional
Poisson excess, and sampled three-component delta mixture would better model
623's sparse teacher.  The completed seed-7 run rejects that hypothesis.

An exact-PC dynamic map routes callbacks through one compact LSTM.  Count and
mean-sorted mixture-component draws are inverse-CDF samples keyed by trace,
policy, role, evaluation decision index, head, action rank, and decoder seed.
There is no mutable cross-event RNG state.  Every capacity consequently receives
the same event-local random quantiles (common random numbers), while recurrent
feedback uses the complete learned mixture expectation in training and
inference.  `--decoder-seed` is independent of the training seed.

The exact capacity sweep is h8/h16/h32/h64/h128 with
1,923/5,243/16,107/54,731/199,563 parameters.

The root comparison is `PASS`, so transport and counter accounting reconcile.
Nevertheless, offline normal Stride remains best: IPC 0.353400 versus 0.353230
for the best neural point (h64).  h16/h32/h128 emit only 978/1,761/9,669
requests.  Even h64 obtains 164,128 requests, close to normal's 166,147, but
selected accuracy falls from 0.016943 to 0.005589 and coverage from 0.005134 to
0.001750.  This is a valid negative result, not an analyzer failure.

The count factorization is also miscalibrated.  Normal emits 166,147 actions
from 84,200 reached positive triggers (1.973 actions/trigger), while h64 emits
164,128 from 111,571 (1.471 actions/trigger).  It matches aggregate traffic by
firing on more trigger rows, not by recovering normal's degree behavior.  The
mixture likelihood learns both means and scales, but replay emits only a
rounded selected component mean; the learned scale is discarded.  Increasing
hidden size does not repair either train/decode mismatch.

Input revision: `stride_source_input_variable_delta_free_running_v9`  
Model revision: `compact_pc_keyed_crn_event_sampled_mixture_v15`  
Default run: `623_offline_lstm_stride_keyed_crn_v15_seed7`

Run `linux/launch_server.sh collect`, train with the A100 notebook, and copy the
output archive into the canonical run directory.  `launch_server.sh replay`
then safely installs that archive automatically before replay and analysis; no
interactive Python extraction command is required.
Sacramento collect/replay/analyze uses Python 3.6 standard-library code only;
PyTorch, NumPy, and SciPy are Colab training dependencies, not server imports.
The committed TeX results are explicitly historical v9 evidence.  They must
not be relabeled as v15; the completed v15 artifacts are a separate negative
checkpoint under the run ID above.

## Result status and canonical artifacts

Do not use recursive `grep '"status": "PASS"'` on
`matched_comparison.json`: the file embeds child manifests with their own
`status` fields.  Read the root status and root failure list with:

```bash
python3 python/check_matched_comparison.py --run-id "623_offline_lstm_stride_keyed_crn_v15_seed7"
```

The checker exits 0 only for a root `PASS` with an empty failure list, 1 for a
structured root `FAIL`, 2 when analysis is not ready, and 3 for malformed or
inconsistent JSON.

This LSTM-only track produces `matched_comparison.json`,
`matched_comparison.csv`, `insight_summary.csv`, and `replay.nohup.log`.
It does not produce `architecture_pair_summary.csv`; cross-family comparisons
belong outside this single-family run.

Generate a fail-closed diagnosis from the existing PASS artifacts without
training or replay:

```bash
python3 python/diagnose_completed_run.py \
  --run-id "623_offline_lstm_stride_keyed_crn_v15_seed7"
```

This writes `model_diagnosis.json` and `model_diagnosis.csv`.  It verifies the
root PASS, binds current metadata/list/input hashes back to analyzer evidence,
and checks the cross-capacity encoder hash.  It also audits the **current
checkout** of ChampSim Stride: `pc` and `address` must be used by
`invoke_prefetcher`, while generic `cache_hit` and `type` arguments must be
signature-only.  The completed v15 artifacts do not record the historical
Stride source blob SHA, so this current-checkout audit is not claimed as proof
of which exact source blob produced the completed run; the JSON records that
provenance boundary explicitly.

## Next controlled revision

Do not overwrite v15.  A Stride v16 should keep the 623 lossless 122-bit
encoder, exact-PC state router, train/guard/eval chronology, keyed transport,
and all audits.  Change only the failed heads/objectives/decoder to the
602-proven method: data-derived balanced categorical hurdle, deterministic
rounded positive log-count, and deterministic scalar signed-log delta with
free-running feedback.  Guard must either select/check a checkpoint or be
named warm-up; it must not be described as validation while it only warms
recurrent state.  Use a new model revision and run ID, and change the Python
entrypoint, notebook assertions, and server metadata assertions atomically.

If replay logs and event files already exist, regenerate only the derived
analysis after an analyzer update:

```bash
BUILD=0 RUN_ID="623_offline_lstm_stride_keyed_crn_v15_seed7" \
  bash formal_NN_training/experiments/623_offline_lstm_stride/linux/launch_server.sh analyze
```
