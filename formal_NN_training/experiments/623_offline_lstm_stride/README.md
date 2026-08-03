# 623 Stride — compact independent LSTM v15

This track compares offline normal Stride with a standalone LSTM on
`623.xalancbmk_s-700B` through the same keyed replay transport.

## Fair input contract

Both methods receive the source-visible current `pc` and aligned `addr` only.
The NN encodes the 64-bit PC and the 58-bit cache-line number losslessly (122
features); the six always-zero byte-offset bits are not model parameters.
Training and inference call the same encoder, and the server/analyzer fail
closed unless field lists and encoder hashes agree.  Captured Stride requests
are supervised labels and the offline-normal comparator, never neural inputs.
The NN receives no Stride tracker state, candidates, degree, request-rate
budget, probability cutoff, or future rows.

## Why this differs from 602 Stride

The successful 602 Stride model used a two-class deterministic sparse gate and
a scalar delta because its teacher action distribution was dominated by zero
actions and one stable stride.  The 623 trace is a different, much sparser and
less successful regime in the existing results.  v15 therefore keeps the
learned Bernoulli hurdle, conditional Poisson excess, and three-component
signed-delta mixture instead of copying the 602 decoder.

An exact-PC dynamic map routes callbacks through one compact LSTM.  Count and
mean-sorted mixture-component draws are inverse-CDF samples keyed by trace,
policy, role, evaluation decision index, head, action rank, and decoder seed.
There is no mutable cross-event RNG state.  Every capacity consequently receives
the same event-local random quantiles (common random numbers), while recurrent
feedback uses the complete learned mixture expectation in training and
inference.  `--decoder-seed` is independent of the training seed.

The exact capacity sweep is h8/h16/h32/h64/h128 with
1,923/5,243/16,107/54,731/199,563 parameters.

Input revision: `stride_source_input_variable_delta_free_running_v9`  
Model revision: `compact_pc_keyed_crn_event_sampled_mixture_v15`  
Default run: `623_offline_lstm_stride_keyed_crn_v15_seed7`

Run `linux/launch_server.sh collect`, train with the A100 notebook, and copy the
output archive into the canonical run directory.  `launch_server.sh replay`
then safely installs that archive automatically before replay and analysis; no
interactive Python extraction command is required.
Sacramento collect/replay/analyze uses Python 3.6 standard-library code only;
PyTorch, NumPy, and SciPy are Colab training dependencies, not server imports.
The committed TeX results are explicitly historical v9 evidence; v15 results
must come from the new run ID and are currently pending.

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

If replay logs and event files already exist, regenerate only the derived
analysis after an analyzer update:

```bash
BUILD=0 RUN_ID="623_offline_lstm_stride_keyed_crn_v15_seed7" \
  bash formal_NN_training/experiments/623_offline_lstm_stride/linux/launch_server.sh analyze
```
