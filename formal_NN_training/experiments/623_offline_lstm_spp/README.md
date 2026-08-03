# 623 SPP — compact independent LSTM v15

This track compares offline normal SPP with a standalone LSTM on
`623.xalancbmk_s-700B` through the same fill-preserving replay transport.

## Fair input contract and claim boundary

The audited source-visible input is the chronological callback stream:
`DEMAND(addr)` and `CACHE_FILL(evicted_addr)`.  PC is replay transport only;
cache hit/type and SPP's private ST/PT/GHR/FILTER state are excluded.  The NN
encodes the 58-bit cache-line number plus one callback-kind bit losslessly (59
features), removing six byte-offset bits that are identically zero.  Captured
SPP targets/fill choices are labels and the offline-normal comparator only.

This is a same-source-input offline comparison, not a closed-loop live-NN
claim: the recorded fill callbacks are consequences of the source SPP run.

## Why this differs from 602 SPP

602 SPP used separate direct-delta and fill heads and deterministic decoding.
For 623, fill level is statistically tied to the requested target and queue
effect, so v15 models one joint distribution over four delta components and two
fill classes.  Its joint likelihood and single sampled `(component, fill)` pair
preserve that dependence instead of sampling two independent heads.

One chronological LSTM retains the learned Bernoulli hurdle and conditional
Poisson excess.  Count and mean-sorted joint-pair choices are stateless
inverse-CDF samples keyed only by the allowed chronological decision identity,
head, action rank, trace/policy/role, and `--decoder-seed`.  They do not use PC,
raw teacher event gaps, private SPP state, or a probability threshold.  Full
joint delta/fill marginals drive recurrent feedback in both training and
inference.

The exact capacity sweep is h8/h16/h32/h64/h128 with
2,682/6,242/16,050/46,418/150,162 parameters.

Input revision: `spp_source_input_variable_delta_fill_feedback_free_running_v11`  
Model revision: `compact_crn_joint_delta_fill_mixture_v15`  
Default run: `623_offline_lstm_spp_keyed_crn_joint_fill_v15_seed7`

Run `linux/launch_server.sh collect`, train with the A100 notebook, and copy the
output archive into the canonical run directory.  `launch_server.sh replay`
then safely installs that archive automatically before replay and analysis; no
interactive Python extraction command is required.
Sacramento collect/replay/analyze uses Python 3.6 standard-library code only;
PyTorch, NumPy, and SciPy are Colab training dependencies, not server imports.
The committed TeX results are explicitly historical v11 evidence; v15 results
must come from the new run ID and are currently pending.

## Result status and canonical artifacts

Do not use recursive `grep '"status": "PASS"'` on
`matched_comparison.json`: the file embeds child manifests with their own
`status` fields.  Read the root status and root failure list with:

```bash
python3 python/check_matched_comparison.py --run-id "623_offline_lstm_spp_keyed_crn_joint_fill_v15_seed7"
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
BUILD=0 RUN_ID="623_offline_lstm_spp_keyed_crn_joint_fill_v15_seed7" \
  bash formal_NN_training/experiments/623_offline_lstm_spp/linux/launch_server.sh analyze
```
