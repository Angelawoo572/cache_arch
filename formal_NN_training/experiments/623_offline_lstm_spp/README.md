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

## v15 hypothesis and completed result

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

The root comparison is `PASS`, so fill-preserving transport and counter
accounting reconcile.  It is still a negative performance result: offline SPP
has IPC 0.353900, while the best neural point h64 has 0.353270.  Unlike Stride,
request count has not collapsed: normal emits 804,086 actions from 707,263
reached triggers (1.137 actions/trigger), and the neural points emit
722,778--781,796 actions at 1.066--1.088 actions/trigger.

Target/fill placement is the next bottleneck.  h32 raises selected accuracy
from 0.001795 to 0.002895, coverage from 0.003038 to 0.004430, and timeliness
from 0.424449 to 0.736434, yet worsens L2 miss rate from 0.360813 to 0.376874
and loses 0.001240 IPC.  These aggregate counters are consistent with harmful
cache interference, but do not directly prove victim pollution.  The joint
mixture also learns means and scales while replay emits only a rounded selected
component mean, discarding scale.

Input revision: `spp_source_input_variable_delta_fill_feedback_free_running_v11`  
Model revision: `compact_crn_joint_delta_fill_mixture_v15`  
Default run: `623_offline_lstm_spp_keyed_crn_joint_fill_v15_seed7`

Run `linux/launch_server.sh collect`, train with the A100 notebook, and copy the
output archive into the canonical run directory.  `launch_server.sh replay`
then safely installs that archive automatically before replay and analysis; no
interactive Python extraction command is required.
Sacramento collect/replay/analyze uses Python 3.6 standard-library code only;
PyTorch, NumPy, and SciPy are Colab training dependencies, not server imports.
The committed TeX results are explicitly historical v11 evidence.  They must
not be relabeled as v15; the completed v15 artifacts are a separate negative
checkpoint under the run ID above.

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

Generate a fail-closed diagnosis from the existing PASS artifacts without
training or replay:

```bash
python3 python/diagnose_completed_run.py \
  --run-id "623_offline_lstm_spp_keyed_crn_joint_fill_v15_seed7"
```

This writes `model_diagnosis.json` and `model_diagnosis.csv`, preserving
request, target-quality, fill, and cache-lifecycle evidence as separate fields.

## Next controlled revision

Do not blindly copy the 602 SPP topology and do not overwrite v15.  The 623
normal labels contain only about two percent L2 fills, so a separate uncalibrated
fill argmax can collapse to all LLC.  First use `model_diagnosis.json` to inspect
normal/NN fill counts, held-out joint-label behavior, useful/useless/late/drop
counters, and delta/fill dependence.  The lowest-risk v16 ablation keeps the
current input, global state, count model, and joint delta/fill head, but compares
the current sampled joint pair against deterministic joint MAP decoding.  If
the joint dependence is weak or the global state is the limiting factor, a
later trace-specific model may combine a page-keyed local LSTM with a small
global fill/event LSTM; page and offset are derived from the same allowed
address, so this changes state organization rather than external input.

Any v16 must use a new model revision/run ID and update the Python entrypoint,
A100 notebook assertions, and server metadata assertions atomically.  Guard
must either participate in checkpoint selection or be named warm-up.

If replay logs and event files already exist, regenerate only the derived
analysis after an analyzer update:

```bash
BUILD=0 RUN_ID="623_offline_lstm_spp_keyed_crn_joint_fill_v15_seed7" \
  bash formal_NN_training/experiments/623_offline_lstm_spp/linux/launch_server.sh analyze
```
