# 623 SPP — v15 checkpoint and deterministic-map v16A

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

The root comparison is `PASS`, so aggregate action transport and simulator
counter accounting reconcile.  It does not independently prove per-fill-class
runtime conservation or victim pollution.  It is still a negative performance
result: offline SPP has IPC 0.353900, while the best neural point h64 has
0.353270.  Unlike Stride,
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

Preserved v15 input revision:
`spp_source_input_variable_delta_fill_feedback_free_running_v11`  
Preserved v15 model revision:
`compact_crn_joint_delta_fill_mixture_v15`  
Preserved v15 run: `623_offline_lstm_spp_keyed_crn_joint_fill_v15_seed7`

Set `EXPERIMENT_MODE=v15` before `linux/launch_server.sh collect` to reproduce
that historical workflow, train with the A100 notebook, and copy the output
archive into the canonical run directory.  `launch_server.sh replay`
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

This writes `model_diagnosis.json` and `model_diagnosis.csv`, binds current
metadata/list/input hashes to analyzer evidence, and keeps request,
target-quality, fill, and cache-lifecycle evidence separate.  Full replay-list
fill totals and runtime issued-event fill counts are distinct domains; the
diagnosis does not silently equate them.

## Controlled v16A checkpoint re-decode

Do not blindly copy the 602 SPP topology and do not overwrite v15.  The v15
metadata reports 16,044 L2 versus 788,050 LLC actions in the complete
804,094-entry normal list (about two percent L2).  Eight list entries are not reached
in the intervention run, so runtime-reachable/requested actions are 804,086;
these are distinct accounting domains.  The diagnosis binds list totals to
analyzer replay-list hashes before using them.  A separate uncalibrated fill
argmax can therefore
collapse to all LLC.  First use `model_diagnosis.json` to inspect
normal/NN fill counts, held-out joint-label behavior, useful/useless/late/drop
counters, and delta/fill dependence.

v16A is the lowest-risk ablation.  It strictly loads each v15 checkpoint and
copies `model.pt` and `training_history.csv` byte-for-byte; it does not retrain
weights.  The keyed Bernoulli hurdle and conditional Poisson count draws are
unchanged.  On guard labels only, it compares:

1. `joint_class_map`: the maximum learned joint component/fill probability;
2. `component_peak_map`: the maximum joint probability divided by learned
   component scale, evaluated at each component mean.  This is a
   component-peak comparison, not a claim to find the exact Gaussian-mixture
   mode.

Exact ties use mean/fill/component canonical order.  The predeclared
lexicographic guard objective is: maximize joint address/fill F1, maximize
address F1, maximize trigger F1, minimize absolute action-count-ratio error,
maximize rare-L2 joint F1, minimize absolute L2-fraction error, then use the
declared decoder-mode order.  Trigger/count terms are expected to tie because
both candidates share the same count draws; recording them keeps the selection
contract explicit.  Evaluation labels never select the mode.

Input revision remains
`spp_source_input_variable_delta_fill_feedback_free_running_v11`; the active
model revision is `compact_crn_joint_delta_fill_guard_map_v16a`, the preserved
weights revision is `compact_crn_joint_delta_fill_mixture_v15`, and the new run
is `623_offline_lstm_spp_keyed_crn_joint_map_v16a_seed7`.  Artifact tags are
`guard_joint_map_spp_lstm_h{8,16,32,64,128}`.

Prepare the v16A Colab input from the completed v15 run without recollecting:

```bash
EXPERIMENT_MODE=v16a \
  bash formal_NN_training/experiments/623_offline_lstm_spp/linux/launch_server.sh collect
```

To reproduce the preserved v15 workflow instead, set `EXPERIMENT_MODE=v15`
and its existing run ID explicitly.  v16A will fail closed if any parent input,
encoder, metadata, checkpoint payload, parameter count, source contract,
normal replay, or artifact byte hash differs.

If v16A does not improve target/fill behavior, a later separately named
retraining revision may use callback-kind-routed demand and fill recurrent
states.  Both states must consume only the same callback kind and callback
address fields; it must not add page routing, offset classes, private teacher
state, or be folded into this decoder-only ablation.

After Colab downloads the v16A output archive, upload it to the canonical run
directory under the exact name
`623_offline_lstm_spp_keyed_crn_joint_map_v16a_seed7.colab_output.tar.gz`.
Then run only replay; replay installs the archive and invokes analysis itself:

```bash
cd ~/cache
export EXPERIMENT_MODE=v16a
export RUN_ID=623_offline_lstm_spp_keyed_crn_joint_map_v16a_seed7
export EXP=formal_NN_training/experiments/623_offline_lstm_spp

test -s "$EXP/runs/$RUN_ID/$RUN_ID.colab_output.tar.gz"
BUILD=0 bash "$EXP/linux/launch_server.sh" replay
tail -F "$EXP/runs/$RUN_ID/replay.nohup.log"
```

Do not launch `analyze` concurrently with `replay`.  After replay finishes,
check the root result and generate the revision-aware diagnosis:

```bash
python3 "$EXP/python/check_matched_comparison.py" --run-id "$RUN_ID"
python3 "$EXP/python/diagnose_completed_run.py" --run-id "$RUN_ID"
```

If replay logs and event files already exist, regenerate only the derived
analysis after an analyzer update:

```bash
BUILD=0 RUN_ID="623_offline_lstm_spp_keyed_crn_joint_fill_v15_seed7" \
  bash formal_NN_training/experiments/623_offline_lstm_spp/linux/launch_server.sh analyze
```
