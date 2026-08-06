# 623 SPP v23 — finite joint rank actions

This is the active matched-input, open-loop SPP experiment for
`623.xalancbmk_s-700B`.

- Run: `623_offline_lstm_spp_finite_joint_rank_v23_seed7`
- Model: `global_chronological_lstm_finite_joint_rank_v23`
- Decoder: `train_derived_horizon_joint_action_prior_corrected_map_v23`

The v22 hard policy collapsed to the same action list at h8 through h128:
every demand emitted one request and every request used `FILL_LLC`. That
collapse happened before analysis; the analyzer only exposed it. v23 removes
the factorized gate/count/delta/fill decision path.

## Input remains unchanged

The NN consumes only the chronological 59-bit source-visible callback stream:

- one `DEMAND/FILL` kind bit;
- the lossless 58-bit demand line or evicted line.

PC is replay transport only. Teacher targets, fill choices, SPP candidates,
thresholds, signatures, confidence, page state, queue state, and request rates
are not inputs. Captured actions are output labels and the offline-normal
comparator. The v22 input directory and archive are reused byte-for-byte.

Because `CACHE_FILL(evicted_addr)` was recorded under source SPP, the valid
claim remains matched-input open-loop replay, not closed-loop live NN.

## Direct joint labels

At each rank the teacher label is one categorical token:

```text
STOP
EMIT(exact TRAIN delta or OTHER, FILL_L2)
EMIT(exact TRAIN delta or OTHER, FILL_LLC)
```

For a realized exact vocabulary of size `V`, there are
`T = 1 + 2*(V+1)` tokens. `OTHER` uses an auxiliary signed-log coordinate only
when its joint token is the label or prediction. There are no separate gate,
count, delta, or fill argmaxes and no previous teacher/predicted action input.

The action horizon is computed from TRAIN:

```text
H = maximum teacher action count observed in TRAIN
decision ranks = H
```

For a teacher list of length `k < H`, ranks below `k` receive exact joint EMIT
labels and every available rank from `k` through `H-1` receives STOP. A
maximum-length sequence occupies all `H` ranks and terminates at the end of the
finite support. Inference checks only these data-derived ranks and ends at the
first STOP or finite support. This is not a copied SPP degree, tuned budget, or
hard-coded action count.

## Loss/decode agreement

One joint cross-entropy directly trains replay actions. TRAIN rank-slot labels
define three groups: `STOP`, `EMIT_L2`, and `EMIT_LLC`. Their training weights
are computed as `N/(3*N_group)`. Deterministic decode removes this weighting:

```text
natural_joint_logit[token]
  = weighted_joint_logit[token] - log(TRAIN_group_weight[token])
```

Then one joint-token argmax is taken. No probability threshold, fill cutoff,
page rule, normal request rate, or policy template is used.

Checkpoint selection is guard-only and lexicographic: joint-action F1, target
F1, L2 joint F1, trigger F1, exact count, matched-target fill accuracy, lower
TRAIN loss, then earlier epoch. Evaluation is decoded once after selection.
The full `h8, h16, h32, h64, h128` sweep remains mandatory.

With hidden size `H`, the realized parameter formula is:

```text
9*H^2 + (74+T)*H + T + 1
```

## Diagnostic control and analyzer semantics

Every capacity also exports the same explicitly non-neural control:

```text
every evaluation callback -> one TRAIN-modal delta, FILL_LLC
```

Only the delta is TRAIN-derived; the one-action/all-LLC behavior is deliberately
fixed to test whether the v22 IPC change came from a trivial aggressive policy.
It is excluded from neural claims.

The analyzer reports action-list SHA-256 and prior-corrected joint-token entropy
per capacity. If all five hard lists match, it emits a collapse warning. When a
method emits zero `FILL_L2` actions, L2-oriented selected accuracy, coverage,
and timeliness are marked `N/A`; raw counters remain available.

## Validation

```bash
python3 formal_NN_training/experiments/623_offline_lstm_spp/python/model_contract.py --self-test
python3 formal_NN_training/experiments/623_offline_lstm_spp/python/train_and_offline_infer.py --self-test
python3 formal_NN_training/experiments/validate_direct_action_contracts.py
```

Use `colab/623_offline_lstm_spp_A100.ipynb` for training and
`linux/launch_server.sh replay` for ChampSim. A root `PASS` establishes fair
input, deterministic contract compliance, and replay accounting—not an IPC
win.
