# 623 SPP v24 — natural callback cardinality

This is the active matched-input, open-loop SPP experiment for
`623.xalancbmk_s-700B`.

- Run: `623_offline_lstm_spp_natural_cardinality_v24_seed7`
- Model: GUARD-selected global or event-routed chronological LSTM
- Decoder: categorical count followed by conditional joint actions

v23 padded every callback to ten rank labels. That changed a callback-level
problem in which 58.05% of EVAL callbacks had actions into a token stream that
was 93.40% STOP. The resulting models emitted between 0 and 5,096 requests,
recovered at most 0.63% of teacher actions, and all matched no-prefetch IPC.
v24 removes the artificial STOP target rather than tuning its weight.

## Unchanged matched input

The NN still receives only the chronological 59-bit source-visible stream:

- one `DEMAND/FILL` kind bit;
- the lossless 58-bit demand line or evicted line.

PC is replay transport only. Teacher targets and fills, SPP candidates,
signature/GHR/confidence state, queue state, hit state, thresholds, and request
rates are not inputs. The v23 input directory and archive are reused
byte-for-byte. Because `CACHE_FILL(evicted_addr)` was recorded under source
SPP, the claim remains matched-input open-loop replay, not closed-loop live NN.

## Natural action-list likelihood

For callback context `h` and teacher list of length `K`, the model learns

```text
P(list | h) = P(K | h) * product over r<K of P(action_r | h, r)
```

`K` is an unweighted categorical class from zero through the maximum TRAIN
teacher count. `K=0` is the implicit no-request decision. The action vocabulary
contains only joint `(signed delta, fill)` pairs observed in TRAIN, capped by
the declared architecture budget, plus `OTHER_L2` and `OTHER_LLC`. OTHER uses
an auxiliary signed-log coordinate.

There is no STOP token or tail padding, hurdle head, count regression, class
weight, decode prior correction, threshold, degree cap, page rule, normal
request budget, or previous-action feedback. At inference, count argmax chooses
`K`, then exactly `K` independent rank-conditioned action argmaxes are emitted.

Checkpoint selection minimizes natural action-list NLL on GUARD, with earlier
epoch as the only tie-break. EVAL is not read during checkpoint or core
selection.

## Recurrent-core ablation

At h32 only, the notebook trains:

- `global`: one ordinary chronological LSTM;
- `event_routed`: one chronological hidden/cell state, with distinct learned
  DEMAND and FILL LSTM transitions selected only by the existing kind bit.

The lower GUARD natural action-list NLL wins; global wins an exact tie. The
selected core is then trained at h8, h16, h32, h64, and h128. The two ablation
checkpoints are architecture-selection evidence only and are never replayed.

For hidden size `H`, count classes `K`, and joint-action classes `A`, parameter
counts are:

```text
global:       9*H^2  + (74+K+A)*H + K+A+1
event_routed: 17*H^2 + (82+K+A)*H + K+A+1
```

## Diagnostics and claims

Each final model records count confusion, count/action entropy, request ratio,
target/fill metrics, and two diagnosis-only decompositions:

- oracle count + NN action;
- NN count + oracle action upper bound.

Neither oracle path is replayed. The all-callback TRAIN-modal-delta/LLC policy
remains a separate non-neural control and cannot support a neural win claim.
The analyzer reports a denominator-zero metric as N/A while preserving a real
zero numerator when its denominator is positive. No composite score is used.

## Run entrypoints

```bash
python3 formal_NN_training/experiments/623_offline_lstm_spp/python/model_contract.py --self-test
python3 formal_NN_training/experiments/623_offline_lstm_spp/python/train_and_offline_infer.py --self-test
python3 formal_NN_training/experiments/validate_direct_action_contracts.py
```

Use `linux/run_server.sh` with `STAGE=reuse-input` to reuse v23 input,
`colab/623_offline_lstm_spp_A100.ipynb` to train, and
`linux/launch_server.sh replay` for ChampSim. A root PASS proves the input and
accounting contract, not an IPC win.
