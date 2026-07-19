# 602 source-input-fair SPP LSTM

This directory compares **offline SPP** with an independent offline LSTM
prefetcher on 602.gcc_s-734B. Both action lists are generated causally from the
same captured SPP-visible callback stream and replayed through the same
PC-line-occurrence, fill-preserving transport.

## Fair input contract

Source audit of `spp_dev2.cc` shows that prediction consumes
`invoke_prefetcher(addr)`, while `cache_fill(evicted_addr)` changes SPP
feedback state. The NN therefore receives only the chronological raw callbacks:

- `DEMAND(addr)`
- `CACHE_FILL(evicted_addr)`

Training and inference use the same lossless 65-feature encoder and record one
code hash. PC is a replay key only. Cache hit/type, cycle, queue state, SPP
signature/pattern/filter/GHR state, source outputs, thresholds 90/40, degree,
future rows, and page-offset restrictions are excluded from NN inputs.

Eviction feedback is retained only as the raw callback that SPP can observe.
There is no eviction target or private cache/SPP state in the model.

## v1 diagnosis and v2 controlled fix

The v1 h128 result had better selected accuracy than offline SPP
(7.29% versus 5.87%) and nearly identical timeliness (99.65% versus 99.71%),
but only 45.34% coverage versus 65.11%. It produced 1.307M actions while the
normal comparator produced 2.254M and recovered only 46.23% of target actions.

The strongest identified cause is the zero/positive gate objective. Only
2.76% of evaluation callbacks had zero SPP actions, while v1 applied
inverse-frequency class balancing before raw categorical argmax. This
combination is consistent with the observed conservative under-emission. The
old archive did not record the exact training positive/zero split, so it does
not support an exact training-weight ratio; v2 records that split explicitly.
For reference, applying the same weighting to the held-out split would give a
zero label about 35 times the positive-label weight. The v2 run removes class
weighting and fits the observed zero/positive prior with unweighted categorical
maximum likelihood.

This is a one-variable ablation: external inputs, h8/h16/h32/h64/h128
capacities, positive-count head, direct-delta mixture, learned fill head,
training chronology, and free-running decoder are unchanged.

## NN design

At each demand callback, the stateful LSTM decoder produces:

1. zero versus positive requests by two-class categorical argmax;
2. an unbounded learned positive request count;
3. free-running autoregressive direct signed cache-line deltas from a
   four-component mixture;
4. learned `FILL_L2` versus `FILL_LLC`.

There is no probability threshold, candidate table, request budget, page rule,
or degree cap. Teacher actions are supervision only; decoder feedback always
uses the model's own modal delta and fill. Parameter counts remain
2,865/6,609/16,785/47,889/153,105.

## Workflow

Default run:
`602_offline_lstm_spp_empirical_prior_hurdle_free_running_v2_seed7`.

```bash
cd "$HOME/cache"
git pull --ff-only origin main
python3 formal_NN_training/experiments/validate_direct_action_contracts.py

export RUN_ID=602_offline_lstm_spp_empirical_prior_hurdle_free_running_v2_seed7
export EXP="$HOME/cache/formal_NN_training/experiments/602_offline_lstm_spp"
export RUN_DIR="$EXP/runs/$RUN_ID"

FORCE=1 BUILD=1 RESET_PATCH=1 bash "$EXP/linux/launch_server.sh" collect
tail -f "$RUN_DIR/collect.nohup.log"
```

Run `colab/602_offline_lstm_spp_A100.ipynb`, return its output archive to the
same run, extract it into `colab_output`, and launch `replay`. The analyzer
fails closed unless source audit, manifest, content hashes, encoder hashes,
gate objective, model metadata, fill-preserving transport, and simulator
outputs agree.
