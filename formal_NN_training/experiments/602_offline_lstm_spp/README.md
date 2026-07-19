# 602 source-input-fair SPP LSTM

This directory is an independent 602.gcc_s-734B experiment. Its primary
comparison is **offline SPP vs. offline LSTM-SPP**, both generated causally from
the same captured SPP-visible callback stream and replayed through the same
PC-line-occurrence, fill-preserving transport.

## Fair input contract

Source audit of `spp_dev2.cc` shows that prediction consumes
`invoke_prefetcher(addr)`, while `cache_fill(evicted_addr)` changes SPP
feedback state. The NN therefore receives only the chronological raw callbacks:

- `DEMAND(addr)`
- `CACHE_FILL(evicted_addr)`

PC is only a replay key. `cache_hit`, access type, cycle, queue state, source
signature/pattern/filter/GHR state, source outputs, thresholds 90/40, degree,
future rows, and same-page/page-offset restrictions are excluded from NN
training and inference inputs. Training and inference invoke the same lossless
65-feature encoder and record one code hash.

Eviction feedback is included as raw input because source SPP reads it. There is
no separate eviction-prediction head: eviction is not an SPP output action, and
adding such a target would not improve input parity.

## NN design

A single stateful LSTM follows the complete callback chronology. At demand
callbacks its compact learned decoder emits:

1. zero versus positive requests by categorical argmax (no probability
   threshold);
2. an unbounded positive request count from a learned log-count;
3. a free-running autoregressive four-component mixture over direct signed
   cache-line deltas;
4. learned `FILL_L2` versus `FILL_LLC`.

Teacher actions compute losses only. Decoder feedback always uses the model's
own modal delta and fill in both training and inference. There is no fixed
candidate table, action budget, page rule, threshold, or degree cap. Capacity
points h8/h16/h32/h64/h128 contain
2,865/6,609/16,785/47,889/153,105 trainable parameters.

## Workflow

Default run: `602_offline_lstm_spp_compact_hurdle_free_running_v1_seed7`.

```bash
cd "$HOME/cache"
python3 formal_NN_training/experiments/validate_direct_action_contracts.py
export RUN_ID=602_offline_lstm_spp_compact_hurdle_free_running_v1_seed7
export EXP="$HOME/cache/formal_NN_training/experiments/602_offline_lstm_spp"
export RUN_DIR="$EXP/runs/$RUN_ID"
FORCE=1 BUILD=1 RESET_PATCH=1 bash "$EXP/linux/launch_server.sh" collect
tail -f "$RUN_DIR/collect.nohup.log"
```

Run `colab/602_offline_lstm_spp_A100.ipynb`, return its output archive to
this same run, extract it into `colab_output`, and launch `replay`. The
analyzer fails closed unless source audit, manifest, content hashes, encoder
hashes, model metadata, fill-preserving transport, and simulator outputs agree.
