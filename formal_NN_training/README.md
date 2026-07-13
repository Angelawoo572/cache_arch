# formal_NN_training

The current experiment is:

```text
experiments/602_offline_lstm_stride/
```

Research question:

> With the same 602 evaluation PC/address stream, the same causal history, one future-stride candidate, and the same keyed replay transport, does a 545-parameter LSTM select a more useful prefetch list than offline stride?

The current workflow is intentionally one trace and one baseline:

```text
Linux collect
  -> 0-20M training PC/address stream
  -> 25M-warmup + 25M evaluation PC/address stream

Colab A100
  -> train tiny LSTM on the training stream
  -> offline causal inference on the evaluation stream
  -> export offline_stride.replay.csv
  -> export offline_lstm.replay.csv

Linux replay
  -> same ListReplayer binary for both lists
  -> no-prefetch and live stride retained as references
  -> matched_comparison.json must report PASS
```

The LSTM and offline stride receive only current PC, current cache-line address, and causal state derived from prior PC/address rows. Hit/miss, cycle, queue occupancy, metadata, and future evaluation rows are excluded.

All prior numbered scripts were moved to `legacy/scripts/`. They remain available only for provenance and are not part of the current experiment.
