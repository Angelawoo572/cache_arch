# cache_arch

The active research track is one isolated experiment:

```text
formal_NN_training/experiments/602_offline_lstm_stride/
```

It studies only `602.gcc_s-734B` and compares a tiny LSTM with stride under a shared offline-inference and keyed-replay contract.

- Colab performs LSTM training and offline inference.
- Colab also generates the matched offline-stride list from the same evaluation PC/address stream.
- Linux/ChampSim replays both lists through the same PC-line-occurrence `ListReplayer`.
- Live stride and no-prefetch are retained as general references, not the primary matched comparison.

Historical scripts are preserved under:

```text
formal_NN_training/legacy/scripts/
```

They are not imported or invoked by the active 602 experiment.
