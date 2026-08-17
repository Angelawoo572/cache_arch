# cache_arch

A matched-input, trace-driven study of offline LSTM cache prefetchers in
ChampSim for `602.gcc_s-734B`, evaluated against Stride, Streamer, AMPM, and
SPP.

## Start here

- [Project website](https://angelawoo572.github.io/cache_arch/)
- [Quick presentation: `602_deck_v7_15.pdf`](https://angelawoo572.github.io/cache_arch/602_deck_v7_15.pdf)
- [All 602 experiments](formal_NN_training/experiments/)
- [Shared implementation](formal_NN_training/common/)

The primary project artifacts are intentionally concentrated in two places:

- `formal_NN_training/experiments/` contains the presentation and the four
  complete 602 experiment tracks.
- `formal_NN_training/common/` contains shared model, policy, validation,
  installation, keyed-sampling, and archive-transfer helpers.

## 602 experiment source

| Conventional comparator | Experiment directory | Neural runtime input boundary |
| --- | --- | --- |
| Stride | [`602_offline_lstm_stride`](formal_NN_training/experiments/602_offline_lstm_stride/) | Current PC and cache-line address |
| Streamer | [`602_offline_lstm_streamer`](formal_NN_training/experiments/602_offline_lstm_streamer/) | Cache-line address |
| AMPM | [`602_offline_lstm_ampm`](formal_NN_training/experiments/602_offline_lstm_ampm/) | Cache-line address |
| SPP | [`602_offline_lstm_spp`](formal_NN_training/experiments/602_offline_lstm_spp/) | Chronological demand and cache-fill address callbacks |

Each experiment directory is self-contained: begin with its README, then use
its track-specific collection, training, offline inference, replay, analysis,
contract, and report files.

## Comparison boundary

For every primary comparison, the offline conventional prefetcher and offline
LSTM consume the same chronological source-visible input stream and their
actions are replayed through the same keyed queue/cache path. Conventional
actions may be supervised labels, but conventional candidates, request budgets,
private state, and future labels are not neural runtime inputs.

This is a same-input comparison, not a requirement that the LSTM reproduce the
conventional prefetcher's internal algorithm. End-to-end replay metrics remain
the final evaluation boundary.

## Repository map

```text
formal_NN_training/
├── experiments/
│   ├── 602_deck_v7_15.pdf
│   ├── 602_offline_lstm_stride/
│   ├── 602_offline_lstm_streamer/
│   ├── 602_offline_lstm_ampm/
│   └── 602_offline_lstm_spp/
└── common/

docs/
├── index.html              # GitHub Pages project website
└── 602_deck_v7_15.pdf      # Pages-hosted copy of the presentation
```

Other repository content supports development history and adjacent experiments;
the paths above are the public entry points for the completed 602 study.
