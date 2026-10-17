# cache_arch

Teacher-action LSTM prefetcher studies in ChampSim for `602.gcc_s-734B`: original offline training and keyed replay, followed by frozen stateful C++ live inference. Stride, Streamer, AMPM and SPP keep their own model and measurement contracts. Measured MAC0 IPC and earlier finite-engine experiments remain separate protocols.

## Start here

- [H16 recovery and capped H8-template supplement](formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/README.md)
- [Canonical cache layout and recovery](docs/602_width_templates/LAYOUT_AND_RECOVERY.md)
- [Corrections to retained replay slides](docs/602_width_templates/SLIDE_ERRATA.md)

- [2026-10-16 formal current-line / gate / payload results](docs/602_gate_mechanism/REPORT.md)
- [Full-ROI mechanism run and recovery guide](formal_NN_training/experiments/602_gate_mechanism/RUN_GUIDE.md)
- [Project website](https://angelawoo572.github.io/cache_arch/)
- [Quick presentation: `602_deck_v7_15.pdf`](https://angelawoo572.github.io/cache_arch/602_deck_v7_15.pdf)
- [All 602 experiments](formal_NN_training/experiments/)
- [Shared implementation](formal_NN_training/common/)

The new mechanism follow-up uses frozen live inference on two future ROIs,
three existing training seeds and a validation-selected request ceiling. Its
MAC4 subset models a serial finite inference engine; MAC0 is a functional
comparison. Historical replay results below retain their own protocol and are
not pooled with these formal results.

The primary project artifacts are intentionally concentrated in two places:

- `formal_NN_training/experiments/` contains the presentation and the four
  complete 602 experiment tracks.
- `formal_NN_training/common/` contains the shared model and policy helpers
  used by the four 602 tracks.

## 602 experiment source

The frozen live continuation is in
[`602_four_prefetcher_formal`](formal_NN_training/experiments/602_four_prefetcher_formal/README.md),
with independent Streamer, AMPM and SPP live-inference siblings beside the
existing Stride live/dynamic tracks. It preserves the real checkpoints,
global versus per-PC state, SPP demand/fill inputs and L2/LLC output targets.
Its new MAC0 measurements use a separate protocol; prior offline, live and
formal results remain available and their IPC is not pooled across protocols.
605 is deferred in this continuation.

The [first frozen Stride formal archive](formal_NN_training/experiments/602_lstm_stride_dynamic_inference/results/formal_20261016_v2/README.md)
contains its lightweight tables, figures, definitions and findings. The
[forward/dependency recovery guide](formal_NN_training/experiments/602_lstm_stride_dynamic_inference/runtime/FORWARD_REPRODUCTION.md)
explains runtime generation and checks against the sources actually built.

| Conventional comparator | Experiment directory | Neural runtime input boundary |
| --- | --- | --- |
| Stride | [`602_offline_lstm_stride`](formal_NN_training/experiments/602_offline_lstm_stride/) | Current PC and cache-line address |
| Streamer | [`602_offline_lstm_streamer`](formal_NN_training/experiments/602_offline_lstm_streamer/) | Cache-line address |
| AMPM | [`602_offline_lstm_ampm`](formal_NN_training/experiments/602_offline_lstm_ampm/) | Cache-line address |
| SPP | [`602_offline_lstm_spp`](formal_NN_training/experiments/602_offline_lstm_spp/) | Chronological demand and cache-fill address callbacks |

Each experiment directory is self-contained: begin with its README, then use
its track-specific collection, training, offline inference, replay, analysis,
contract, and report files.

## Historical replay comparison boundary

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

The paths above are the public entry points for the completed 602 study.
