# 602 teacher-supervised prefetchers

- **Current conclusions and reading order:** [RESULTS_INDEX](../RESULTS_INDEX.md).
- **Latest results:** [H8 and H16 native-versus-NN results](../formal_study/results/602_width_templates/README.md). Gains are limited to retained seed 7 models, 602, two ROIs and MAC0; simple templates explain much of the benefit. Finite-resource neural benefit remains unproven.
- Weights are frozen during live inference. Recurrent h/c changes with callbacks; no online weight training is used.
- Offline replay, synchronous live inference and finite-cost live inference are separate protocols. Their IPC values are not interchangeable.

## Experiments

| Route | Original offline/replay | Live implementation |
| --- | --- | --- |
| Stride | [Offline](experiments/602_offline_lstm_stride/) | [Original live](experiments/602_lstm_stride_live_inference/), [finite dynamic engine](experiments/602_lstm_stride_dynamic_inference/) |
| Streamer | [Offline](experiments/602_offline_lstm_streamer/) | [Global-state student](experiments/602_lstm_streamer_live_inference/README.md) |
| AMPM | [Offline](experiments/602_offline_lstm_ampm/) | [Global-state student](experiments/602_lstm_ampm_live_inference/README.md) |
| SPP | [Offline](experiments/602_offline_lstm_spp/) | [Demand/fill student](experiments/602_lstm_spp_live_inference/README.md) |

- [Four-route H8 protocol](experiments/602_four_prefetcher_formal/README.md): 64 completed rows, retained models, actual C++ inference and mechanism controls.
- [H16/template supplement](experiments/602_four_prefetcher_formal/width_templates/README.md): 24 new rows plus explicitly bound H8 references; no retraining.
- [Old slide/replay evidence](experiments/602_four_prefetcher_formal/width_templates/EVIDENCE_ALIGNMENT.md): historical state histories, request matching and corrected denominators.
- [Canonical layout and recovery](../docs/602_width_templates/LAYOUT_AND_RECOVERY.md): local/Sacramento cache roots, preserved archives and path translation.

## Input and fairness boundaries

- Teacher actions supply training labels only. Private teacher tables, future events and teacher actions are absent from neural inference inputs.
- Free-running decoders feed back their **own** selected unrounded coordinate; SPP also feeds back its own fill choice. Teacher actions are not decoder inputs.
- Streamer/AMPM encode 64 LSB-first bits of the aligned **byte address**, not line-address bits; each has one global chronological h/c pair and no PC feature.
- SPP adds a callback-kind bit and uses demand addresses or raw L2-fill evicted addresses. Its student sees all raw fills; native SPP FILTER feedback sees prefetch fills only. Effective state-update streams differ.
- The retained formal Stride model uses PC/address features and per-PC h/c. It was trained on 0–1M for 12 epochs; the three global models use 0–20M and 8 epochs. These routes are not a matched training-budget ranking.
- Historical offline comparisons replay actions through keyed queues on a shared recorded input stream. Live runs instead receive cache callbacks affected by their own prefetches.
- A common request ceiling limits output; it does not equalize realized requests or DRAM bytes. Report actual controller traffic and separate complete L2/LLC intent cohorts.

## Shared code and static check

- [common/](common/): direct-action models, conventional references, feature encoders and the scalar `live602_runtime` backend.
- Each experiment retains collection, training, inference, replay, contracts and reports together; original artifacts remain recoverable.
- From the cache root, the static validator checks contracts, runtime input boundaries, free-running metadata, required files and Python/notebook syntax:

```sh
python3 formal_NN_training/experiments/validate_direct_action_contracts.py
```

- Static validation or numerical parity does not replace full-ROI simulation evidence.
