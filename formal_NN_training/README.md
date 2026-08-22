# 602 matched-input offline LSTM prefetching

This directory contains the completed `602.gcc_s-734B` study. It compares four
offline LSTM prefetchers with matched conventional Stride, Streamer, AMPM, and
SPP comparators in ChampSim.

## Start here

- [`experiments/602_offline_lstm_stride/`](experiments/602_offline_lstm_stride/)
- [`experiments/602_offline_lstm_streamer/`](experiments/602_offline_lstm_streamer/)
- [`experiments/602_offline_lstm_ampm/`](experiments/602_offline_lstm_ampm/)
- [`experiments/602_offline_lstm_spp/`](experiments/602_offline_lstm_spp/)
- [`experiments/602_deck_v7_15.pdf`](experiments/602_deck_v7_15.pdf)

Each experiment keeps its collection, training, offline inference, replay,
analysis, stream contract, and report files together. Shared 602 model and
policy code lives in `common/`.

## Fairness boundary

For every primary comparison, the offline conventional prefetcher and offline
LSTM consume the same chronological source-visible input stream. Their actions
are replayed through the same keyed queue/cache path.

Conventional actions may be used as supervised labels. Conventional
candidates, request budgets, private state, and future labels are never neural
runtime inputs. Teacher or decoded actions are not fed back into the decoder.
This is a same-input comparison, not a copy of the conventional prefetcher's
internal algorithm.

## Track map

| Comparator | Neural runtime input boundary | Experiment |
| --- | --- | --- |
| Stride | Current PC and cache-line address | [`602_offline_lstm_stride`](experiments/602_offline_lstm_stride/) |
| Streamer | Cache-line address | [`602_offline_lstm_streamer`](experiments/602_offline_lstm_streamer/) |
| AMPM | Cache-line address | [`602_offline_lstm_ampm`](experiments/602_offline_lstm_ampm/) |
| SPP | Chronological demand and cache-fill address callbacks | [`602_offline_lstm_spp`](experiments/602_offline_lstm_spp/) |

## Shared code

The retained `common/` modules are all used by the 602 tracks:

- `direct_action_lstm.py`: shared recurrent direct-action implementation;
- `normal_policy_reference.py`: conventional-policy reference utilities;
- `threshold_free_policy.py`: learned threshold-free policy components.

## Static validation

From the repository root, run:

```bash
python3 formal_NN_training/experiments/validate_direct_action_contracts.py
```

The validator checks all four stream contracts, runtime fairness boundaries,
free-running decoder metadata, required experiment files, Python syntax,
notebook syntax, and the SPP source-input boundary.
