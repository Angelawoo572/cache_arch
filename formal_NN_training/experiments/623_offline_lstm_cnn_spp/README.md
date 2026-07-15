# 623 SPP candidate gate: matched LSTM versus three-step CNN

This is the independent **SPP** track for `623.xalancbmk_s-700B`. Do not
continue a run created under the former combined
`623_offline_lstm_cnn_stride_spp` directory. SPP has different causal state and
replay semantics from stride, so it now has its own collection, archive,
validator, Colab notebook, binary, and analyzer.

## Exact comparison and claim boundary

Primary rows are `offline_spp`, `offline_spp_lstm_<size>`, and
`offline_spp_cnn_<channels>`. `no_pref` and `live_spp_reference` are
context/transport checks. Every neural model receives the same captured causal
SPP candidate-action bank and may only suppress actions; it cannot invent a
target or change its destination. This is an **SPP candidate gate**, not a
standalone neural reimplementation of SPP.

The normal `spp_dev2` policy causally maintains a Signature Table, Pattern
Table, GHR, confidence, and prefetch-filter usefulness feedback. Those private
states generate the fixed candidate bank but are not direct neural inputs. This
distinction is asserted in the contract, metadata, and analyzer.

SPP also chooses `FILL_L2` or `FILL_LLC` from confidence. The bundled
fill-preserving PC-line-occ replayer records a fifth replay column and reproduces
that action exactly. Fill level remains replay metadata and is forbidden as an
LSTM/CNN feature. This fixes the old four-column replayer's implicit `FILL_L2`
conversion.

## Fixed neural input

At demand event `t`, both LSTM and CNN receive exactly:

- demand vector: current page offset, signed line delta from `t-1`, log absolute
  delta, same-page bit, and signed page delta;
- candidate vector: signed target-line delta, target page offset, and fixed
  candidate rank (`min(rank,32)/32`).

Forbidden inputs include PC, cycle, hit/miss, queue state, all SPP private state
and confidence, candidate acceptance/duplicate result, fill level, and future
evaluation rows. Future demand reuse is used only for offline labels.

## CNN architecture

The CNN follows the professor's short moving-filter sketch:

- exactly one causal `Conv1d` temporal layer;
- kernel size 3, stride 1, dilation 1, left padding 2;
- output `t` depends only on `[t-2,t-1,t]`;
- no TCN stack, pooling, second temporal convolution, or future padding;
- the temporal output is joined with each candidate vector in a pointwise gate.

Parameter-matched pairs are LSTM-h4 (213) vs CNN-c8 (233), LSTM-h8
(585) vs CNN-c16 (593), and LSTM-h15 (1621) vs CNN-c32 (1697).

## Policy and windows

Normal SPP is pinned to `spp_dev2`, fill threshold 90, and prefetch threshold
40. Collection fails if any request is dropped, because an incomplete bank
cannot support the matched claim.

- 0--20M instructions: chronological training stream; first 80% fit, final 20%
  threshold calibration;
- 20--25M: guard stream for LSTM state initialization or two-event CNN context;
- 25--50M: untouched evaluation/replay window.

## Run

From `~/cache`, after pulling the commit containing this directory:

```bash
export RUN_ID=623_offline_lstm_cnn_spp_seed7
export EXP="$HOME/cache/formal_NN_training/experiments/623_offline_lstm_cnn_spp"
export RUN_DIR="$EXP/runs/$RUN_ID"

FORCE=1 bash "$EXP/linux/launch_server.sh" collect
tail -f "$RUN_DIR/collect.nohup.log"
```

The launcher creates `RUN_DIR` before redirecting. Under the shared ChampSim
build lock, a recognized older experiment logger is backed up and automatically
replaced; unknown `cache.cc` edits still fail closed.

Upload `$RUN_DIR/$RUN_ID.colab_input.tar.gz` and run
`colab/623_offline_lstm_cnn_spp_A100.ipynb`. Copy the downloaded output archive
back to `$RUN_DIR`, then:

```bash
mkdir -p "$RUN_DIR/colab_output"
tar -xzf "$RUN_DIR/$RUN_ID.colab_output.tar.gz" -C "$RUN_DIR/colab_output"

FORCE=1 BUILD=1 bash "$EXP/linux/launch_server.sh" replay
tail -f "$RUN_DIR/replay.nohup.log"

python3 -m json.tool "$RUN_DIR/matched_comparison.json"
column -s, -t < "$RUN_DIR/insight_summary.csv"
column -s, -t < "$RUN_DIR/architecture_pair_summary.csv"
```

`matched_comparison.json` is `PASS` only when provenance hashes, explicit
trigger attachment, SPP fill-level preservation, parameter matching,
candidate-bank identity, replay statistics, and all model metadata agree.

## Presentation metrics

The analyzer reports IPC/speedup, L2 load miss rate, selected accuracy,
coverage against no-prefetch L2 misses, timeliness, request reduction, and the
within-track Balanced Parity Index. It also preserves the normal and gated
`FILL_L2`/`FILL_LLC` counts in model metadata. The four BPI components have
equal geometric weight:

`BPI = 100 * (q_miss_rate * q_selected_accuracy * q_coverage * q_timeliness)^(1/4)`.

Each `q` is capped at 1 relative to offline SPP, so SPP itself scores 100 and
one strong component cannot hide a weak one. BPI is a compact parity summary;
IPC/speedup remain the outcome, and `1-selected_accuracy` is only a pollution
risk proxy rather than a direct harmful-eviction count.
