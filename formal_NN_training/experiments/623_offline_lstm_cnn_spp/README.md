# 623 direct SPP action prediction: matched LSTM versus three-step CNN

This is the independent **SPP** track for `623.xalancbmk_s-700B`.  It is not
the former SPP-candidate gate and it must not reuse outputs from
`623_offline_lstm_cnn_stride_spp` or `spp_gate_sliding_cnn_v1`.

## What is compared

The primary rows are:

- `offline_spp`: the actions emitted by the pinned normal `spp_dev2` policy;
- `offline_direct_spp_lstm_h{4,8,16}`;
- `offline_direct_spp_cnn_c{5,10,24}`.

`no_pref` and `live_spp_reference` are context and transport checks.  Every
offline row is keyed by the same `(PC,line,occurrence)` transport and is replayed
by the same fill-preserving replayer.

The neural models do **not** receive an SPP candidate bank.  They directly
produce one of 128 actions: 64 target offsets in the current 4 KiB page times
`FILL_L2`/`FILL_LLC`.  A model may produce an action that normal SPP did not
emit.  The captured SPP actions are teacher labels during training, the normal
offline comparator during evaluation, and an action-fidelity reference only.
They are never evaluation-time model inputs.

All 64 offsets are real output classes, including the trigger's own offset.
The audited source can return to that line after multiple lookahead deltas and
contains no `pf_addr != addr` guard.  Such self-target calls are therefore
retained for normal SPP, neural decoding, and replay; removing them would make
the comparator cleaner than the source policy and bias the fidelity metrics.

If source SPP calls `prefetch_line` repeatedly for the same target during one
callback, normalization applies ChampSim's queue-visible rule: one canonical
target is retained and the minimum fill level wins (`FILL_L2=2` dominates
`FILL_LLC=4`).  Raw-call, collapsed-call, and self-target counts remain in the
collection manifest for diagnosis.

## Source-derived input contract

The checked `SPP_dev2::invoke_prefetcher(ip, addr, cache_hit, type, ...)` source
uses `addr` to update ST/PT/GHR/FILTER and generate `pf_addr`.  `ip` is passed
only to `prefetch_line`; `cache_hit` and `type` are present in the signature but
are not read by its prediction path.  `cache_fill()` supplies private filter
feedback to normal SPP.

To preserve the existing restricted-input requirement, both neural models see
only the causal address sequence.  Each line-aligned `addr` becomes nine
features:

- current page offset;
- four fixed 16-bit chunks of the page number (no trace-fitted normalization);
- signed line delta, log absolute line delta, same-page bit, and signed page
  delta from the immediately prior callback.

PC, hit/miss, access type, cycle, queue state, acceptance/duplicate outcomes,
SPP tables/confidence/filter state, teacher actions, and future evaluation rows
are forbidden.  The recurrent/sliding state approximates SPP's private history;
this is a direct I/O student, not a bit-identical replacement of SPP's tables.

## CNN and matched capacities

The CNN matches the professor's short moving-filter sketch exactly: one causal
`Conv1d`, kernel 3, stride 1, dilation 1, two-event left padding, no pooling,
no second convolution, and no TCN stack.  Output `t` depends only on
`[t-2,t-1,t]`.

Parameter-matched pairs are:

- LSTM-h4 (880) versus CNN-c5 (908);
- LSTM-h8 (1760) versus CNN-c10 (1688);
- LSTM-h16 (3904) versus CNN-c24 (3872).

The LSTM uses chronological stateful TBPTT: hidden/cell values cross chunk
boundaries and are detached only to truncate the gradient graph.  The guard
window initializes recurrent state before evaluation; it is not training data.

## Windows and calibration

- 0--20M instructions: train stream; first 80% fits, final 20% calibrates the
  action threshold without exceeding normal SPP's action/event budget;
- 20--25M: guard context for LSTM state or the CNN's final two rows;
- 25--50M: untouched direct inference, action-fidelity audit, and replay.

Normal SPP is pinned to `spp_dev2`, fill threshold 90, and PF threshold 40.
Collection fails on an incomplete logger schema, noncausal action attachment,
cross-page SPP output, invalid fill level, more than the audited 32 raw actions
per callback, or dropped prefetch requests.  A source-legal self target is not
an error.

## Run

```bash
export RUN_ID=623_offline_lstm_cnn_spp_direct_seed7
export EXP="$HOME/cache/formal_NN_training/experiments/623_offline_lstm_cnn_spp"
export RUN_DIR="$EXP/runs/$RUN_ID"

FORCE=1 BUILD=1 bash "$EXP/linux/launch_server.sh" collect
tail -f "$RUN_DIR/collect.nohup.log"
```

When raw event logs already exist and only normalization changed, use
`FORCE=0 BUILD=0`; this reuses the three validated event logs without rerunning
ChampSim.

Upload `$RUN_DIR/$RUN_ID.colab_input.tar.gz` and run
`colab/623_offline_lstm_cnn_spp_A100.ipynb`.  Return the output archive, then:

```bash
mkdir -p "$RUN_DIR/colab_output"
tar -xzf "$RUN_DIR/$RUN_ID.colab_output.tar.gz" -C "$RUN_DIR/colab_output"

FORCE=1 BUILD=1 bash "$EXP/linux/launch_server.sh" replay
tail -f "$RUN_DIR/replay.nohup.log"

python3 -m json.tool "$RUN_DIR/matched_comparison.json"
column -s, -t < "$RUN_DIR/insight_summary.csv"
column -s, -t < "$RUN_DIR/architecture_pair_summary.csv"
```

The report includes IPC/speedup, L2 load miss rate, selected accuracy, coverage,
timeliness/late requests, request rate, action precision/recall/F1/Jaccard, and
the equal-weight within-track BPI:

`BPI = 100 * (q_miss_rate * q_accuracy * q_coverage * q_timeliness)^(1/4)`.

Each `q` is capped at one relative to `offline_spp`; BPI summarizes parity but
does not replace IPC or direct cache-pollution measurements.
