# 623 matched stride/SPP: LSTM versus sliding CNN

This experiment contains two separate fair comparisons:

- `offline_stride` versus `offline_stride_lstm_*` and `offline_stride_cnn_*`
- `offline_spp` versus `offline_spp_lstm_*` and `offline_spp_cnn_*`

AMPM is intentionally excluded. On the historical 623 baseline, AMPM IPC
(`0.34924`) is below no-prefetch IPC (`0.35321`), while stride (`0.35340`) and
SPP (`0.35391`) are the relevant normal-policy reference points.

## CNN architecture

The CNN is a shallow causal sliding-window model matching the professor sketch:

- one temporal `Conv1d` layer;
- kernel/window size: 3 demand events;
- stride: 1 event;
- dilation: 1;
- left-only padding: 2 events;
- receptive field: `[t-2, t-1, t]`.

There is no dilated TCN stack. The three parameter-paired capacity points use
8, 16, and 32 CNN output channels. The temporal output is joined with each
candidate's address/rank features and passed through a pointwise candidate
head. Thus the convolution discovers short local address correlation, while
the final head decides whether to keep each candidate from the matched normal
prefetcher. Runtime self-tests enforce exactly one temporal convolution,
kernel 3, stride 1, no future dependence, and chunk/full-sequence equivalence.

Parameter-matched pairs are `LSTM-h4` (213) vs `CNN-c8` (233), `LSTM-h8`
(585) vs `CNN-c16` (593), and `LSTM-h15` (1621) vs `CNN-c32` (1697).

## Fair-input contract

The model receives only cache-line-derived causal features and candidate
address/rank features. PC and occurrence are replay transport identity only.
Cycle, hit/miss, queue state, candidate acceptance/duplicate outcomes, and
future evaluation rows are forbidden model inputs.

The fixed input vectors are:

- demand: page offset, signed/log line delta, same-page bit, page delta;
- candidate: signed target-line delta, target page offset, candidate rank;
- transport only, never model input: PC, PC-line occurrence, event ID;
- forbidden: cycle, hit/miss, queue occupancy, accepted/duplicate result.

Candidate rank uses the predeclared transform
`min(candidate_rank, 32) / 32`; it is not scaled from evaluation statistics.
The `623_causal_trigger_v4` logger writes each completed L2 demand before its
synchronous normal-prefetcher callback. Every PF row carries that exact
`trigger_event_id`; normalization also checks trigger CPU, PC, line, cycle,
and `base_addr`. There is no future-demand or address-only fallback. A stalled
RQ retry is not allowed to create a second model timestep.

Each policy is collected separately. Its offline normal list and all gated
neural lists use the same captured candidate bank. A neural model may suppress
normal-policy candidates but cannot invent candidates.

SPP is fully included, not deferred: collection runs `spp_dev2` with fill
threshold 90 and prefetch threshold 40 for train/guard/eval; Colab trains all
six SPP models; replay compares only those SPP-gated lists with
`offline_spp`. Stride and SPP are never normalized against each other.

## Server stages

```bash
cd ~/cache
git pull --ff-only

export RUN_ID=623_offline_lstm_cnn_stride_spp_seed7
export EXP="$HOME/cache/formal_NN_training/experiments/623_offline_lstm_cnn_stride_spp"
export RUN_DIR="$EXP/runs/$RUN_ID"

RESET_PATCH=1 FORCE=1 BUILD=1 bash "$EXP/linux/launch_server.sh" collect

tail -f "$RUN_DIR/collect.nohup.log"
```

Upload the generated `*.colab_input.tar.gz` archive and run
`colab/623_offline_lstm_cnn_stride_spp_A100.ipynb`. After copying and
extracting the Colab output archive into `"$RUN_DIR/colab_output"`, run:

```bash
mkdir -p "$RUN_DIR/colab_output"
tar -xzf "$RUN_DIR/$RUN_ID.colab_output.tar.gz" \
  -C "$RUN_DIR/colab_output"

RESET_PATCH=1 FORCE=1 BUILD=1 bash "$EXP/linux/launch_server.sh" replay

tail -f "$RUN_DIR/replay.nohup.log"
```

The analyzer emits IPC, L2 load miss rate, selected accuracy, coverage,
timeliness, and a balanced parity index normalized independently to each
track's own offline normal policy. It also emits:

- `matched_comparison.csv/json`: complete counters and provenance;
- `insight_summary.csv`: compact normal-vs-NN metric gaps and bottleneck;
- `architecture_pair_summary.csv`: CNN-minus-LSTM deltas at matched size.

Collection also emits `colab_input/collection_manifest.json`. It must report
`PASS`, schema `623_causal_trigger_v4`, and explicit trigger attachment for all
six policy/window files. It records stride/SPP demand-identity equality as a
diagnostic, not a fairness requirement: stride and SPP remain independent
matched tracks and may perturb callback timing/order differently.

The live stride/SPP rows are validation references. The primary claims use
the matched offline normal list and its NN-suppressed subsets through the same
PC-line-occ replay transport.
