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
8, 16, and 32 CNN output channels.

## Fair-input contract

The model receives only cache-line-derived causal features and candidate
address/rank features. PC and occurrence are replay transport identity only.
Cycle, hit/miss, queue state, candidate acceptance/duplicate outcomes, and
future evaluation rows are forbidden model inputs.

Each policy is collected separately. Its offline normal list and all gated
neural lists use the same captured candidate bank. A neural model may suppress
normal-policy candidates but cannot invent candidates.

## Server stages

```bash
cd ~/cache
git pull --ff-only

export RUN_ID=623_offline_lstm_cnn_stride_spp_seed7
export EXP="$HOME/cache/formal_NN_training/experiments/623_offline_lstm_cnn_stride_spp"
export RUN_DIR="$EXP/runs/$RUN_ID"

nohup env RUN_ID="$RUN_ID" FORCE=1 BUILD=1 STAGE=collect \
  bash "$EXP/linux/run_server.sh" \
  > "$RUN_DIR/collect.nohup.log" 2>&1 &

tail -f "$RUN_DIR/collect.nohup.log"
```

Upload the generated `*.colab_input.tar.gz` archive and run
`colab/623_offline_lstm_cnn_stride_spp_A100.ipynb`. After copying and
extracting the Colab output archive into `"$RUN_DIR/colab_output"`, run:

```bash
nohup env RUN_ID="$RUN_ID" FORCE=1 BUILD=1 STAGE=replay \
  bash "$EXP/linux/run_server.sh" \
  > "$RUN_DIR/replay.nohup.log" 2>&1 &

tail -f "$RUN_DIR/replay.nohup.log"
```

The analyzer emits IPC, L2 load miss rate, selected accuracy, coverage,
timeliness, and a balanced parity index normalized independently to each
track's own offline normal policy.
