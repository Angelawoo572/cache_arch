# Source-input-fair direct-action rerun

## Contract

The normal prefetcher and its neural comparator receive the same effective
external input, in the same chronological order.  Training and inference call
the same runtime encoder and record the same encoder SHA256.  The autoregressive
decoder is also free-running in both modes: its previous input is its own
prediction, never the previous teacher action.

| Track | Normal comparator | Runtime input used by both policies | NN |
|---|---|---|---|
| 602 Stride | offline Stride | PC + aligned address | stateful LSTM |
| 602 Streamer | offline Streamer | aligned address | stateful LSTM |
| 602 AMPM | offline AMPM | aligned address | stateful LSTM |
| 623 Stride | offline Stride | PC + aligned address | separate LSTM and CNN runs |
| 623 SPP | offline SPP | chronological `DEMAND(addr)` and `CACHE_FILL(evicted_addr)` callbacks | separate LSTM and CNN runs |

Normal actions are labels and matched offline replay only.  Neural inference
does not receive normal candidates, thresholds, confidence, degree, request
budget, private tables, page-offset classes, same-page rules, or future rows.
It learns a Poisson request count and direct signed cache-line deltas; SPP also
learns the fill class.

The CNN is intentionally shallow: a 1x1 projection and two causal residual
temporal convolutions, each with 17 taps, stride 1, and dilations 1 and 17.
The contiguous receptive field is 289 callbacks, with exactly 288 prior
callbacks carried across a computation chunk.  The LSTM is one stateful layer
trained with chronological TBPTT; hidden/cell values cross chunks and only the
graph is detached.

`64` appears only as a hardware/interface fact where applicable: a 64-byte
cache line, a `uint64_t` address width, and source-only validation of the
normal prefetcher's page behavior.  It is not an NN action table, output cap,
threshold, or degree.  Model widths, epochs, optimizer settings, CNN kernel,
and dilation are ordinary reported model/training configuration.

## 1. Update and audit on Sacramento

```bash
cd ~/cache
git pull --ff-only origin main

python3 formal_NN_training/experiments/validate_direct_action_contracts.py
```

The audit must print three `[PASS]` lines.  Do not use output from an older run
ID.

## 2. Reuse unchanged 602 inputs and collect 623 inputs once per policy

```bash
cd ~/cache

bash formal_NN_training/experiments/stage_threshold_free_inputs.sh

RUN_ID=623_offline_lstm_stride_variable_delta_free_running_v9_seed7 \
RESET_PATCH=1 FORCE=1 BUILD=1 \
bash formal_NN_training/experiments/623_offline_lstm_stride/linux/launch_server.sh collect

RUN_ID=623_offline_lstm_spp_variable_delta_free_running_v11_seed7 \
RESET_PATCH=1 FORCE=1 BUILD=1 \
bash formal_NN_training/experiments/623_offline_lstm_spp/linux/launch_server.sh collect
```

These are two `nohup` jobs.  The shared build lock serializes ChampSim builds;
the jobs may be launched back-to-back.  Monitor them with:

```bash
tail -F \
  formal_NN_training/experiments/623_offline_lstm_stride/runs/623_offline_lstm_stride_variable_delta_free_running_v9_seed7/collect.nohup.log \
  formal_NN_training/experiments/623_offline_lstm_spp/runs/623_offline_lstm_spp_variable_delta_free_running_v11_seed7/collect.nohup.log
```

After both logs report a ready Colab archive, create byte-identical standalone
CNN archives:

```bash
cd ~/cache
bash formal_NN_training/experiments/stage_623_split_inputs.sh stride
bash formal_NN_training/experiments/stage_623_split_inputs.sh spp
```

## 3. Copy all seven input archives to the Mac

Run on the Mac:

```bash
mkdir -p ~/Documents/cache_prefetch_inputs_free_running

scp \
  qianruw@sacramento:/home/qianruw/cache/formal_NN_training/experiments/602_offline_lstm_stride/runs/602_offline_lstm_stride_variable_delta_free_running_v7_seed7/602_offline_lstm_stride_variable_delta_free_running_v7_seed7.colab_input.tar.gz \
  qianruw@sacramento:/home/qianruw/cache/formal_NN_training/experiments/602_offline_lstm_streamer/runs/602_offline_lstm_streamer_variable_delta_free_running_v7_seed7/602_offline_lstm_streamer_variable_delta_free_running_v7_seed7.colab_input.tar.gz \
  qianruw@sacramento:/home/qianruw/cache/formal_NN_training/experiments/602_offline_lstm_ampm/runs/602_offline_lstm_ampm_variable_delta_free_running_v7_seed7/602_offline_lstm_ampm_variable_delta_free_running_v7_seed7.colab_input.tar.gz \
  qianruw@sacramento:/home/qianruw/cache/formal_NN_training/experiments/623_offline_lstm_stride/runs/623_offline_lstm_stride_variable_delta_free_running_v9_seed7/623_offline_lstm_stride_variable_delta_free_running_v9_seed7.colab_input.tar.gz \
  qianruw@sacramento:/home/qianruw/cache/formal_NN_training/experiments/623_offline_cnn_stride/runs/623_offline_cnn_stride_variable_delta_free_running_v9_seed7/623_offline_cnn_stride_variable_delta_free_running_v9_seed7.colab_input.tar.gz \
  qianruw@sacramento:/home/qianruw/cache/formal_NN_training/experiments/623_offline_lstm_spp/runs/623_offline_lstm_spp_variable_delta_free_running_v11_seed7/623_offline_lstm_spp_variable_delta_free_running_v11_seed7.colab_input.tar.gz \
  qianruw@sacramento:/home/qianruw/cache/formal_NN_training/experiments/623_offline_cnn_spp/runs/623_offline_cnn_spp_variable_delta_free_running_v11_seed7/623_offline_cnn_spp_variable_delta_free_running_v11_seed7.colab_input.tar.gz \
  ~/Documents/cache_prefetch_inputs_free_running/
```

## 4. Run seven independent Colab notebooks

Open the notebook in each experiment's `colab/` directory.  Seven Colab
runtimes may run concurrently.  Each notebook clones/pulls this repository,
requires a GPU, asks for the exactly matching input archive, validates
`SHA256SUMS`, trains its own sweep, validates metadata, and writes
`<RUN_ID>.colab_output.tar.gz` to the printed Google Drive path.

Do not upload one architecture's archive into the other architecture's
notebook even though the contained 623 input bytes are identical; the archive
name is intentionally run-specific.

Download the seven output archives from Drive into:

```text
~/Documents/cache_prefetch_outputs_free_running
```

## 5. Copy and install Colab outputs on Sacramento

Run on the Mac:

```bash
ssh qianruw@sacramento 'mkdir -p ~/direct_action_uploads'
scp ~/Documents/cache_prefetch_outputs_free_running/*.colab_output.tar.gz \
  qianruw@sacramento:/home/qianruw/direct_action_uploads/
```

Run on Sacramento:

```bash
cd ~/cache
bash formal_NN_training/experiments/install_direct_action_outputs.sh \
  "$HOME/direct_action_uploads"
```

The installer fails if a revisioned `colab_output` directory is already
nonempty, preventing mixed old/new model files.

## 6. Launch all seven replay/analyze pipelines

```bash
cd ~/cache
FORCE=1 BUILD=1 JOBS=8 \
bash formal_NN_training/experiments/launch_direct_action_replays.sh replay
```

The seven launchers use `nohup`; the build lock serializes shared ChampSim
builds.  Every completed run must produce `matched_comparison.json` with
`status: PASS` and `fair_comparison_claim_allowed: true`.

## 7. Cross-directory LSTM/CNN checks

```bash
cd ~/cache

python3 formal_NN_training/experiments/compare_623_split_architectures.py \
  --policy stride \
  --lstm-run-dir formal_NN_training/experiments/623_offline_lstm_stride/runs/623_offline_lstm_stride_variable_delta_free_running_v9_seed7 \
  --cnn-run-dir formal_NN_training/experiments/623_offline_cnn_stride/runs/623_offline_cnn_stride_variable_delta_free_running_v9_seed7 \
  --out-dir formal_NN_training/results/623_stride_lstm_vs_cnn_free_running

python3 formal_NN_training/experiments/compare_623_split_architectures.py \
  --policy spp \
  --lstm-run-dir formal_NN_training/experiments/623_offline_lstm_spp/runs/623_offline_lstm_spp_variable_delta_free_running_v11_seed7 \
  --cnn-run-dir formal_NN_training/experiments/623_offline_cnn_spp/runs/623_offline_cnn_spp_variable_delta_free_running_v11_seed7 \
  --out-dir formal_NN_training/results/623_spp_lstm_vs_cnn_free_running
```

These checks require byte-identical normalized input manifests, identical
runtime encoder SHA256 values, identical offline normal lists, and identical
normal replay simulation results before comparing LSTM and CNN.
