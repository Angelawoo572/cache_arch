# Same-input, threshold-free neural rerun

A fair pair shares the effective external inputs read by the audited normal
source.  The normal policy's emitted actions are supervised labels and its
offline baseline replay.  They are never neural inference inputs, gates,
request budgets, degree caps, or private state.

| Experiment | Shared external input | Learned neural decision |
|---|---|---|
| 602 Stride | PC + cache-line address | LSTM count argmax + ranking of 64 same-page L2 targets |
| 602 Streamer | cache-line address | LSTM count argmax + ranking of 64 same-page L2 targets |
| 602 AMPM | cache-line address | LSTM count argmax + ranking of 64 same-page L2 targets |
| 623 Stride | PC + cache-line address | LSTM or causal CNN count argmax + 64-target ranking |
| 623 SPP | chronological callback kind + demand address or cache-fill evicted address | LSTM or causal CNN count argmax + 64-target ranking + L2/LLC fill argmax |

The model uses lossless binary encodings rather than semantic hand features.
There is no probability threshold, future-use label window, fill-distance
cutoff, manual class/loss weight, weight decay, gradient clipping, normal
request-rate budget, or normal degree cap.  The 64 page offsets and SPP's two
fill levels define the complete hardware action interface, not tuned policy
limits.  Learning rate, epoch count, model width, kernel size, and dilation
schedule remain explicit training/architecture hyperparameters; they never
decide whether a particular prefetch is emitted.

The CNN is a causal residual TCN over the complete chronological stream.  It
has one 1x1 input projection and only four kernel-7, stride-1 temporal blocks
with dilations 1, 6, 36, and 216, followed by learned count/rank heads (and an
SPP fill head).  Kernel 7 is only each local filter, not the input-window
length; the stacked receptive field is a contiguous 1,555 callbacks.  Chunked
computation carries exactly 1,554 prior events into the next chunk, so a chunk
boundary does not reset declared TCN history.  Runtime tests reject future
dependence, temporal holes, and any input older than that receptive field. The
LSTM uses chronological stateful TBPTT and carries hidden/cell values across
chunks while detaching the graph at chunk boundaries.

This is behavior cloning under a strict same-input contract.  It tests whether
an independently structured NN can reproduce or improve the normal policy's
system effect; it is not end-to-end IPC reinforcement learning.

Use `stage_threshold_free_inputs.sh` to copy the existing collected `.csv.gz`
bytes for 602 Stride/Streamer/AMPM.  Collect 623 Stride once in
`623_offline_lstm_stride` and strict SPP once in `623_offline_lstm_spp`, then
run `stage_623_split_inputs.sh` for each policy.  The stager validates and
copies every normalized input byte into the corresponding standalone CNN
directory, producing independent LSTM/CNN archives with identical model-input
and teacher-label files.  New Colab training is required because the
architecture, loss, and output decoder have changed.  After extracting all
seven output archives, use
`launch_direct_action_replays.sh replay`; its per-run `nohup` launchers start
concurrently while the repository build lock serializes ChampSim builds.
