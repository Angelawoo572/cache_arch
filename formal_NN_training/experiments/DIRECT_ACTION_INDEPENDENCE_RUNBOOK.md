# Independent direct-action rerun

This revision fixes the boundary between the normal prefetcher and the neural
prefetcher.  A fair pair shares only the effective external inputs read by the
audited normal source.  The neural model owns its feature extractor, temporal
state, target generation, and confidence scores.

| Experiment | Shared external input | Independent neural output | Normal data excluded from tensors/labels |
|---|---|---|---|
| 602 Stride | PC + cache-line address | LSTM, 64 same-page offsets, degree 2 | tracker table, last stride, candidates |
| 602 Streamer | cache-line address | LSTM, 64 same-page offsets, degree 5 | page trackers, direction, candidates |
| 602 AMPM | cache-line address | LSTM, 64 same-page offsets, degree 4 | bitmap/LRU state, AMPM candidates |
| 623 Stride | PC + cache-line address | LSTM or one-layer causal CNN, 64 offsets | tracker state and captured candidates |
| 623 SPP | cache-line address | LSTM or one-layer causal CNN, 64 offsets x L2/LLC | ST/PT/GHR/filter state and SPP actions |

Normal request files remain necessary only to replay the normal baseline and
to set a training-split traffic budget.  They are never model inputs and never
supervision.  Neural labels are future-demand utility labels created strictly
inside the training stream.  Evaluation future rows are used only after
inference for an audit metric.

The CNN is intentionally small: exactly one causal `Conv1d`, kernel size 3,
stride 1, dilation 1, followed by a linear direct-action head.  Its receptive
field is the current callback and the previous two callbacks.  Runtime tests
reject future dependence, receptive fields longer than three, and chunked vs.
continuous inference disagreement.

Use `stage_direct_action_inputs.sh` to copy an existing run's collected input
bytes into the new run ID and regenerate only metadata/checksum containers.
Use `launch_direct_action_replays.sh` after all Colab output archives have been
extracted into their corresponding `colab_output` directories.

