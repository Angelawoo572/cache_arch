# 602 source-input-fair PC-keyed LSTM versus Stride

This experiment compares offline Stride with a standalone neural prefetcher
under one external-input contract: both receive only the current PC and
cache-line address.  Normal Stride actions are supervised targets and the
normal replay baseline; they are never neural inputs, gates, candidates,
budgets, or private state.

## Why the compact v9 model is different

The previous shared Poisson request-count head failed on sparse Stride labels.
The held-out teacher averaged only 0.308 actions per callback, so Poisson-mode
decoding returned zero for every callback at every tested hidden size.

The compact v9 student fixes that mismatch inside this experiment directory:

1. **Dynamic PC-keyed recurrence.** Each observed PC selects its own learned
   LSTM state.  The map has no fixed 64-entry capacity and uses no normal
   tracker contents.
2. **Learned hurdle count.** A two-class neural head chooses zero versus
   positive by argmax.  A learned positive log-count then produces any positive
   integer.  There is no probability threshold or degree cap.
3. **Data-derived sparse-class balance.** Zero/positive class weights come
   directly from the observed training-label frequencies, giving both classes
   equal aggregate loss mass.  There is no tuned coefficient; this prevents
   the 84.3% zero rows from making an all-zero gate the easy solution.
4. **One compact shared encoder.** A lossless 128-bit PC/address projection and
   one single-layer LSTM supply the gate, count, and action heads.  Stride does
   not need two separate recurrent encoders.
5. **Free-running decoder.** Teacher deltas contribute loss but are never fed
   back to the one-cell GRU decoder; training and inference use the model's own
   previous signed-log delta.  The decoder is deterministic rather than a
   multi-component density model.
6. **Sweep-level collapse check.** Individual tiny models may legitimately
   fail, but the notebook and server reject an output archive if every capacity
   produces an empty neural replay list.

The NN has no page-offset output table, same-page rule, normal request-rate
budget, future-use window, handcrafted semantic feature, manual loss weight,
training regularizer, probability threshold, normal degree, or fixed PC-state
capacity.

External input/action-space contract revision:
`source_input_variable_delta_free_running_v7`.

Model revision: `compact_shared_pc_hurdle_delta_v9`.

For feature width `F` and hidden width `H`, the exact trainable-parameter
formula is `11H^2 + (F+22)H + 4`.  The lossless PC/address input fixes `F=128`,
so the capacity sweep records 1,908 / 5,220 / 16,068 / 54,660 / 199,428
parameters for H = 8 / 16 / 32 / 64 / 128, respectively.  This keeps h8 below
the 2,821-parameter h8 shared model used by the existing 602 Streamer/AMPM
tracks.  `run_metadata.json` and `model.pt` also store the measured count so
every cache result remains size-auditable.  Because the exact-PC state map is
dynamic, metadata also records unique-PC counts and peak persistent recurrent
state bytes (`2H` float32 values per observed PC); parameter count is not
misrepresented as total deployment storage.

The normal Stride implementation's 64 trackers and degree 2 remain only in the
normal comparator.  They are neither neural inputs nor neural limits.

Default run: `602_offline_lstm_stride_compact_hurdle_v9_seed7`.
