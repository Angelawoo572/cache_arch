# 602 source-input-fair PC-keyed LSTM versus Stride

This experiment compares offline Stride with a standalone neural prefetcher
under one external-input contract: both receive only the current PC and
cache-line address.  Normal Stride actions are supervised targets and the
normal replay baseline; they are never neural inputs, gates, candidates,
budgets, or private state.

## Why the v8 model is different

The previous shared Poisson request-count head failed on sparse Stride labels.
The held-out teacher averaged only 0.308 actions per callback, so Poisson-mode
decoding returned zero for every callback at every tested hidden size.

The v8 student fixes that mismatch inside this experiment directory:

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
4. **Independent action path.** A second PC-keyed LSTM generates
   autoregressive direct signed cache-line deltas.  Its continuous density loss
   cannot overwhelm the sparse emit decision because the recurrent parameters
   are disjoint.
5. **Free-running decoder.** Teacher deltas contribute loss but are never fed
   back to the decoder; training and inference use the model's own previous
   delta.
6. **Sweep-level collapse check.** Individual tiny models may legitimately
   fail, but the notebook and server reject an output archive if every capacity
   produces an empty neural replay list.

The NN has no page-offset output table, same-page rule, normal request-rate
budget, future-use window, handcrafted semantic feature, manual loss weight,
training regularizer, probability threshold, normal degree, or fixed PC-state
capacity.

External input/action-space contract revision:
`source_input_variable_delta_free_running_v7`.

Model revision: `pc_keyed_hurdle_direct_delta_v8`.

With the default four delta-mixture components and the 128-bit lossless input,
the exact trainable-parameter formula is `12H^2 + 1066H + 16`.  The sweep
therefore records 9,312 / 20,144 / 46,416 / 117,392 / 333,072 parameters for
H = 8 / 16 / 32 / 64 / 128, respectively.  `run_metadata.json` and `model.pt`
also store the measured count so every cache result remains size-auditable.

Default run: `602_offline_lstm_stride_pc_keyed_hurdle_v8_seed7`.
