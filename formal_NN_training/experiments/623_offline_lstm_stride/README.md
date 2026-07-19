# 623 Stride — compact independent LSTM

This track compares normal Stride with a standalone LSTM on
`623.xalancbmk_s-700B`.

## Fair input contract

Both methods receive the source-visible current `pc` and aligned `addr` only.
Training and inference call the same lossless encoder, and the server/analyzer
fail closed unless all three encoder hashes and field lists agree. Captured
Stride requests are supervised labels and the offline normal comparator; they
are never neural inputs. The NN receives no Stride tracker state, candidates,
degree, request-rate budget, or future rows.

## Independent NN design

Revision `compact_pc_keyed_hurdle_delta_v10` replaces the old shared Poisson
count head. An exact-PC dynamic state map routes events through one compact
single-layer LSTM. A learned two-class gate selects zero versus positive by
categorical argmax; class weights are derived only from the training-label
frequencies. For positive events the model learns an unbounded positive count
and generates direct signed cache-line deltas autoregressively using its own
previous prediction. There is no probability threshold, fixed tracker
capacity, degree cap, candidate list, page-offset vocabulary, or same-page
rule.

The measured capacity sweep is h8/h16/h32/h64/h128 with
1,908/5,220/16,068/54,660/199,428 parameters.

Input revision: `stride_source_input_variable_delta_free_running_v9`  
Model revision: `compact_pc_keyed_hurdle_delta_v10`  
Default run: `623_offline_lstm_stride_compact_hurdle_v10_seed7`

Run `linux/launch_server.sh collect`, train with the A100 notebook, return the
output archive, and run `linux/launch_server.sh replay`. Previous v9 outputs
remain separate and are not overwritten.
