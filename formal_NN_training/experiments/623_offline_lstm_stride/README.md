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

Revision `compact_pc_keyed_mass_hurdle_mixture_v13` corrects the v10 inverse-
frequency gate, which over-issued on this sparse trace. An exact-PC dynamic
state map routes events through one compact single-layer LSTM. An unweighted
Bernoulli hurdle learns zero versus positive callbacks and a conditional
Poisson learns the unbounded positive excess count. Causal per-PC probability-
mass scheduling preserves the learned sparse trigger rate without `p>c`, while
the excess residual preserves Stride's near-two-action bursts. A small
autoregressive three-component mixture learns direct signed cache-line deltas
without averaging incompatible modes. There is no selected probability
threshold, fixed tracker capacity, degree cap, candidate list, page-offset
vocabulary, or same-page rule.

The measured capacity sweep is h8/h16/h32/h64/h128 with
1,971/5,339/16,299/55,115/200,331 parameters.

Input revision: `stride_source_input_variable_delta_free_running_v9`  
Model revision: `compact_pc_keyed_mass_hurdle_mixture_v13`  
Default run: `623_offline_lstm_stride_mass_hurdle_v13_seed7`

Run `linux/launch_server.sh collect`, train with the A100 notebook, return the
output archive, and run `linux/launch_server.sh replay`. Previous v9/v10
outputs remain separate and are not overwritten.
