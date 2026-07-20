# 623 SPP — compact independent LSTM

This track compares normal SPP with a standalone LSTM on
`623.xalancbmk_s-700B`.

## Fair input contract

The audited source-visible input is the complete chronological callback stream:
`DEMAND(addr)` and `CACHE_FILL(evicted_addr)`. PC is replay transport only.
Cache hit/type and SPP's private ST/PT/GHR/FILTER contents are excluded.
Training and inference use the same lossless callback encoder, and all field
lists and encoder hashes must match. Captured SPP requests/fill choices are
labels and the fill-preserving offline comparator only.

## Independent NN design

Revision `compact_mass_hurdle_mixture_fill_v13` corrects the two v12 argmax
collapses: almost every callback issued and every action selected LLC. One
compact single-layer LSTM still processes every demand and fill callback in
time order. An unweighted Bernoulli learns zero versus positive callbacks and
a conditional Poisson learns the unbounded positive excess count. A causal
probability-mass scheduler preserves both the learned trigger rate and SPP's
positive-count bursts instead of smearing one average count across callbacks.
The autoregressive signed-delta mixture remains independent, while its feedback
uses the complete learned fill distribution. A second probability-mass decoder
converts that distribution to L2/LLC choices without discarding rare learned L2
mass. No selected probability threshold, degree cap, candidate list, fixed
page-offset vocabulary, same-page rule, SPP private state, or future row is
used. Eviction feedback is retained only because it is an actual source-visible
SPP callback; it is not a handcrafted eviction rule or prediction target.

The measured capacity sweep is h8/h16/h32/h64/h128 with
2,856/6,592/16,752/47,824/152,976 parameters.

Input revision: `spp_source_input_variable_delta_fill_feedback_free_running_v11`  
Model revision: `compact_mass_hurdle_mixture_fill_v13`  
Default run: `623_offline_lstm_spp_mass_hurdle_fill_v13_seed7`

Run `linux/launch_server.sh collect`, train with the A100 notebook, return the
output archive, and run `linux/launch_server.sh replay`. Previous v11/v12
outputs remain separate and are not overwritten.
