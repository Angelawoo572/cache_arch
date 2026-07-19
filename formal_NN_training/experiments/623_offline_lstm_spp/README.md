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

Revision `compact_empirical_prior_hurdle_autoregressive_gmm_fill_v12` replaces
the shared Poisson head. One compact single-layer LSTM processes every demand
and fill callback in time order. Its unweighted empirical-prior hurdle gate
chooses zero versus positive by categorical argmax. Positive events use a
learned unbounded count and a free-running autoregressive signed-delta mixture;
fill placement is learned as L2 versus LLC. No probability threshold, degree
cap, candidate list, fixed page-offset vocabulary, same-page rule, SPP private
state, or future row is used. Eviction feedback is retained only because it is
an actual source-visible SPP callback; it is not a handcrafted eviction rule or
separate prediction target.

The measured capacity sweep is h8/h16/h32/h64/h128 with
2,865/6,609/16,785/47,889/153,105 parameters.

Input revision: `spp_source_input_variable_delta_fill_feedback_free_running_v11`  
Model revision: `compact_empirical_prior_hurdle_autoregressive_gmm_fill_v12`  
Default run: `623_offline_lstm_spp_empirical_prior_hurdle_v12_seed7`

Run `linux/launch_server.sh collect`, train with the A100 notebook, return the
output archive, and run `linux/launch_server.sh replay`. Previous v11 outputs
remain separate and are not overwritten.
