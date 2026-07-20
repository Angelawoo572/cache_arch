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

Revision `compact_event_sampled_mixture_fill_v14` removes both v12's independent
argmax collapse and v13's cross-callback probability-credit phase. One compact
single-layer LSTM processes every demand and fill callback in time order. An
unweighted Bernoulli and conditional Poisson describe the complete
zero/positive/unbounded-count distribution. Every demand samples its own
learned count distribution with a reproducible run seed. The autoregressive
signed-delta head samples a learned mixture component, and the fill head samples
the learned L2/LLC categorical distribution. Recurrent feedback uses the full
delta mixture expectation and full fill probabilities in both training and
inference, never a teacher action or sampled output. No selected probability
threshold, degree cap, candidate list, fixed page-offset vocabulary, same-page
rule, SPP private state, or future row is used. Eviction feedback is retained
only because it is an actual source-visible SPP callback; it is not a
handcrafted eviction rule or prediction target.

The measured capacity sweep is h8/h16/h32/h64/h128 with
2,856/6,592/16,752/47,824/152,976 parameters.

Input revision: `spp_source_input_variable_delta_fill_feedback_free_running_v11`  
Model revision: `compact_event_sampled_mixture_fill_v14`  
Default run: `623_offline_lstm_spp_event_sampled_fill_v14_seed7`

Run `linux/launch_server.sh collect`, train with the A100 notebook, return the
output archive, and run `linux/launch_server.sh replay`. Previous v11--v13
outputs remain separate and are not overwritten.
