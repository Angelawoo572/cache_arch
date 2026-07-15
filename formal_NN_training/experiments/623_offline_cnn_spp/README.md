# 623 SPP — independent causal CNN

This track compares normal SPP with a standalone shallow causal CNN on
`623.xalancbmk_s-700B`.

The audited SPP source input is the chronological callback sequence
`DEMAND(addr)` plus `CACHE_FILL(evicted_addr)`. PC is replay transport only;
cache-hit/type and SPP private ST/PT/GHR/FILTER state are not NN inputs.
Training and inference call the same lossless callback encoder and validators
require identical encoder hashes. Captured SPP actions/fill levels are labels
and the fill-preserving normal comparator only.

The CNN has two stride-1 causal residual filters, 17 taps each, with dilations
1 and 17: a 289-callback moving receptive field, exact 288-event overlap, and
no future input. A learned Poisson count and free-running autoregressive direct signed-delta
and fill decoder has no SPP thresholds, degree, fixed page-offset classes,
same-page rule, candidate list, or private SPP state.
Teacher actions compute loss but are never recurrent decoder inputs.

Revision: `spp_source_input_variable_delta_fill_feedback_free_running_v11`  
Default run: `623_offline_cnn_spp_variable_delta_free_running_v11_seed7`

Use `linux/launch_server.sh collect` on the server, the A100 notebook for
training, and `linux/launch_server.sh replay` after returning the output.
