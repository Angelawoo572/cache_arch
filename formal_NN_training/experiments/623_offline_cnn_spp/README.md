# 623 threshold-free causal CNN versus source SPP

This directory runs only the causal CNN student against the fill-preserving
offline SPP comparator. Its input is the same chronological
`DEMAND(addr)`/`CACHE_FILL(evicted_addr)` stream used by source SPP; private SPP
tables, confidence, GHR/filter state, and SPP actions are not neural inputs.

The CNN has one 1x1 input projection and only four stride-1 causal residual
convolution blocks. Local kernel width 7 with dilations 1, 6, 36, and 216 gives
a contiguous 1,555-event receptive field. Exact 1,554-event overlap makes
chunking numerically equivalent to a sliding causal window and exposes no
future input.

Learned heads own request count, all 64 same-page target ranks, and L2/LLC fill
class. Inference uses argmax and top-count only—no probability threshold,
request budget, degree cap, fill cutoff, future-use window, or copied SPP
constant. CNN c8/c13/c22 are parameter-matched to separate LSTM h8/h16/h32.

Revision: `spp_threshold_free_fill_feedback_split_v9`.
Default run: `623_offline_cnn_spp_threshold_free_v9_seed7`.
