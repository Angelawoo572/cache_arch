# 623 threshold-free LSTM versus source SPP

This directory runs only the LSTM student against the fill-preserving offline
SPP comparator. The audited source-effective input is the chronological union
of `DEMAND(addr)` and `CACHE_FILL(evicted_addr)` callbacks. Both methods receive
that same information; SPP private signature/pattern/GHR/filter state and SPP
actions are never neural inputs.

The LSTM carries hidden/cell state through complete train, guard, and evaluation
history. It learns a categorical count, all 64 same-page target ranks, and an
L2/LLC fill class. Inference uses argmax and top-count only—no probability
threshold, request budget, degree cap, fill cutoff, future-use window, or copied
SPP constant.

This track is separate from `623_offline_cnn_spp`. SPP is collected once here;
every strict input file is copied byte-for-byte into a separately named CNN
archive before its Colab run.

Revision: `spp_threshold_free_fill_feedback_split_v9`.
Default run: `623_offline_lstm_spp_threshold_free_v9_seed7`.
