# 623 threshold-free causal CNN versus Stride

This directory runs only the causal CNN student against the offline Stride
comparator. Both receive byte-identical chronological PC and cache-line-address
information. The CNN input contains no Stride candidate, confidence, tracker
state, request outcome, or future row.

The CNN has one 1x1 input projection and only four stride-1 causal residual
convolution blocks. Their local kernel width is 7 and dilations are 1, 6, 36,
and 216, yielding a contiguous 1,555-event receptive field. Chunked execution
recomputes the exact previous 1,554 events, so chunk boundaries do not reset
visible history. Kernel width 7 is not a seven-event input limit.

The network learns request count and all 64 target ranks directly. Inference is
count argmax plus top-count ranking, with no probability threshold, normal
request-rate budget, normal degree cap, future-use cutoff, handcrafted semantic
feature, or copied normal-policy constant. Parameter groups c10/c16/c25 are
matched to the separate LSTM h8/h16/h32 groups.

Revision: `stride_threshold_free_split_v7`.
Default run: `623_offline_cnn_stride_threshold_free_v7_seed7`.
