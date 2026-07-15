# 623 threshold-free LSTM versus Stride

This directory runs only the LSTM student against the offline Stride
comparator. Both receive the same source-effective external information:
chronological PC and cache-line address. The lossless input encoding contains
no Stride candidate, confidence, tracker state, request outcome, or future row.

The LSTM carries hidden/cell state through the complete train stream, then the
guard stream, then evaluation. It learns a categorical request count over
0..64 and a ranking of all 64 same-page L2 targets. Inference is count argmax
plus top-count ranking: no probability threshold, normal request-rate budget,
normal degree cap, future-use cutoff, handcrafted semantic feature, or copied
normal-policy constant is used.

This track is separate from `623_offline_cnn_stride`. The two run-specific
archives contain byte-identical input files and share parameter groups
p0/p1/p2, so results are merged only after both independent analyses pass.

Revision: `stride_threshold_free_split_v7`.
Default run: `623_offline_lstm_stride_threshold_free_v7_seed7`.
