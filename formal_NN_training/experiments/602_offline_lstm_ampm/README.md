# 602 threshold-free LSTM versus AMPM

AMPM and the LSTM receive the same effective external input: cache-line
address. The LSTM receives a lossless binary encoding, carries its own state,
and learns request count plus target ranking. Mirrored AMPM requests are
supervised targets and the normal replay only. There is no neural probability
threshold, request-rate budget, future-use window, AMPM degree cap, AMPM
bitmap/table input, handcrafted semantic feature, manual loss weight, or
training regularizer. Normal constants are confined to the comparator and
supervised-label generator. The guard stream initializes the two policies
independently.

Revision: `threshold_free_count_rank_v5`.
Default run: `602_offline_lstm_ampm_threshold_free_v5_seed7`.
