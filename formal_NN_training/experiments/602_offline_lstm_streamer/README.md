# 602 threshold-free LSTM versus Streamer

Streamer and the LSTM receive the same effective external input: cache-line
address. The LSTM receives a lossless binary encoding, carries its own state,
and learns request count plus target ranking. Mirrored Streamer requests are
supervised targets and the normal replay only. There is no neural probability
threshold, request-rate budget, future-use window, Streamer degree cap,
handcrafted semantic feature, manual loss weight, or training regularizer.
Normal constants are confined to the comparator and supervised-label
generator; neural inference never reads them.

Revision: `threshold_free_count_rank_v5`.
Default run: `602_offline_lstm_streamer_threshold_free_v5_seed7`.
