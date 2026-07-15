# 602 source-input-fair LSTM versus AMPM

AMPM and the LSTM receive the same effective external input: cache-line
address. The LSTM receives a lossless binary encoding, carries its own state,
and learns an unbounded count plus autoregressive direct cache-line deltas.
Decoder training and inference both feed back the model's own prediction;
teacher actions only compute loss.
Mirrored AMPM requests are
supervised targets and the normal replay only. There is no neural probability
threshold, request-rate budget, future-use window, fixed page-offset output
table, same-page rule, AMPM degree cap, AMPM
bitmap/table input, handcrafted semantic feature, manual loss weight, or
training regularizer. Normal constants are confined to the comparator and
supervised-label generator. The guard stream initializes the two policies
independently.

Revision: `source_input_variable_delta_free_running_v7`.
Default run: `602_offline_lstm_ampm_variable_delta_free_running_v7_seed7`.
