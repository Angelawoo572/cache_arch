# 623 Stride — independent LSTM

This track compares normal Stride with a standalone stateful LSTM on
`623.xalancbmk_s-700B`.

Fair input is exact: both policies receive only the current `pc` and aligned
`addr`. Training and inference call the same lossless runtime encoder; its
source hash is recorded three times and validators require equality. Captured
Stride requests are labels and the offline normal comparator, never NN input.

The NN is not a neural copy of Stride. One chronological stateful LSTM feeds a
learned Poisson request-count model and an autoregressive mixture over direct
signed cache-line deltas. Training and inference both feed back the model's
own previous action; teacher actions only compute loss. It has no tracker table, candidate list, threshold,
degree cap, fixed page-offset classes, or same-page rule. Hidden sizes are
ordinary model configurations, not normal-prefetcher constants.

Revision: `stride_source_input_variable_delta_free_running_v9`  
Default run: `623_offline_lstm_stride_variable_delta_free_running_v9_seed7`

Use `linux/launch_server.sh collect` on the server, the A100 notebook for
training, and `linux/launch_server.sh replay` after returning the output.
