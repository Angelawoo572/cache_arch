# 623 Stride — independent causal CNN

This track compares normal Stride with a standalone shallow causal CNN on
`623.xalancbmk_s-700B`.

Fair input is exact: both policies receive only the current `pc` and aligned
`addr`. Training and inference call the same lossless runtime encoder; its
source hash is recorded three times and validators require equality. Captured
Stride requests are labels and the offline normal comparator, never NN input.

The CNN has two stride-1 causal residual temporal filters, 17 taps each, with
dilations 1 and 17. Its moving receptive field is 289 chronological callbacks
with exact 288-event overlap and no future input. A learned Poisson count and
free-running autoregressive direct signed-delta decoder generate addresses independently of
Stride: no tracker, candidate list, threshold, degree cap, fixed page-offset
classes, or same-page rule.
Training and inference both feed the decoder its own previous prediction.

Revision: `stride_source_input_variable_delta_free_running_v9`  
Default run: `623_offline_cnn_stride_variable_delta_free_running_v9_seed7`

Use `linux/launch_server.sh collect` on the server, the A100 notebook for
training, and `linux/launch_server.sh replay` after returning the output.
