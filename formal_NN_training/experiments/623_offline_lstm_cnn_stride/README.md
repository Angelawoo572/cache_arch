# 623 Stride: independent LSTM/CNN direct-action comparison

This is not a candidate gate. The audited `stride.cc` decision body reads PC
and address, so both neural families receive only the same PC/address stream.
They independently predict up to two of 64 same-page L2 targets. Captured
Stride requests remain in the archive only to replay the normal comparator and
set the calibration request budget; they are never tensors or labels.

The CNN has exactly one causal `Conv1d`: kernel 3, stride 1, dilation 1, left
padding 2. Output `t` sees only `t-2,t-1,t`. Three parameter-matched pairs are
LSTM h4 / CNN c5, LSTM h8 / CNN c12, and LSTM h16 / CNN c29.

Run ID: `623_offline_lstm_cnn_stride_direct_v3_seed7`.
