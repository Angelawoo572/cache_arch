# 623 SPP: independent LSTM/CNN target+fill comparison

The audited SPP decision body effectively reads address only. Both neural
families therefore receive only the causal address stream and independently
predict 64 same-page offsets times L2/LLC fill. Captured SPP actions are used
only for the fill-preserving normal replay and calibration request budget; they
are not model inputs or labels. Neural labels come from future demand reuse in
the training stream, with lead distance selecting L2 versus LLC.

The CNN has exactly one causal kernel-3, stride-1, dilation-1 filter. The
parameter-matched pairs remain h4/c5, h8/c10, and h16/c24.

Run ID: `623_offline_lstm_cnn_spp_direct_v4_seed7`.
