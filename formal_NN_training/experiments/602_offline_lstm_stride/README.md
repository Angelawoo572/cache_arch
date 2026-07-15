# 602 Stride matched-input direct-action LSTM

The audited Stride source effectively reads `pc` and cache-line `address`.
The normal mirror and LSTM receive that same stream, but the LSTM owns its
state and predicts up to two of 64 same-page L2 targets. Stride candidates,
tracker entries, and last-stride state are never model inputs. Labels are
future demand reuse from the training stream; evaluation is causal.

Use `602_offline_lstm_stride_A100.ipynb` with the unchanged input archive.
Revision: `direct_action_independent_v3`.
