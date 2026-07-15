# 602 AMPM matched-input direct-action LSTM

The audited AMPM source effectively reads only cache-line `address`.
The normal mirror and LSTM receive the same guard-plus-evaluation stream, but
the LSTM owns its state and predicts up to four of 64 same-page L2 targets.
AMPM candidates, page bitmaps, page buffer, and LRU state are never model
inputs. Labels are future demand reuse from the training stream.

Use `602_offline_lstm_ampm_A100.ipynb` with the unchanged input archive.
Revision: `direct_action_independent_v3`.
