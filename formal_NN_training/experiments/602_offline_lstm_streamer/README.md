# 602 Streamer matched-input direct-action LSTM

The audited Streamer source effectively reads only cache-line `address`.
The normal mirror and LSTM receive that same stream, but the LSTM owns its
state and predicts up to five of 64 same-page L2 targets. Streamer candidates,
page trackers, and direction state are never model inputs. Labels are future
demand reuse from the training stream; evaluation is causal.

Use `602_offline_lstm_streamer_A100.ipynb` with the unchanged input archive.
Revision: `direct_action_independent_v3`.
