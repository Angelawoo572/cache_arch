# Deprecated combined 623 experiment

Do not start, resume, train, replay, or cite results from this directory.
Its launcher and server entry point intentionally exit with an error.

Use the two independent matched tracks instead:

- `../623_offline_lstm_cnn_stride`: offline stride versus stride-gated LSTM/CNN;
- `../623_offline_lstm_cnn_spp`: offline SPP versus SPP-gated LSTM/CNN.

The split is required because SPP has private Signature/Pattern/GHR/filter
state and chooses both `FILL_L2` and `FILL_LLC`, whereas stride uses a different
candidate generator and only `FILL_L2`. The new SPP track preserves the
captured fill action during replay, and both tracks use the v5 explicit-trigger
event schema. Old combined archives are incompatible and must not be reused.
