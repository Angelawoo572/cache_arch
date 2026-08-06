# 623 Stride v23 — decoder-only hurdle-prior correction

This is a strict matched-input ablation for `623.xalancbmk_s-700B`.

- Run: `623_offline_lstm_stride_prior_corrected_hurdle_v23_seed7`
- Parent: `623_offline_lstm_stride_raw_hurdle_count_v22_seed7`
- Model: `pc_keyed_raw_hurdle_count_rank_delta_v22_reused_v23`
- Decoder: `train_weight_prior_corrected_hurdle_decode_v23`

The v22 evidence showed useful target learning but roughly 60% excess request
traffic. v22 trained its `ZERO/POSITIVE` hurdle with TRAIN-derived inverse-
frequency weights, then decoded the weighted logits directly. That changes the
natural class prior. v23 changes only the inference score:

```text
natural_logit[class] = weighted_logit[class] - log(TRAIN_class_weight[class])
```

This exactly removes the training reweighting. It is not a probability
threshold, request budget, tuned constant, or normal-Stride rule.

## Fair-input and identity contract

The runtime tensor is still only lossless `PC64 + aligned line58`. The LSTM is
routed by exact PC. Captured Stride actions remain labels and comparator rows;
they are never runtime features, candidates, feedback, degree hints, or private
state. The TRAIN-derived exact-delta vocabulary, positive log-count head,
rank-conditioned direct-delta heads, five capacities, splits, and chronology
are unchanged.

The A100 notebook requires both the reused v22 input archive and v22 output
archive. For each capacity it:

1. loads the v22 checkpoint and training history;
2. performs the original raw-logit decode;
3. requires that replay list to match the v22 NN list byte-for-byte;
4. performs the prior-corrected decode;
5. copies `model.pt` and `training_history.csv` byte-for-byte;
6. records every parent and output SHA-256 in `run_metadata.json`.

Any mismatch aborts before packaging. No v23 training or checkpoint selection
occurs. The active five tags are:

- `prior_corrected_hurdle_count_stride_lstm_h8`
- `prior_corrected_hurdle_count_stride_lstm_h16`
- `prior_corrected_hurdle_count_stride_lstm_h32`
- `prior_corrected_hurdle_count_stride_lstm_h64`
- `prior_corrected_hurdle_count_stride_lstm_h128`

## Validation

Torch-free checks:

```bash
python3 formal_NN_training/experiments/623_offline_lstm_stride/python/model_contract.py --self-test
python3 formal_NN_training/experiments/623_offline_lstm_stride/python/train_and_offline_infer.py --self-test
python3 formal_NN_training/experiments/validate_direct_action_contracts.py
```

Use `colab/623_offline_lstm_stride_A100.ipynb` for the two A100 decodes and
`linux/launch_server.sh replay` for ChampSim. A root `PASS` proves identity,
fair input, deterministic decode, and replay accounting; it does not claim an
IPC improvement.
