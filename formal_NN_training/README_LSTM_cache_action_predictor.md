# LSTM Cache Action Predictor

This README records the current `formal_NN_training/LSTM_cache_action_predictor.ipynb` setup and the current observed data/evaluation results for `602.gcc_s-734B`.

The research intent is **not** to make the LSTM a simple SPP yes/no filter. In this workflow, SPP-derived logs are used as training/evaluation data and context. The current notebook implementation, however, is a next-demand-line / next-delta LSTM formulation, not yet an outcome-useful cache-action formulation.

---

## Main files

```text
formal_NN_training/LSTM_cache_action_predictor.ipynb
formal_NN_training/scripts/02_actions_to_prefetch_list.py
formal_NN_training/scripts/04_eval_lstm_accuracy.py
formal_NN_training/scripts/05_eval_current_label_lstm_vs_spp.py
```

---

## Current data files used in the 602.gcc_s-734B run

### Event table

```text
formal_NN_training/data/generated/lstm_events_602.gcc_s-734B.csv
```

Observed evaluator counts:

```text
events_total       = 9,822,930
events_useful      = 271,727
events_useful_rate = 0.02766252
events_duplicate   = 9,574,847
```

For the rows that joined with the current exported LSTM action table:

```text
joined_action_rows = 1,282,137
useful_joined      = 65,648
duplicate_joined   = 1,219,154
good_prefetch rows = 47,707
```

Here, the diagnostic `good_prefetch` definition used for analysis was:

```text
outcome_useful == 1 AND outcome_duplicate == 0 AND true delta != 0
```

This diagnostic label is **not** currently used by the notebook as a training label.

### Exported LSTM action table

Original exported file:

```text
formal_NN_training/artifacts/full_lstm_cache_actions.csv
```

Observed file properties on the cluster:

```text
file size       = 302 MB
line count      = 1,282,138 including header
parsed rows     = 1,282,137
NUL byte count  = 113,573,888
```

A cleaned copy was created by stripping NUL bytes:

```bash
tr -d '\000' < formal_NN_training/artifacts/full_lstm_cache_actions.csv \
  > formal_NN_training/artifacts/full_lstm_cache_actions.clean.csv
```

Cleaned file properties:

```text
formal_NN_training/artifacts/full_lstm_cache_actions.clean.csv
file size       = 193 MB
NUL byte count  = 0
newline count   = 1,282,138
```

Current action-table columns:

```text
trace,event_id,cycle_num,pc_int,addr_int,line_addr,
pred_delta_id,pred_delta,pred_delta_conf,
pred_future_hit_prob,pred_bypass_prob,pred_timing_bin,
nn_action,prefetch_line_addr,prefetch_addr
```

Current parsed `nn_action` distribution:

```text
INSERT_NORMAL_NO_PREFETCH       1,229,030
BYPASS_OR_LOW_PRIORITY_INSERT      52,709
PREFETCH_DELTA                        397
invalid/malformed action value           1
```

---

## What the current notebook labels actually are

The current notebook derives cache-line features from `addr_int`:

```python
df["line_addr"] = df["addr_int"] // CONFIG["cache_line_bytes"]
df["prev_line_addr"] = df.groupby("trace")["line_addr"].shift(1)
df["delta"] = df["line_addr"] - df["prev_line_addr"]
```

The current delta target is next-demand-line delta:

```python
H = CONFIG["prediction_horizon"]
df["future_line_addr"] = df.groupby("trace")["line_addr"].shift(-H)
df["future_delta"] = df["future_line_addr"] - df["line_addr"]
```

The current hit target is the next event's hit/miss value:

```python
df["future_hit"] = df.groupby("trace")["hit_int"].shift(-H)
```

The current bypass target is based on reuse distance of the current demand line:

```python
df["next_pos_same_line"] = df.groupby(["trace", "line_addr"])["pos_in_trace"].shift(-1)
df["reuse_distance_events"] = df["next_pos_same_line"] - df["pos_in_trace"]
df["bypass_label"] = df["reuse_distance_events"] > CONFIG["bypass_reuse_distance_events"]
```

Current config value:

```text
bypass_reuse_distance_events = 2048
prediction_horizon = 1
cache_line_bytes = 64
```

The current notebook does **not** define or train on a `good_prefetch_label` based on `outcome_useful` and `outcome_duplicate`.

---

## Current notebook export action rule

The current action export rule is:

```python
if pred_bypass_prob >= CONFIG["prediction_threshold_bypass"]:
    nn_action = "BYPASS_OR_LOW_PRIORITY_INSERT"
elif pred_delta_conf >= CONFIG["prediction_threshold_prefetch"] and pred_delta != 0:
    nn_action = "PREFETCH_DELTA"
else:
    nn_action = "INSERT_NORMAL_NO_PREFETCH"
```

The exported prefetch line is currently computed as:

```python
prefetch_line_addr = line_addr + pred_delta
```

In the current notebook, `pred_delta` is the predicted next-demand-line delta, not an `outcome_useful` candidate-action label.

---

## Current outcome-useful/action diagnostic result

Using:

```text
formal_NN_training/scripts/04_eval_lstm_accuracy.py
```

with:

```text
--events  formal_NN_training/data/generated/lstm_events_602.gcc_s-734B.csv
--actions formal_NN_training/artifacts/full_lstm_cache_actions.clean.csv
--policy  action
```

Observed result:

```text
lstm_prefetch_emitted                = 397
lstm_prefetch_useful                 = 0
lstm_prefetch_precision_accuracy     = 0.00000000
lstm_prefetch_recall_useful_coverage = 0.00000000
lstm_duplicate                       = 397
lstm_duplicate_rate                  = 1.00000000
```

Observed action distribution on joined useful rows:

```text
Actions on useful rows:
INSERT_NORMAL_NO_PREFETCH       57,703  (0.8789757495)
BYPASS_OR_LOW_PRIORITY_INSERT    7,945  (0.1210242505)
```

Observed action distribution on joined duplicate rows:

```text
Actions on duplicate rows:
INSERT_NORMAL_NO_PREFETCH       1,166,047  (0.9564394654)
BYPASS_OR_LOW_PRIORITY_INSERT      52,709  (0.0432340787)
PREFETCH_DELTA                        397  (0.0003256356)
invalid/malformed action value          1
```

This records the current behavior of the exported `nn_action` field. It does not by itself prove that an LSTM sequence model cannot be useful; it shows that the current exported action rule is not aligned with the `outcome_useful` / `outcome_duplicate` diagnostic.

---

## Current-label LSTM vs SPP address-level evaluation

Using:

```text
formal_NN_training/scripts/05_eval_current_label_lstm_vs_spp.py
```

with:

```text
--events  formal_NN_training/data/generated/lstm_events_602.gcc_s-734B.csv
--actions formal_NN_training/artifacts/full_lstm_cache_actions.clean.csv
--window 32
```

Observed output:

```text
total_usable_rows                        1,282,136
true_future_delta_zero                   1,219,063
true_future_delta_zero_rate              0.95080631
lstm_pred_delta_zero                     1,234,661
lstm_pred_delta_zero_rate                0.96297195
naive_always_zero_acc                    0.95080631
lstm_delta_top1_acc                      0.98374587
spp_delta_top1_acc                       0.04247209
lstm_next_line_addr_acc                  0.98374587
spp_next_line_addr_acc                   0.00216826
lstm_future_window32_hit                 0.98642656
spp_future_window32_hit                  0.10854153
both_correct                             2,613
lstm_only                                1,258,683
spp_only                                 167
both_wrong                               20,673
nonzero_total                            63,073
nonzero_rate                             0.04919369
lstm_nonzero_delta_acc                   0.67907028
spp_nonzero_delta_acc                    0.83041872
lstm_nonzero_addr_acc                    0.67907028
spp_nonzero_addr_acc                     0.01109825
lstm_nonzero_pred_rate                   0.74321818
spp_nonzero_pred_rate                    0.96705405
lstm_nonzero_future_window32_hit         0.72441457
spp_nonzero_future_window32_hit          0.02065860
sanity_checked_rows                      1,282,136
sanity_event_line_mismatch               0
sanity_pf_line_minus_line_ne_delta       1,156,851
```

The sanity result shows that the event-table `delta` field does not equal `pf_line - current_line` for most joined rows:

```text
sanity_pf_line_minus_line_ne_delta = 1,156,851 / 1,282,136
```

Therefore, exact SPP delta metrics that use the event-table `delta` field should not be mixed with address metrics unless the field meaning is first verified. Address/window metrics use the exported/present `pf_addr` or prefetch line address directly.

---

## Current factual interpretation

The current notebook can be evaluated as a next-demand-line predictor. Under that current formulation, the exported LSTM predictions show:

```text
LSTM next-line address accuracy      = 0.98374587
LSTM future-window-32 hit            = 0.98642656
LSTM nonzero next-line address acc   = 0.67907028
LSTM nonzero future-window-32 hit    = 0.72441457
```

The same evaluator reports SPP address/window metrics on the same joined rows:

```text
SPP next-line address accuracy       = 0.00216826
SPP future-window-32 hit             = 0.10854153
SPP nonzero next-line address acc    = 0.01109825
SPP nonzero future-window-32 hit     = 0.02065860
```

The current notebook should not be described as having improved `outcome_useful` prefetch-action precision, because the current `nn_action == PREFETCH_DELTA` rows were all duplicate in the joined diagnostic above.

---

## Important distinction for the next step

Current implemented objective:

```text
predict next demand-line delta / next demand-line address
```

Outcome-aware cache-action objective not yet implemented in the notebook:

```text
learn whether a candidate/action is useful, duplicate, bypass-worthy, or timing-sensitive
using outcome_useful / outcome_duplicate labels
```

A future outcome-aware version would need explicit labels derived from the event table, such as:

```text
good_prefetch = outcome_useful == 1 AND outcome_duplicate == 0
bypass        = outcome_duplicate == 1 OR outcome_useful == 0
```

Those labels are recorded here as the next intended direction, not as the current notebook behavior.
