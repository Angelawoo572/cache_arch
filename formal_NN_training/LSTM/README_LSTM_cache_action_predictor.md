# Outcome-aware SPP-assisted LSTM Cache Action Predictor

This note is now a **legacy/reference note** for the older LSTM cache-action experiments. The repo has moved to the Pythia-based `external/ChampSim`, so the old ChampSim `spp_dev` candidate-logging / `list_replayer` scripts were removed from `formal_NN_training/scripts/`.

Current active scripts are:

```text
formal_NN_training/scripts/17_parse_prefetch_behavior_audit.py
formal_NN_training/scripts/18_run_prefetch_behavior_audit.sh
```

Current active first step:

```text
Pythia no_pref / SPP / IPCP behavior audit
  -> logs in formal_NN_training/results/LSTM/behavior_audit/logs/
  -> summary_nodup.csv
  -> decide residual-booster labels before writing the new LSTM notebook
```

Old LSTM notebooks are still kept here for reference:

```text
formal_NN_training/LSTM/notebooks/LSTM_cache_action_predictor.ipynb
formal_NN_training/LSTM/notebooks/LSTM_cache_action_predictor_SPP_LSTM_direct_hybrid.ipynb
formal_NN_training/LSTM/LSTM_cache_action_pipeline_story.md
```

Research framing kept from the old version:

```text
SPP was used as candidate + context + supervision.
LSTM learned whether a candidate cache action was useful, non-duplicate, and worth issuing.
```

New framing going forward:

```text
normal prefetcher = SPP first, IPCP later
NN = LSTM first, tiny Transformer later
label = residual demand misses not covered in time by the normal prefetcher
metrics = accuracy + timeliness + coverage + IPC/speedup
```

Old result files remain under:

```text
formal_NN_training/results/LSTM/draft/
```
