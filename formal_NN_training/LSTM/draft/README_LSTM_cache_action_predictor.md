# Legacy SPP-assisted cache-action predictor

This directory is historical reference material for the retired SPP/action-predictor
experiments. It is **not** part of the current standalone no-prefetch pipeline.

The notebooks and notes here reference scripts that were retired with the old
candidate-logging workflow. Do not repair those paths, do not run these notebooks
as current experiments, and do not treat their outputs as evidence for the current
keyed standalone replay design.

## Current pipeline boundary

The active pipeline is built from no-prefetch demand events, a standalone oracle,
frozen rich-list exports, and PC-line-occurrence keyed replay. Its active scripts
are documented in `formal_NN_training/scripts/README.md`.

## Historical material retained here

```text
formal_NN_training/LSTM/notebooks/LSTM_cache_action_predictor.ipynb
formal_NN_training/LSTM/notebooks/LSTM_cache_action_predictor_SPP_LSTM_direct_hybrid.ipynb
formal_NN_training/LSTM/LSTM_cache_action_pipeline_story.md
formal_NN_training/results/LSTM/draft/
```

The old framing was: SPP supplied candidates/context/supervision and an LSTM
predicted a cache action. That framing is archived; it must not be mixed with the
current standalone model's measured results.
