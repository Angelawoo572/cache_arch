# cache_arch

This repository contains two matched-input neural-prefetcher studies:

- `formal_NN_training/experiments/602_offline_lstm_{stride,streamer,ampm,spp}/`
  is the completed four-prefetcher 602 study.
- `formal_NN_training/experiments/623_offline_lstm_{stride,spp}/` is the
  active 623 redesign.

Matched input means that a neural policy receives only the source-visible
callback fields available to its corresponding normal prefetcher.  Normal
actions, candidates, request budgets, and private normal-prefetcher state are
not neural runtime inputs.  This fairness boundary does **not** require the NN
to copy the normal prefetcher's internal algorithm, output templates, page
rules, thresholds, or degree.

The active 623 v21 models therefore remain independent direct-action learners.
Their stable, torch-free architecture and run metadata live in each track's
`python/model_contract.py`; `data/stream_contract.json` records the external
input and causal-use contract.  Their rank-conditioned decoder learns a direct
`STOP`/`EMIT(delta)` action sequence without a separate rounded-count head and
without teacher/predicted-action feedback.  Colab trains and performs offline
inference, and Linux/ChampSim replays the NN and matched offline-normal lists
through the same keyed replayer.

Run data are intentionally ignored by Git.  Shared helpers under
`formal_NN_training/common/` split gzip archives into verified parts no larger
than 90 MiB for Mac/Google Drive/Colab transfer; track-specific scripts remain
inside their experiment directories.

See `formal_NN_training/README.md` and the per-experiment READMEs for the 602
reference design, the audited 623 failure history, current architecture, and
the limits of each comparison.

Historical split-workflow helpers are retained only under
`formal_NN_training/legacy/scripts/direct_action_split_workflow/`; active
experiments do not import them.
