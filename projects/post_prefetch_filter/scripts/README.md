# Scripts for post-prefetch filter

This directory is intentionally separate from the old top-level `scripts/` directory.

Old GRU/replay scripts stay where they are for compatibility. New scripts for the post-prefetch filter should go here first, then can be promoted into top-level `scripts/` only after the flow is stable.

Planned scripts:

```text
01_probe_champsim_prefetchers.sh     # list available prefetchers in local ChampSim
02_run_baselines.sh                  # run no-prefetch and baseline prefetcher
03_dump_shadow_candidates.sh         # log baseline prefetch candidates and outcomes
04_train_filter.py                   # train logistic/perceptron utility filter
05_replay_filter.sh                  # evaluate baseline+filter vs baseline alone
06_summarize_metrics.py              # IPC/MPKI/accuracy/coverage/timeliness summary
```
