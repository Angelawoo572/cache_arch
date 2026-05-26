# Scripts for post-prefetch filter

This directory is intentionally separate from the old top-level `projects/legacy_gru_prefetch/scripts/` directory.

Old GRU/replay scripts stay where they are for compatibility. New scripts for the post-prefetch filter should go here first, then can be promoted into top-level `projects/legacy_gru_prefetch/scripts/` only after the flow is stable.

## Current stable flow

```text
01_probe_champsim_prefetchers.sh
  List available prefetchers in local ChampSim. Confirm `spp_dev` exists.

02_run_spp_baseline_stats.sh
  Run a real ChampSim binary and parse IPC/cache/prefetch aggregate stats.

03_patch_spp_final_stats.sh
  Patch spp_dev::prefetcher_final_stats() to print SPP_FINAL counters.

04_patch_spp_candidate_logger.sh
  Patch spp_dev to emit candidate-level CAND/USE events for ML/RL data.

05_events_to_candidate_table.py
  Convert event stream into one-row-per-candidate table.
  Default scope is `spp_l2_issue`, not all lookahead attempts.
```

## Important candidate-scope rule

The raw CAND event stream includes many SPP lookahead/candidate attempts. Training on all of them creates an extreme class imbalance and the trivial learned policy is to suppress almost everything.

For the first controlled experiment, generate the Colab table with:

```bash
python3 projects/post_prefetch_filter/scripts/05_events_to_candidate_table.py \
  --trace 602.gcc_s-734B \
  --events projects/post_prefetch_filter/data/generated/spp_candidate_events_602_gcc_25m.csv \
  --out projects/post_prefetch_filter/data/generated/spp_candidate_log.csv.xz \
  --scope spp_l2_issue \
  --min-confidence 90
```

This scope asks:

```text
Among candidates SPP itself would issue to L2,
can a tiny filter suppress bad/resource-risky candidates?
```

Only after this controlled scope is meaningful should we expand to all lookahead candidates, LLC-only candidates, fill-level control, or degree control.
