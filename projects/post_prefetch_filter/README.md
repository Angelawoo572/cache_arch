# Post-Prefetch Candidate Utility Filter — PAUSED WIP

## Status

This folder is a paused research snapshot, not the active training/replay pipeline.

It is kept because it contains useful pieces for a future clean restart:

```text
scripts/04_patch_spp_candidate_logger.sh   SPP candidate/event logger patch
scripts/05_events_to_candidate_table.py    event-log -> candidate-table converter
results/colab_feature_sweeps/              small offline summaries for 602/605
experiment_plan.md                         earlier plan and rationale
notebooks/rl_filter_story_starter.ipynb    paused Colab workflow snapshot
```

Do not keep adding new experiments here. If the project restarts, create a new clean directory such as:

```text
projects/behavior_utility_filter_v2/
```

or

```text
projects/cache_rl_replay_v2/
```

## Core idea preserved

The goal was not to replace SPP with a neural next-address predictor. The idea was:

```text
baseline prefetcher proposes a candidate
        ↓
small hardware-friendly filter estimates utility
        ↓
admit candidate or suppress candidate
```

The action space was intentionally tiny:

```text
0 = suppress
1 = admit
```

This is still a useful research direction because it avoids full-address prediction and instead judges whether an existing speculative action is worth spending cache/MSHR/bandwidth resources on.

## What was completed

The project reached the offline candidate-level analysis stage:

```text
ChampSim spp_dev candidate logging
  -> candidate table conversion
  -> offline/Colab feature sweep
  -> candidate-level admit/suppress metrics
```

The main candidate scope was:

```text
CANDIDATE_SCOPE = spp_l2_issue
condition       = spp_fill_l2 == 1 or spp_confidence >= 90
```

Useful metrics from the offline analysis were:

```text
issued_ratio        admitted candidates / candidate rows
accuracy            useful admitted / admitted
useful_kept_ratio   useful admitted / useful available
bad_suppressed      useless candidates suppressed
estimated_reward    offline utility proxy
```

## Main observations

### 602.gcc_s-734B

Observed behavior:

```text
SPP candidate useful rate was high.
MSHR pressure was very low.
The learned filter mostly admitted candidates.
```

Interpretation:

```text
602.gcc = TRUST_SPP / DO-NO-HARM case
```

This workload showed that the controller needs a mode that recognizes when the baseline SPP prefetcher is already good and should not be disturbed.

### 605.mcf_s-994B

Observed behavior:

```text
SPP candidate stream was noisy.
Simple candidate identity features could suppress many bad candidates while keeping most useful ones.
```

Interpretation:

```text
605.mcf = FILTER_BAD_GENERATOR case
```

This workload showed that SPP should not always be trusted blindly; a post-generator utility filter can be meaningful when the candidate stream is low-quality.

## What was not completed

This project did not finish the final online replay step.

Do not claim:

```text
final IPC improvement
final L1/L2/LLC hit-rate improvement
final miss-rate improvement
final hardware latency improvement
```

Those require an online ChampSim replay where the learned policy is connected inside `spp_dev` before `prefetch_line(...)`.

The intended final evaluation would compare original SPP against SPP plus the learned filter using:

```text
IPC / speedup
L1D, L2C, LLC hit rate
L1D, L2C, LLC miss rate
L1D, L2C, LLC MPKI
prefetch issued / useful / useless
prefetch accuracy
prefetch coverage
MSHR/PQ pressure
timeliness / late / evicted-unused if logged
```

## Why this folder is paused

The current folder accumulated too many intermediate states:

```text
candidate logger patches
Colab training attempts
hit/miss feature attempts
policy-table export attempts
partial replay plans
```

For a cleaner restart, keep the lessons and reusable scripts, but start a new project folder with a simpler contract:

```text
1. exact online feature schema
2. exact training/export format
3. exact ChampSim replay patch
4. final metric table from the beginning
```

## Recommended future restart

A new project should begin with:

```text
projects/behavior_utility_filter_v2/
  README.md
  scripts/
    01_patch_candidate_logger.sh
    02_events_to_table.py
    03_train_policy.py
    04_patch_champsim_policy_replay.sh
    05_run_replay_sweep.sh
  notebooks/
    analysis.ipynb
  results/
    summaries/
```

The first milestone should not be a bigger model. It should be:

```text
one workload
one exact feature schema
one exported tiny policy table
one ChampSim replay comparison against original SPP
```
