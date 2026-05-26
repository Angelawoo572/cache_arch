# Post-Prefetch Candidate Utility Filter

This folder contains the current cache/prefetch research direction. It is separate from the earlier GRU next-address and replay-list experiments.

## 1. Core idea

The goal is not to replace the hardware prefetcher with a neural network.

The goal is:

```text
baseline prefetcher proposes a candidate
        ↓
small hardware-friendly filter estimates utility
        ↓
admit candidate or suppress candidate
```

So the first action space is intentionally tiny:

```text
0 = suppress
1 = admit
```

This is cleaner than direct next-address prediction because the model does not need to search the full address space. It only judges whether an already-proposed candidate is worth spending cache/MSHR/bandwidth resources on.

## 2. What question are we answering first?

The first controlled question is:

```text
Among candidates SPP itself would issue to L2,
can a tiny filter suppress bad or resource-risky candidates
without killing useful coverage?
```

Current candidate scope:

```text
CANDIDATE_SCOPE = spp_l2_issue
condition       = spp_fill_l2 == 1 or spp_confidence >= 90
```

This keeps the first experiment focused. Later we can expand to all SPP lookahead candidates, LLC-only candidates, fill-level control, or degree control.

## 3. Input and output of the notebook

Notebook input:

```text
projects/post_prefetch_filter/data/generated/spp_candidate_log.csv.xz
```

Each row is one scoped SPP candidate with candidate-time features and delayed outcome labels.

Optional baseline input:

```text
projects/post_prefetch_filter/results/l2_spp_stats_25m/l2_spp_stats_25m_summary.csv
```

This contains aggregate ChampSim metrics such as IPC, hit rate, miss rate, MPKI, and prefetch accuracy.

Notebook outputs:

```text
projects/post_prefetch_filter/results/feature_sweep_summary.csv
projects/post_prefetch_filter/results/feature_sweep_summary.png
projects/post_prefetch_filter/results/policy_table_<feature_set>.json
```

## 4. What the notebook measures

The notebook does not directly prove IPC improvement. It is an offline feature-sweep tool that decides which policy is worth replaying in ChampSim.

Candidate-level metrics:

```text
issued_ratio        admitted candidates / candidate rows
accuracy            useful admitted / admitted
useful_kept_ratio   useful admitted / useful available
bad_suppressed      useless candidates suppressed
estimated_reward    offline utility proxy
```

How to read these metrics:

```text
accuracy goes up, useful_kept_ratio drops a lot
  -> filter is probably over-suppressing; IPC may drop

issued_ratio drops, accuracy remains high, useful_kept_ratio remains high
  -> promising traffic reduction

F1 MSHR/PQ changes the policy only when MSHR/PQ pressure exists
  -> resource-aware filtering is workload dependent

F3 bandwidth does not change anything while bandwidth_bucket is constant
  -> need real bandwidth logging before claiming bandwidth awareness
```

## 5. Final ChampSim metrics

Final evaluation must replay selected policies in ChampSim.

For each workload and policy, record:

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

The important performance question is:

```text
Does the filter improve or preserve IPC while reducing useless traffic/resource pressure?
```

Accuracy alone is not enough. A filter can look accurate by suppressing too much, but that may hurt coverage and IPC.

## 6. Feature sweep structure

The notebook compares feature sets in a controlled order:

```text
F0_candidate
  pc_bucket, delta_bucket, confidence_bucket

F1_candidate_mshr_pq
  F0 + mshr_bucket + pq_bucket

F2_add_recent_accuracy
  F1 + recent_spp_accuracy + recent_pc_accuracy + recent_delta_accuracy

F3_add_bandwidth
  F2 + bandwidth_bucket

F4_add_cache_pressure
  F3 + cache_hit + set_pressure
```

Only one group is added at a time. This makes the interpretation clear.

## 7. Workload-specific interpretation

The point is not to force one universal filter to work everywhere. The point is to discover which workload/phase needs which behavior.

Expected roles:

```text
602.gcc
  sanity / high SPP usefulness / low MSHR pressure
  expected behavior: mostly trust SPP

605.mcf
  likely resource-pressure workload
  expected behavior: MSHR/PQ-aware gating may matter

619.lbm
  streaming / coverage-sensitive workload
  expected behavior: avoid killing coverage
```

## 8. Creative architecture direction

The emerging design is a behavior-class controller:

```text
TRUST_SPP
  SPP accuracy high, resource pressure low
  -> admit almost all

RESOURCE_GATE
  MSHR/PQ/bandwidth pressure high
  -> suppress resource-risky candidates

COVERAGE_PROTECT
  streaming/high-coverage phase
  -> avoid over-filtering

DUPLICATE_CLEANUP
  many repeated candidates
  -> use a small recent-exact duplicate filter
```

The feature sweep is therefore not just model training. It is a way to discover workload behavior and decide what hardware control logic is worth building.
